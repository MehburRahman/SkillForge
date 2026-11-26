"""One durable evolution round: training evidence -> validation -> decision."""

from src.evolution.scope import validate_evolution_scope

import json
import hashlib
from pathlib import Path
import time
from uuid import uuid4

from src.agents.llm_client import CallCountingClient, RecordingClient, validate_judge_pair
from src.agents.task_spec_cache import TaskSpecCache
from src.benchmark.runner import benchmark_manifest, select_benchmark_tasks
from src.evolution.promotion import apply_validation, check_rollbacks
from src.evolution.prompt_evolver import propose_candidate_prompt
from src.evolution.trigger import recover_evolution_attempt
from src.evolution.validator import validate_candidate_prompt
from src.experiments import preflight_execution, validate_seed


def run_evolution_round(client, dataset, registry, store, *, meta_client=None, judge_client=None,
                        seed=42, max_attempts=3, round_id=None, resume=False,
                        max_observations=50, proposal_attempts=2, task_spec_cache=None,
                        on_progress=None, retry_proposal=False):
    """Resume reserved work; completed rounds never propose or validate again.

    If a proposer dies before saving a candidate, recovery records that failed
    attempt instead of inventing a candidate or silently reusing its evidence.
    """
    validate_seed(seed)
    if seed is None:
        raise ValueError("evolution requires an explicit seed")
    for name, value in (("max_attempts", max_attempts), ("max_observations", max_observations),
                        ("proposal_attempts", proposal_attempts)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if resume and not round_id:
        raise ValueError("--resume requires --run-id")
    if retry_proposal and not resume:
        raise ValueError('--retry-proposal requires --resume and the original --run-id')
    if round_id is not None and (not isinstance(round_id, str) or not round_id.strip()):
        raise ValueError("round ID must be a non-empty string")
    validate_judge_pair(client, judge_client)
    tasks = select_benchmark_tasks(dataset, splits=["validation"])
    language, execution_mode = validate_evolution_scope(tasks, store=store)
    for task in tasks:
        preflight_execution(task.language, task.execution_mode)
    round_id = round_id or str(uuid4())
    meta_client = meta_client or client
    cache = task_spec_cache if task_spec_cache is not None else TaskSpecCache(store.path.parent / "task_spec_cache")

    def progress(message):
        if on_progress is not None:
            on_progress(message)

    def finish_without_candidate(trigger):
        status = "failed" if trigger["status"] in {"failed", "cancelled"} else "blocked"
        result = {"round_id": round_id, "status": status, "reason": trigger["reason"],
                  "trigger_id": trigger["decision_id"]}
        store.update_evolution_round(round_id, status=status, result=result)
        return result

    with registry.evolution_lock():
        check_rollbacks(store, registry)
        registry.refresh()
        saved = store.get_evolution_round(round_id)
        baseline_id = (json.loads(saved["manifest_json"])["baseline"]["prompt_id"]
                       if saved else registry.active_prompt.prompt_id)
        baseline = registry.get(baseline_id)
        manifest = {
            "protocol": 2, "registry_path": str(registry.path.resolve()),
            "proposal_policy": hashlib.sha256((Path(__file__).parents[1] / 'agents' / 'metaprompt_agent.py').read_bytes()).hexdigest(),
            "baseline": benchmark_manifest(
                client, dataset, tasks, baseline, config_name="full", seed=seed,
                max_attempts=max_attempts, judge_client=judge_client, task_spec_cache=cache,
            ),
            "meta_model": getattr(meta_client, "model", "unknown"),
            "max_observations": max_observations, "proposal_attempts": proposal_attempts,
        }
        saved = store.begin_evolution_round(round_id, manifest, resume=resume)
        progress(f"Evolution run ID: {round_id}; stage: {saved['status']}")
        if retry_proposal and saved['status'] != 'failed':
            raise ValueError('Only a failed proposal round can be retried')
        if saved["status"] in {"complete", "blocked", "failed"} and not retry_proposal:
            return json.loads(saved["result_json"])
        if saved["trigger_id"] is None or retry_proposal:
            counter = CallCountingClient(RecordingClient(meta_client, store, round_id, 'meta'), seed=seed, task_key=round_id)
            counter.set_stage("prompt_evolution_retry" if retry_proposal else "prompt_evolution")
            started = time.perf_counter()
            progress("Checking training evidence and proposing a candidate.")
            try:
                propose_candidate_prompt(
                    counter, registry, store, prompt_id=baseline_id,
                    max_observations=max_observations, max_attempts=proposal_attempts,
                    on_trigger=lambda key: store.update_evolution_round(
                        round_id, status="proposing", trigger_id=key),
                    language=language, execution_mode=execution_mode,
                    retry_decision_id=saved['trigger_id'] if retry_proposal else None,
                )
            except BaseException:
                # The proposer records known failures before propagating them.
                # Finalize the round immediately; only genuinely interrupted
                # reservations/candidates remain resumable unfinished work.
                interrupted = store.get_evolution_round(round_id)
                if interrupted["trigger_id"] is not None:
                    trigger = store.get_evolution_check(interrupted["trigger_id"])
                    if trigger["status"] in {"failed", "cancelled"}:
                        finish_without_candidate(trigger)
                raise
            finally:
                stage = store.get_evolution_round(round_id)["status"]
                store.update_evolution_round(round_id, status=stage, meta_calls=counter.call_count,
                                             meta_seconds=time.perf_counter() - started)
            saved = store.get_evolution_round(round_id)
        trigger = store.get_evolution_check(saved["trigger_id"])
        if trigger["status"] == "started":
            trigger = recover_evolution_attempt(store, registry, trigger["decision_id"])
        if trigger["status"] != "candidate":
            return finish_without_candidate(trigger)
        candidate_id = trigger["candidate_prompt_id"]
        validation_id = saved["validation_id"]
        # A crash after applying the registry action must not send an already
        # active/rejected candidate back into the validator.
        action = store.prompt_action(f"{validation_id}:decision")
        if action is None:
            store.update_evolution_round(round_id, status="validating", candidate_prompt_id=candidate_id)
            progress(f"Validating {candidate_id} against {baseline_id}; validation ID: {validation_id}")
            validate_candidate_prompt(
                client, dataset, registry, store, candidate_id, baseline_prompt_id=baseline_id,
                seed=seed, max_attempts=max_attempts, judge_client=judge_client,
                task_spec_cache=cache, validation_id=validation_id,
                resume=store.get_prompt_validation(validation_id) is not None,
                on_task_complete=lambda row: progress(
                    f"{row.prompt_version} {row.task_id}: visible={row.final_visible_score:.2f}, "
                    f"hidden={row.hidden_score:.2f}, attempts={row.attempt_count}"),
            )
        store.update_evolution_round(round_id, status="applying", candidate_prompt_id=candidate_id)
        action = apply_validation(store, registry, validation_id)
        validation = json.loads(store.get_prompt_validation(validation_id)["result_json"])
        result = {"round_id": round_id, "status": "complete", "baseline_prompt_id": baseline_id,
                  "candidate_prompt_id": candidate_id, "validation_id": validation_id,
                  "decision": "promoted" if action["kind"] == "promote" else "rejected",
                  "comparison": validation["comparison"], "active_prompt_id": registry.active_prompt.prompt_id}
        if action["kind"] == "promote":
            result["rollback_reference"] = json.loads(action["evidence_json"])["rollback_reference"]
        store.update_evolution_round(round_id, status="complete", result=result)
        progress(f"{candidate_id} {result['decision']}; active prompt: {result['active_prompt_id']}")
        return result
