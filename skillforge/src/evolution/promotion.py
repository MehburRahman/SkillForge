"""Recoverable prompt decisions and training-only regression monitoring."""

from collections import defaultdict
from dataclasses import asdict
from fractions import Fraction
import json

from src.benchmark.runner import _fingerprint
from src.evaluation.statistics import compare_paired_outcomes
from src.evolution.models import PromptValidationResult
from src.evolution.prompt_registry import EvolutionBusyError, load_default_registry


ROLLBACK_RUNS = 20
ROLLBACK_DROP = Fraction(1, 20)  # Strictly more than five percentage points.
_COHORT_FIELDS = (
    "model_name", "judge_model", "judge_enabled", "config_name", "seed", "language",
    "execution_mode", "attempt_budget", "attempt_budget_source", "test_source", "reasoning_mode",
)


def _snapshot(registry):
    return [asdict(v) for v in registry.versions]


def _training_outcomes(store, prompt_id, **kwargs):
    for row in store.rollback_training_runs(prompt_id, **kwargs):
        grade = json.loads(row["functional_grade_json"])
        score = grade.get("score")
        if (type(score) not in (int, float) or not 0 <= score <= 1
                or type(grade.get("passed_all")) is not bool):
            continue
        cohort = {key: row[key] for key in _COHORT_FIELDS}
        cohort['reasoning_mode'] = cohort['reasoning_mode'] or 'direct'
        manifest = json.loads(row["benchmark_manifest_json"]) if row["benchmark_manifest_json"] else None
        cohort["benchmark_context"] = (_fingerprint({key: manifest.get(key) for key in (
            "dataset_hash", "prompt_templates", "evaluation_policy", "analyzer", "seed_policy",
        )}) if manifest is not None else None)
        yield row, cohort, bool(grade["passed_all"] and score == 1
                               and row["hard_gate_passed"] and row["evaluation_status"] != "fail")


def _rollback_reference(store, parent_id, manifest):
    groups = defaultdict(list)
    for row, cohort, passed in _training_outcomes(store, parent_id):
        if (row["model_name"] != manifest["generator_model"]
                or (row["judge_model"] if row["judge_enabled"] else None) != manifest["judge_model"]
                or cohort["reasoning_mode"] != manifest.get("reasoning_mode", "direct")):
            continue
        groups[json.dumps(cohort, sort_keys=True)].append((row, passed))
    eligible = [(key, rows[-ROLLBACK_RUNS:]) for key, rows in groups.items() if len(rows) >= ROLLBACK_RUNS]
    reference = {"run_ids": [], "watermark": store.training_run_watermark(),
                 "metric": "visible-functional-and-hard-gates", "window": ROLLBACK_RUNS,
                 "drop_threshold": float(ROLLBACK_DROP)}
    if not eligible:
        reference["reason"] = "Need 20 completed parent training runs with comparable settings."
        return reference
    key, rows = max(eligible, key=lambda item: item[1][-1][0]["run_sequence"])
    reference.update(cohort=json.loads(key), run_ids=[r["run_id"] for r, _ in rows],
                     passed=sum(passed for _, passed in rows))
    return reference


def _finish_action(store, registry, action):
    evidence = json.loads(action["evidence_json"])
    if evidence["registry_path"] != str(registry.path.resolve()):
        raise ValueError("prompt action belongs to a different registry")
    registry.refresh()
    current = _snapshot(registry)
    before, after = json.loads(action["before_json"]), json.loads(action["after_json"])
    if current == before:
        child = evidence["child_prompt_id"]
        if action["kind"] == "promote":
            registry.activate(child, validation_metrics=evidence["validation_metrics"])
        elif action["kind"] == "reject":
            registry.reject(child, validation_metrics=evidence["validation_metrics"])
        else:
            registry.rollback(child)
        registry.refresh()
        current = _snapshot(registry)
    if current != after:
        raise ValueError("registry changed outside the pending action; refusing to overwrite it")
    return store.finish_prompt_action(action["action_id"])


def recover_prompt_actions(store, registry):
    """Reconcile a crash before/after registry replacement without reapplying history."""
    with registry.evolution_lock():
        return [_finish_action(store, registry, action) for action in store.pending_prompt_actions()]


def apply_validation(store, registry, validation_id):
    """Apply only a completed, persisted paired validation; retries are idempotent."""
    with registry.evolution_lock():
        recover_prompt_actions(store, registry)
        saved = store.get_prompt_validation(validation_id)
        if saved is None or saved["completed_at"] is None or saved["result_json"] is None:
            raise ValueError("promotion requires a completed saved validation")
        result = PromptValidationResult.from_dict(json.loads(saved["result_json"]))
        manifest = json.loads(saved["manifest_json"])
        baseline = {r.task_id: r.baseline_passed for r in result.task_outcomes}
        candidate = {r.task_id: r.candidate_passed for r in result.task_outcomes}
        if (result.validation_id != validation_id or len(baseline) != len(result.task_outcomes)
                or sorted(baseline) != manifest["baseline"]["task_ids"]
                or manifest["baseline"]["task_ids"] != manifest["candidate"]["task_ids"]
                or result.comparison != compare_paired_outcomes(baseline, candidate)
                or result.baseline_prompt_id != manifest["baseline"]["prompt_id"]
                or result.candidate_prompt_id != manifest["candidate"]["prompt_id"]):
            raise ValueError("saved validation evidence is inconsistent")
        kind = result.comparison.decision
        action_id = f"{validation_id}:decision"
        existing = store.prompt_action(action_id)
        if existing is not None:
            if json.loads(existing["evidence_json"])["registry_path"] != str(registry.path.resolve()):
                raise ValueError("prompt action belongs to a different registry")
            return existing
        registry.refresh()
        parent, child = registry.get(result.baseline_prompt_id), registry.get(result.candidate_prompt_id)
        if parent.status != "active" or child.status != "candidate" or child.parent_prompt_id != parent.prompt_id:
            raise ValueError("validation is stale: requires the active parent and its candidate")
        if any(_fingerprint(prompt.prompt_text) != manifest[name]["prompt_hash"]
               for name, prompt in (("baseline", parent), ("candidate", child))):
            raise ValueError("prompt text changed since validation")
        metrics = {"baseline_pass_rate": result.comparison.baseline_pass_rate,
                   "pass_rate": result.comparison.candidate_pass_rate,
                   "task_count": result.comparison.task_count, "p_value": result.comparison.p_value,
                   "candidate_only_passes": result.comparison.candidate_only_passes,
                   "baseline_only_passes": result.comparison.baseline_only_passes}
        before = _snapshot(registry)
        after = json.loads(json.dumps(before))
        for version in after:
            if version["prompt_id"] == child.prompt_id:
                version.update(status="active" if kind == "promote" else "rejected", validation_metrics=metrics)
            elif version["prompt_id"] == parent.prompt_id and kind == "promote":
                version["status"] = "archived"
        evidence = {"registry_path": str(registry.path.resolve()), "parent_prompt_id": parent.prompt_id,
                    "child_prompt_id": child.prompt_id, "validation_metrics": metrics,
                    "reason": result.comparison.reason}
        if kind == "promote":
            evidence["rollback_reference"] = _rollback_reference(store, parent.prompt_id, manifest["baseline"])
        action = store.prepare_prompt_action(action_id, validation_id, kind, before, after, evidence)
        return _finish_action(store, registry, action)


def check_rollbacks(store, registry):
    """Use exactly the first 20 comparable post-promotion completed training runs.

    The frozen parent reference uses its most recent 20 matching runs. REVIEW
    alone is not a regression: correctness uses visible tests and hard gates.
    Existing benchmark sweeps stay pinned to their original prompt; restoring
    the parent changes the default for future runs, never a sweep's settings.
    """
    with registry.evolution_lock():
        recover_prompt_actions(store, registry)
        registry.refresh()
        reports = []
        for monitor in store.rollback_monitors(watching_only=True):
            validation_id = monitor["validation_id"]
            if registry.active_prompt.prompt_id != monitor["child_prompt_id"]:
                report = {"status": "superseded", "reason": "A later prompt is active."}
                store.update_rollback_monitor(validation_id, "superseded", report)
                reports.append(report)
                continue
            reference = json.loads(monitor["reference_json"])
            # Saved monitors from before reasoning modes used direct solves.
            reference['cohort'].setdefault('reasoning_mode', 'direct')
            outcomes = []
            for row, cohort, passed in _training_outcomes(
                store, monitor["child_prompt_id"], after=monitor["started_at"], watermark=reference["watermark"],
            ):
                if cohort == reference["cohort"]:
                    outcomes.append((row["run_id"], passed))
                    if len(outcomes) == ROLLBACK_RUNS:
                        break
            report = {"validation_id": validation_id, "status": "watching", "runs": len(outcomes),
                      "required_runs": ROLLBACK_RUNS, "run_ids": [key for key, _ in outcomes]}
            if len(outcomes) == ROLLBACK_RUNS:
                parent_rate = Fraction(reference["passed"], ROLLBACK_RUNS)
                child_rate = Fraction(sum(passed for _, passed in outcomes), ROLLBACK_RUNS)
                rollback = parent_rate - child_rate > ROLLBACK_DROP
                report.update(status="rolled_back" if rollback else "kept",
                              parent_pass_rate=float(parent_rate), child_pass_rate=float(child_rate),
                              drop=float(parent_rate - child_rate))
                if rollback:
                    before = _snapshot(registry)
                    after = json.loads(json.dumps(before))
                    for version in after:
                        if version["prompt_id"] == monitor["child_prompt_id"]:
                            version["status"] = "rolled_back"
                        elif version["prompt_id"] == monitor["parent_prompt_id"]:
                            version["status"] = "active"
                    evidence = {"registry_path": str(registry.path.resolve()), "result": report,
                                "child_prompt_id": monitor["child_prompt_id"]}
                    action = store.prepare_prompt_action(f"{validation_id}:rollback", validation_id,
                                                         "rollback", before, after, evidence)
                    _finish_action(store, registry, action)
                    reports.append(report)
                    continue
            store.update_rollback_monitor(validation_id, report["status"], report)
            reports.append(report)
        return reports


def monitor_training_completion(store, registry, split):
    """Cheap hook shared by standalone training and benchmark completion."""
    if split != "train" or not store.rollback_monitors(watching_only=True):
        return []
    registry = registry if registry is not None else load_default_registry()
    try:
        return check_rollbacks(store, registry)
    except EvolutionBusyError:
        # An evolution round holds the lock. Its next check will catch up using
        # the first 20 saved outcomes, even if more tasks completed meanwhile.
        return []
