from dataclasses import replace
import json

import pytest

from src.benchmark.models import BenchmarkDataset
from src.evolution.workflow import run_evolution_round
from src.history import ExperimentStore, ExperimentStoreError
from tests.test_benchmark_runner import BenchmarkClient, write_dataset
from tests.test_prompt_evolver import MetaClient, PROPOSED_PROMPT, add_runs, registry


class SeededMetaClient(MetaClient):
    model = "meta-model"

    def generate(self, prompt, system=None, temperature=0.2, *, seed=None):
        return super().generate(prompt, system=system, temperature=temperature)


class EvolutionClient(BenchmarkClient):
    def generate(self, prompt, system=None, temperature=0.2, *, seed=None):
        if "problem analyzer" in (system or "").lower():
            return super().generate(prompt, system=system, temperature=temperature, seed=seed)
        self.requests.append({"prompt": prompt, "system": system, "seed": seed})
        return "a, b = map(int, input().split())\nprint(a + b)" if PROPOSED_PROMPT in system else "print(0)"


@pytest.fixture
def workflow(tmp_path):
    original = write_dataset(tmp_path / "tasks.json")
    task = original.tasks[1]
    return dict(client=EvolutionClient(), meta_client=SeededMetaClient(),
                dataset=BenchmarkDataset(tasks=(original.tasks[0], *(
                    replace(task, task_id=f"v{i}", problem_statement=f"Task {i}: {task.problem_statement}")
                    for i in range(4)))), registry=registry(tmp_path),
                seed=42, max_attempts=2, round_id="round-1")


def test_full_round_proposes_validates_promotes_and_resume_makes_no_calls(tmp_path, workflow):
    with ExperimentStore(tmp_path / "history.db") as store:
        add_runs(store)
        result = run_evolution_round(store=store, **workflow)
        assert result["status"] == "complete" and result["decision"] == "promoted"
        assert result["active_prompt_id"] == "P1"
        assert result["comparison"]["p_value"] == 0.0625
        assert store.get_evolution_round("round-1")["meta_model_calls"] == 1
        requests = len(workflow["client"].requests)
        assert run_evolution_round(store=store, resume=True, **workflow) == result
        assert len(workflow["client"].requests) == requests
        assert len(workflow["meta_client"].requests) == 1
        assert len(store.prompt_experiments()) == 1


def test_not_enough_evidence_saves_blocked_round_without_model_calls(tmp_path, workflow):
    with ExperimentStore(tmp_path / "history.db") as store:
        result = run_evolution_round(store=store, **workflow)
        assert result["status"] == "blocked"
        assert "0/20" in result["reason"]
        assert run_evolution_round(store=store, resume=True, **workflow) == result
    assert workflow["client"].requests == workflow["meta_client"].requests == []


def test_resume_interrupted_pair_does_not_propose_twice(tmp_path, workflow):
    def interrupt(message):
        if message.startswith("P1 v0:"):
            raise KeyboardInterrupt()
    path = tmp_path / "history.db"
    with ExperimentStore(path) as store:
        add_runs(store)
        with pytest.raises(KeyboardInterrupt):
            run_evolution_round(store=store, on_progress=interrupt, **workflow)
        assert store.get_evolution_round("round-1")["status"] == "validating"
        assert workflow["registry"].active_prompt.prompt_id == "P0"
        with pytest.raises(ExperimentStoreError, match="unfinished"):
            run_evolution_round(store=store, **dict(workflow, round_id="other-round"))
    with ExperimentStore(path) as store:
        assert run_evolution_round(store=store, resume=True, **workflow)["decision"] == "promoted"
        assert store._connection.execute("SELECT COUNT(*) FROM runs WHERE benchmark_split='validation'").fetchone()[0] == 8
    assert len(workflow["meta_client"].requests) == 1


def test_resume_after_registry_action_before_round_completion(tmp_path, workflow, monkeypatch):
    with ExperimentStore(tmp_path / "history.db") as store:
        add_runs(store)
        original = store.update_evolution_round
        def crash(round_id, **kwargs):
            if kwargs["status"] == "complete":
                raise RuntimeError("crash after action")
            return original(round_id, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(store, "update_evolution_round", crash)
            with pytest.raises(RuntimeError, match="after action"):
                run_evolution_round(store=store, **workflow)
        assert workflow["registry"].active_prompt.prompt_id == "P1"
        requests = len(workflow["client"].requests)
        result = run_evolution_round(store=store, resume=True, **workflow)
        assert result["decision"] == "promoted"
        assert len(workflow["client"].requests) == requests


def test_proposer_failure_is_recorded_and_resume_does_not_spend_more_calls(tmp_path, workflow):
    class FailingMeta:
        model = "meta-model"
        def generate(self, *a, **kw):
            raise RuntimeError("model offline")
    with ExperimentStore(tmp_path / "history.db") as store:
        add_runs(store)
        with pytest.raises(RuntimeError, match="offline"):
            run_evolution_round(store=store, **dict(workflow, meta_client=FailingMeta()))
        saved = store.get_evolution_round("round-1")
        assert saved["trigger_id"] and saved["meta_model_calls"] == 1
        assert saved["status"] == "failed"
        assert "model offline" in json.loads(saved["result_json"])["reason"]
        result = run_evolution_round(store=store, resume=True, **workflow)
        assert result["status"] == "failed"
        assert len(workflow["meta_client"].requests) == 0
        # A finished failure must not masquerade as an unfinished round and
        # block a later eligibility check. It still consumes its old evidence.
        next_result = run_evolution_round(store=store, **dict(workflow, round_id="next-round"))
        assert next_result["status"] == "blocked" and "cooldown" in next_result["reason"]


def test_unchanged_live_style_proposals_fail_without_registering_or_validating(tmp_path, workflow):
    class UnchangedMeta(SeededMetaClient):
        def generate(self, *args, **kwargs):
            payload = json.loads(super().generate(*args, **kwargs))
            payload["proposed_prompt"] = workflow["registry"].active_prompt.prompt_text
            return json.dumps(payload)
    meta = UnchangedMeta()
    with ExperimentStore(tmp_path / "history.db") as store:
        add_runs(store)
        with pytest.raises(ValueError, match="must differ"):
            run_evolution_round(store=store, **dict(workflow, meta_client=meta))
        saved = store.get_evolution_round("round-1")
        assert saved["status"] == "failed" and saved["meta_model_calls"] == 2
        assert "must differ" in json.loads(saved["result_json"])["reason"]
        assert store.get_prompt_validation(saved["validation_id"]) is None
        assert store.pending_prompt_actions() == []
        assert len(workflow["registry"].versions) == 1


def test_reservation_and_round_link_are_atomic(tmp_path, workflow, monkeypatch):
    with ExperimentStore(tmp_path / "history.db") as store:
        add_runs(store)
        original = store.update_evolution_round
        def fail_link(round_id, **kwargs):
            if kwargs.get("trigger_id"):
                raise RuntimeError("link write failed")
            return original(round_id, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(store, "update_evolution_round", fail_link)
            with pytest.raises(RuntimeError, match="link write failed"):
                run_evolution_round(store=store, **workflow)
        assert store.evolution_checks() == []
        assert store.evolution_evidence("P0")["training_runs"] == 20
        assert run_evolution_round(store=store, resume=True, **workflow)["decision"] == "promoted"


@pytest.mark.parametrize("options", [{"seed": 43}, {"max_attempts": 3}, {"max_observations": 10}])
def test_resume_rejects_changed_settings_without_new_calls(tmp_path, workflow, options):
    with ExperimentStore(tmp_path / "history.db") as store:
        run_evolution_round(store=store, **workflow)
        with pytest.raises(ExperimentStoreError, match="changed"):
            run_evolution_round(store=store, resume=True, **dict(workflow, **options))
    assert workflow["meta_client"].requests == []
