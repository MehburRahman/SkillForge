from dataclasses import replace
import json
import sqlite3

import pytest

from src.benchmark.models import BenchmarkDataset
from src.benchmark.runner import benchmark_manifest, run_benchmark
from src.evolution.promotion import apply_validation, check_rollbacks, recover_prompt_actions
from src.evolution.prompt_registry import EvolutionBusyError, PromptRegistry, PromptRegistryError
from src.evolution.validator import validate_candidate_prompt
from src.history import ExperimentStore
from tests.test_benchmark_runner import BenchmarkClient
from tests.test_prompt_evolver import add_runs
from tests.test_prompt_validator import ValidationClient, setup


def training(store, count=20, passed=20, *, prompt="P0", split="train", **overrides):
    ids = add_runs(store, count, prompt=prompt, split=split)
    with store._connection:
        for i, key in enumerate(ids):
            success = i < passed
            settings = dict(model_name=BenchmarkClient.model, config_name="full", seed=91,
                            language="python", attempt_budget=2, attempt_budget_source="manual_override",
                            best_attempt_number=1, final_status="review" if success else "fail")
            settings.update(overrides)
            store._connection.execute(
                "UPDATE runs SET completed_at = strftime('%Y-%m-%dT%H:%M:%fZ','now'), "
                + ", ".join(f"{k} = ?" for k in settings) + " WHERE run_id = ?", (*settings.values(), key),
            )
            store._connection.execute(
                """UPDATE attempts SET model_name = ?, hard_gate_passed = ?, evaluation_status = ?,
                   functional_grade_json = ? WHERE run_id = ?""",
                (settings["model_name"], int(success), "review" if success else "fail",
                 json.dumps({"score": float(success), "passed_all": success}), key),
            )
    return ids


def validate(store, setup, client=None):
    return validate_candidate_prompt(client or ValidationClient(), history_store=store, **setup)


def test_promote_saved_decision_and_preserve_audit_on_retry(tmp_path, setup):
    prompts = setup["prompt_registry"]
    with ExperimentStore(tmp_path / "history.db") as store:
        baseline = training(store)
        result = validate(store, setup)
        action = apply_validation(store, prompts, result.validation_id)
        again = apply_validation(store, prompts, result.validation_id)
        assert action == again and action["status"] == "applied"
        assert store.prompt_experiments()[0]["decision"] == "promote"
        monitor, = store.rollback_monitors()
        assert monitor["status"] == "watching"
        assert set(json.loads(monitor["reference_json"])["run_ids"]) == set(baseline)
    assert prompts.active_prompt.prompt_id == "P1"
    assert prompts.get("P0").status == "archived"
    assert prompts.get("P1").validation_metrics["p_value"] == 0.0625


def test_direct_promotion_reference_ignores_planned_parent_runs_added_after_validation(tmp_path, setup):
    prompts = setup['prompt_registry']
    with ExperimentStore(tmp_path / 'history.db') as store:
        direct_runs = training(store)
        result = validate(store, setup)
        training(store, passed=0, reasoning_mode='plan_critique')

        action = apply_validation(store, prompts, result.validation_id)
        reference = json.loads(action['evidence_json'])['rollback_reference']

        assert reference['cohort']['reasoning_mode'] == 'direct'
        assert reference['run_ids'] == direct_runs
        assert reference['passed'] == len(direct_runs)


def test_rejection_keeps_parent_and_is_idempotent(tmp_path, setup):
    with ExperimentStore(tmp_path / "history.db") as store:
        result = validate(store, setup, BenchmarkClient())
        action = apply_validation(store, setup["prompt_registry"], result.validation_id)
        assert action["kind"] == "reject"
        assert store.rollback_monitors() == []
        assert apply_validation(store, setup["prompt_registry"], result.validation_id) == action
    assert setup["prompt_registry"].active_prompt.prompt_id == "P0"
    assert setup["prompt_registry"].get("P1").status == "rejected"


@pytest.mark.parametrize("point", ["registry", "database"])
def test_recover_action_across_process_restart(tmp_path, setup, monkeypatch, point):
    path = tmp_path / "history.db"
    prompts = setup["prompt_registry"]
    with ExperimentStore(path) as store:
        training(store)
        result = validate(store, setup)
        target, method = (prompts, "activate") if point == "registry" else (store, "finish_prompt_action")
        with monkeypatch.context() as patch:
            patch.setattr(target, method, lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("crash")))
            with pytest.raises(RuntimeError, match="crash"):
                apply_validation(store, prompts, result.validation_id)
        assert len(store.pending_prompt_actions()) == 1
    with ExperimentStore(path) as reopened:
        fresh = PromptRegistry(prompts.path)
        assert len(recover_prompt_actions(reopened, fresh)) == 1
        assert recover_prompt_actions(reopened, fresh) == []
        assert fresh.active_prompt.prompt_id == "P1"
        assert len(reopened.rollback_monitors()) == 1


def test_recovery_refuses_unrelated_registry_edits(tmp_path, setup, monkeypatch):
    prompts = setup["prompt_registry"]
    with ExperimentStore(tmp_path / "history.db") as store:
        result = validate(store, setup)
        with monkeypatch.context() as patch:
            patch.setattr(prompts, "activate", lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
            with pytest.raises(RuntimeError):
                apply_validation(store, prompts, result.validation_id)
        prompts.register_candidate("Unrelated edit", "Manual change")
        before = prompts.path.read_bytes()
        with pytest.raises(ValueError, match="refusing to overwrite"):
            recover_prompt_actions(store, prompts)
        assert prompts.path.read_bytes() == before


@pytest.mark.parametrize("passed,status", [(20, "kept"), (19, "kept"), (18, "rolled_back"), (0, "rolled_back")])
def test_rollback_strict_threshold_and_original_decision_is_not_reapplied(tmp_path, setup, passed, status):
    prompts = setup["prompt_registry"]
    with ExperimentStore(tmp_path / "history.db") as store:
        training(store)
        result = validate(store, setup)
        apply_validation(store, prompts, result.validation_id)
        training(store, 19, min(passed, 19), prompt="P1")
        assert check_rollbacks(store, prompts)[0]["status"] == "watching"
        assert prompts.active_prompt.prompt_id == "P1"
        training(store, 1, max(passed - 19, 0), prompt="P1")
        report, = check_rollbacks(store, prompts)
        assert report["status"] == status and report["runs"] == 20
        assert check_rollbacks(store, prompts) == []
        apply_validation(store, prompts, result.validation_id)
        assert prompts.active_prompt.prompt_id == ("P0" if status == "rolled_back" else "P1")
        if status == "rolled_back":
            assert prompts.get("P1").status == "rolled_back"
            assert prompts.rejected_duplicate(prompts.get("P1").prompt_text).prompt_id == "P1"
            with pytest.raises(PromptRegistryError, match="rolled_back"):
                prompts.activate("P1")


def test_monitor_ignores_other_splits_settings_and_preexisting_runs(tmp_path, setup):
    prompts = setup["prompt_registry"]
    with ExperimentStore(tmp_path / "history.db") as store:
        training(store)
        training(store, prompt="P1", passed=0)  # Before promotion.
        result = validate(store, setup)
        apply_validation(store, prompts, result.validation_id)
        for split in ("hidden", "validation", "adhoc"):
            training(store, prompt="P1", passed=0, split=split)
        training(store, prompt="P1", passed=0, seed=92)
        training(store, prompt="P1", passed=0, config_name="single_shot")
        training(store, prompt="P1", passed=0, model_name="different-model")
        assert check_rollbacks(store, prompts)[0]["runs"] == 0
        first_twenty = training(store, prompt="P1")
        training(store, prompt="P1", passed=0)  # Later failures cannot alter the first window.
        report, = check_rollbacks(store, prompts)
        assert report["status"] == "kept"
        assert report["run_ids"] == first_twenty


@pytest.mark.parametrize('legacy_reference', [False, True])
def test_direct_rollback_monitor_ignores_planned_training(tmp_path, setup, legacy_reference):
    prompts = setup['prompt_registry']
    with ExperimentStore(tmp_path / 'history.db') as store:
        training(store)
        result = validate(store, setup)
        apply_validation(store, prompts, result.validation_id)
        monitor, = store.rollback_monitors()
        reference = json.loads(monitor['reference_json'])
        assert reference['cohort']['reasoning_mode'] == 'direct'
        if legacy_reference:
            reference['cohort'].pop('reasoning_mode')
            with store._connection:
                store._connection.execute(
                    'UPDATE rollback_monitors SET reference_json = ? WHERE validation_id = ?',
                    (json.dumps(reference), result.validation_id),
                )

        training(store, prompt='P1', passed=0, reasoning_mode='plan_critique')
        report, = check_rollbacks(store, prompts)
        assert report['status'] == 'watching' and report['runs'] == 0
        assert prompts.active_prompt.prompt_id == 'P1'

        direct_runs = training(store, prompt='P1')
        report, = check_rollbacks(store, prompts)
        assert report['status'] == 'kept'
        assert report['run_ids'] == direct_runs


def test_insufficient_parent_reference_is_explicit(tmp_path, setup):
    with ExperimentStore(tmp_path / "history.db") as store:
        training(store, count=19, passed=19)
        result = validate(store, setup)
        apply_validation(store, setup["prompt_registry"], result.validation_id)
        assert store.rollback_monitors()[0]["status"] == "unavailable"
        assert check_rollbacks(store, setup["prompt_registry"]) == []


def test_rollback_recovers_if_database_commit_fails_after_registry_restore(tmp_path, setup, monkeypatch):
    path = tmp_path / "history.db"
    prompts = setup["prompt_registry"]
    with ExperimentStore(path) as store:
        training(store)
        result = validate(store, setup)
        apply_validation(store, prompts, result.validation_id)
        training(store, prompt="P1", passed=0)
        with monkeypatch.context() as patch:
            patch.setattr(store, "finish_prompt_action", lambda *a: (_ for _ in ()).throw(RuntimeError("crash")))
            with pytest.raises(RuntimeError, match="crash"):
                check_rollbacks(store, prompts)
        assert prompts.active_prompt.prompt_id == "P0"
        assert store.rollback_monitors()[0]["status"] == "watching"
    with ExperimentStore(path) as store:
        fresh = PromptRegistry(prompts.path)
        assert len(recover_prompt_actions(store, fresh)) == 1
        assert store.rollback_monitors()[0]["status"] == "rolled_back"
        assert fresh.get("P1").status == "rolled_back"
        assert check_rollbacks(store, fresh) == []


def test_rejects_incomplete_stale_or_changed_validation(tmp_path, setup):
    prompts = setup["prompt_registry"]
    with ExperimentStore(tmp_path / "history.db") as store:
        with pytest.raises(ValueError, match="completed"):
            apply_validation(store, prompts, "missing")
        result = validate(store, setup)
        prompts._write(replace(v, prompt_text="Modified") if v.prompt_id == "P1" else v for v in prompts.versions)
        with pytest.raises(ValueError, match="text changed"):
            apply_validation(store, prompts, result.validation_id)
        assert store.pending_prompt_actions() == []


def test_registry_lock_prevents_two_evolution_writers(setup):
    prompts = setup["prompt_registry"]
    other = PromptRegistry(prompts.path)
    with prompts.evolution_lock():
        with prompts.evolution_lock():
            with pytest.raises(EvolutionBusyError):
                with other.evolution_lock():
                    pytest.fail("second writer entered")
    with other.evolution_lock():
        pass


def test_completed_training_benchmark_automatically_checks_rollback(tmp_path, setup):
    prompts = setup["prompt_registry"]
    with ExperimentStore(tmp_path / "history.db") as store:
        task = replace(setup["dataset"].tasks[1], task_id="last-training", split="train")
        dataset = BenchmarkDataset(tasks=(task,))
        context = lambda prompt: benchmark_manifest(
            ValidationClient(), dataset, dataset.tasks, prompts.get(prompt), config_name="full", seed=91,
            max_attempts=2, language=None)
        training(store, benchmark_run_id=store.begin_benchmark(context("P0")))
        result = validate(store, setup)
        apply_validation(store, prompts, result.validation_id)
        training(store, count=19, passed=17, prompt="P1", benchmark_run_id=store.begin_benchmark(context("P1")))
        summary = run_benchmark(ValidationClient(), BenchmarkDataset(tasks=(task,)), prompt_registry=prompts,
                                history_store=store, config_name="full", max_attempts=2, seed=91)
        assert summary.prompt_version == "P1"  # The sweep itself never switches prompts.
        assert prompts.active_prompt.prompt_id == "P0"
        assert store.rollback_monitors()[0]["status"] == "rolled_back"


def test_schema_8_migration_preserves_validation(tmp_path, setup):
    path = tmp_path / "history.db"
    with ExperimentStore(path) as store:
        result = validate(store, setup)
    with sqlite3.connect(path) as db:
        db.executescript("DROP TABLE evolution_rounds; DROP TABLE rollback_monitors; DROP TABLE prompt_actions; "
                         "UPDATE metadata SET value='8' WHERE key='schema_version';")
    with ExperimentStore(path) as store:
        assert store.get_prompt_validation(result.validation_id)["completed_at"]
        assert store.pending_prompt_actions() == []
        assert apply_validation(store, setup["prompt_registry"], result.validation_id)["status"] == "applied"
