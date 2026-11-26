import json
import sys

import pytest

import run_evolution
from src.history import ExperimentStore
from tests.test_prompt_evolver import registry
from tests.test_benchmark_runner import write_dataset


def arguments(tmp_path, *extra):
    prompts = registry(tmp_path)
    dataset = tmp_path / "tasks.json"
    write_dataset(dataset)
    return ["run_evolution.py", str(dataset), "--prompt-registry", str(prompts.path),
            "--history-db", str(tmp_path / "history.db"), *extra]


def test_cli_records_blocked_run_without_ollama(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "argv", arguments(tmp_path, "--run-id", "blocked"))
    run_evolution.main()
    text = capsys.readouterr().out
    assert "Evolution run ID: blocked" in text
    assert '"status": "blocked"' in text
    with ExperimentStore(tmp_path / "history.db") as store:
        saved = store.get_evolution_round("blocked")
        manifest = json.loads(saved["manifest_json"])
        assert manifest["baseline"]["max_attempts"] == 3
        assert manifest["baseline"]["config_name"] == "full"
        assert manifest["baseline"]["seed"] == 42


@pytest.mark.parametrize("flags", [
    ["--resume"], ["--max-attempts", "0"], ["--seed", "-1"], ["--proposal-attempts", "0"],
    ["--check-rollback-only", "--run-id", "x"],
])
def test_bad_options_fail_before_history_creation(monkeypatch, tmp_path, flags):
    monkeypatch.setattr(sys, "argv", arguments(tmp_path, *flags))
    with pytest.raises(SystemExit) as error:
        run_evolution.main()
    assert error.value.code == 1
    assert not (tmp_path / "history.db").exists()


def test_rollback_only_needs_no_dataset_or_models(monkeypatch, tmp_path, capsys):
    argv = arguments(tmp_path, "--check-rollback-only")
    argv[1] = "missing-dataset.json"
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(run_evolution, "build_clients", lambda *a: pytest.fail("model client created"))
    run_evolution.main()
    assert '"active_prompt_id": "P0"' in capsys.readouterr().out


def test_cli_forwards_models_and_resume_options(monkeypatch, tmp_path, capsys):
    captured = {}
    def run(client, dataset, prompts, store, **kwargs):
        captured.update(kwargs, client=client)
        return {"status": "complete"}
    monkeypatch.setattr(run_evolution, "run_evolution_round", run)
    monkeypatch.setattr(sys, "argv", arguments(tmp_path, "--model", "qwen2.5-coder:7b",
        "--judge-model", "llama3.2:3b", "--meta-model", "qwen3:4b", "--seed", "7",
        "--max-attempts", "2", "--run-id", "round-1", "--resume"))
    run_evolution.main()
    assert captured["judge_client"].model == "llama3.2:3b"
    assert captured["meta_client"].model == "qwen3:4b"
    assert captured["seed"] == 7 and captured["max_attempts"] == 2
    assert captured["resume"] and captured["round_id"] == "round-1"
