"""Isolated HTTP fixture for frontend browser tests; never uses project data."""

from pathlib import Path
import argparse
import json
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn

from api.config import Settings
from api.main import create_app
from src.history import ExperimentStore
from src.reinforcement_loop import run_candidate_refinement_loop
from tests.test_benchmark_runner import BenchmarkClient, make_registry, write_dataset


class FixtureClient(BenchmarkClient):
    def __init__(self, model):
        super().__init__()
        self.requested_model = model

    def generate(self, prompt, system=None, temperature=0.2, *, seed=None, max_output_tokens=None):
        time.sleep(0.25)
        if self.requested_model == "unavailable-model":
            raise RuntimeError("The requested model is unavailable.")
        if 'Design a SMALL-INPUT correctness reference' in (system or ''):
            return 'Parse two whitespace-separated integers and return their sum as decimal text.'
        if 'input-construction function' in (system or ''):
            return 'def build_inputs():\n    return [dict(input="2 3", category="normal")]'
        if 'Write ONLY Python 3 source' in (system or ''):
            return 'def solve_input(input_text: str) -> str:\n    a,b = map(int,input_text.split())\n    return str(a+b)'
        if "review generated programming tests" in (system or ""):
            return json.dumps([dict(test_number=1, input_valid=True, output_verified=True,
                expected_output='5', category_valid=True, reason='Two plus three equals five.')])
        if "generate executable test cases" in (system or ""):
            return json.dumps([{"input": "2 3", "expected_output": "5", "category": "normal"}])
        return super().generate(prompt, system=system, temperature=temperature, seed=seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=18765)
    parser.add_argument('--frontend-origin')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="skillforge-browser-") as directory:
        root = Path(directory)
        registry = make_registry(root / "prompts.json")
        dataset = write_dataset(root / "tasks.json")
        settings = Settings(history_db=root / "history.sqlite3", registry=registry.path,
                            jobs_db=root / "jobs.sqlite3", cache_dir=root / "cache",
                            allowed_origins=Settings.allowed_origins +
                                ((args.frontend_origin,) if args.frontend_origin else ()),
                            catalogs={"problems": root / "tasks.json", "mbpp": root / "tasks.json"})
        with ExperimentStore(settings.history_db) as store:
            task = dataset.tasks[0]
            run_candidate_refinement_loop(BenchmarkClient(), task.problem_statement, prompt_registry=registry,
                                          trusted_test_suite=task.visible_tests, history_store=store,
                                          task_id=task.task_id, benchmark_split="train")
        app = create_app(settings, client_factory=lambda **kw: (FixtureClient(kw["model"]), None))
        uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
