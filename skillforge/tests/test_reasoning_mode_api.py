"""Reasoning mode survives API validation and durable job serialization."""

from fastapi.testclient import TestClient
import pytest

from api.config import Settings
from api.jobs import JobStore
from api.main import create_app
from api.models import SolveRequest
from api.service import Service
from tests.test_benchmark_runner import BenchmarkClient, make_registry, write_dataset
from tests.test_api import finished


@pytest.fixture
def setup(tmp_path):
    make_registry(tmp_path / "prompts.json")
    write_dataset(tmp_path / "catalog.json")
    settings = Settings(history_db=tmp_path / "history.sqlite3", registry=tmp_path / "prompts.json",
                        jobs_db=tmp_path / "jobs.sqlite3", cache_dir=tmp_path / "cache",
                        catalogs={"problems": tmp_path / "catalog.json"})
    models = []
    def factory(**kwargs):
        model = BenchmarkClient()
        models.append(model)
        return model, None
    return settings, factory, models


def test_default_reasoning_mode_is_direct():
    assert SolveRequest(problem="Read one integer.").reasoning_mode == "direct"


def test_planned_reasoning_is_preserved_without_exposing_execution_fixtures(setup):
    settings, factory, models = setup
    service = Service(settings, client_factory=factory)
    request = SolveRequest(problem="Read two integers and print their sum.",
                           reasoning_mode="plan_critique",
                           tests=[{"input": "2 3", "expected_output": "5"}])
    payload = service.prepare("solve", request)
    jobs = JobStore(settings.jobs_db)
    jobs.initialize()
    job = jobs.enqueue("solve", payload, 8)
    assert job["request"]["reasoning_mode"] == "plan_critique"
    assert "tests" not in job["request"] and "context" not in job["request"]
    assert jobs.get(job["job_id"])["request"]["reasoning_mode"] == "plan_critique"
    assert all(not model.requests for model in models)


def test_planned_solve_records_mode_calls_and_keeps_candidate_budget(setup):
    settings, _, models = setup
    class PlanningClient(BenchmarkClient):
        def generate(self, prompt, system=None, temperature=0.2, *, seed=None, max_output_tokens=None):
            if max_output_tokens == 1500:
                self.requests.append({"prompt": prompt, "system": system, "seed": seed})
                return 'Read two integer tokens and print their sum. Addition is exact; time and space are constant.'
            return super().generate(prompt, system=system, temperature=temperature, seed=seed)
    def factory(**kwargs):
        model = PlanningClient()
        models.append(model)
        return model, None
    with TestClient(create_app(settings, client_factory=factory)) as client:
        response = client.post('/solves', json={
            'task_id': 'sum-validation', 'reasoning_mode': 'plan_critique',
            'config': 'single_shot', 'seed': 42,
        })
        assert response.status_code == 202, response.text
        job = finished(client, response.json()['job_id'])
        assert job['status'] == 'completed', job
        assert job['request']['reasoning_mode'] == 'plan_critique'
        assert job['result']['reasoning_mode'] == 'plan_critique'
        assert job['result']['attempt_budget'] == job['result']['attempt_count'] == 1
        assert job['result']['model_calls'] == 4
        history = client.get('/runs/' + job['resource_id']).json()
        assert history['reasoning_mode'] == 'plan_critique'
    assert all('401 402' not in request['prompt'] and '803' not in request['prompt']
               for model in models for request in model.requests)


@pytest.mark.parametrize("mode", ["unknown", "PLAN_CRITIQUE", "", None, True, 3])
def test_invalid_reasoning_mode_never_starts_a_model_job(setup, mode):
    settings, factory, models = setup
    with TestClient(create_app(settings, client_factory=factory)) as client:
        response = client.post("/solves", json={"problem": "Read one integer.", "reasoning_mode": mode})
        assert response.status_code == 422
        assert client.get("/health").json()["jobs"] == {}
    assert all(not model.requests for model in models)
    assert not settings.history_db.exists()
