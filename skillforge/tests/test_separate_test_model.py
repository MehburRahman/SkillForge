"""Exercise model routing and trusted-test bypass through real execution."""
import json

from fastapi.testclient import TestClient
import pytest

from api.main import create_app
from api.models import SolveRequest
from src.agents.test_generator import build_test_suite
from src.history import ExperimentStore
from src.reinforcement_loop import run_candidate_refinement_loop
from tests.test_api import setup, finished
from tests.test_benchmark_runner import BenchmarkClient
from tests.test_executable_test_oracle import case, reference, spec, unsure
from tests.test_generated_test_review import SequenceClient


class SolverClient(BenchmarkClient):
    model = 'candidate:7b'

    def generate(self, prompt, system=None, temperature=0.2, *, seed=None):
        if 'problem analyzer' in (system or '').lower():
            return super().generate(prompt, system, temperature, seed=seed)
        assert 'test cases' not in (system or '').lower()
        assert 'review generated' not in (system or '').lower()
        self.requests.append(dict(prompt=prompt, system=system, seed=seed))
        return 'import sys\nvalues = list(map(int, sys.stdin.read().split()))\nprint(sum(values[1:]))'


class TestModel(SequenceClient):
    __test__ = False
    model = 'reference:larger'

    def __init__(self):
        super().__init__([[case(), case('2 -4 7', '3')], [unsure(), unsure(2)], reference()])


def test_auxiliary_model_handles_generation_review_and_reference_with_correct_audit_and_cost(tmp_path):
    solver, tests = SolverClient(), TestModel()
    with ExperimentStore(tmp_path / 'history.sqlite3') as store:
        result = run_candidate_refinement_loop(solver, spec().problem_statement, test_client=tests,
            reference_test_suite=build_test_suite([case('1 13', '13')]),
            seed=42, max_attempts=1, history_store=store)
        run = store._connection.execute('SELECT * FROM runs WHERE run_id=?', (result.history_run_id,)).fetchone()
        rows = store._connection.execute('SELECT * FROM model_requests ORDER BY call_id').fetchall()
    assert result.success and result.best_attempt.candidate.model_name == solver.model
    assert result.test_model == run['test_model'] == tests.model
    assert result.test_model_calls == run['test_model_calls'] == 3
    assert result.model_calls == len(rows) == 5
    assert [row['role'] for row in rows] == ['generator', 'generator', 'test_generator', 'test_generator', 'test_generator']
    assert [row['stage'] for row in rows] == ['analysis', 'candidate:1', 'tests:1', 'test_validation:1', 'test_oracle:1']
    assert len(solver.requests) == 2 and len(tests.requests) == 3
    assert all(json.loads(row['request_json'])['seed'] is not None for row in rows)
    assert all(json.loads(row['request_json'])['options']['num_predict'] == 4096
               for row in rows if row['role'] == 'test_generator')
    assert 'num_predict' not in getattr(solver, 'generation_options', {})


def test_trusted_tests_bypass_the_auxiliary_model_and_report_zero_test_calls():
    tests = TestModel()
    result = run_candidate_refinement_loop(SolverClient(), spec().problem_statement, test_client=tests,
        trusted_test_suite=build_test_suite([case(), case('2 -4 7', '3')]), max_attempts=1)
    assert result.success and result.test_source == 'trusted'
    assert result.test_model is None and result.test_model_calls == 0
    assert result.model_calls == 2 and not tests.requests


def test_api_preserves_auxiliary_model_settings_and_history(setup):
    settings, _, _ = setup
    models = []
    def test_factory(name):
        models.append(name)
        return TestModel()
    with TestClient(create_app(settings, client_factory=lambda **kwargs: (SolverClient(), None),
                               test_client_factory=test_factory)) as client:
        response = client.post('/solves', json=dict(problem=spec().problem_statement,
            model='candidate:7b', test_model='reference:larger', seed=42, max_attempts=1,
            reference_tests=[case('1 13', '13')],
            test_generation_method='json'))
        assert response.status_code == 202, response.text
        job = finished(client, response.json()['job_id'])
        assert job['status'] == 'completed', job
        assert job['request']['test_model'] == job['result']['test_model'] == 'reference:larger'
        assert job['result']['model_calls'] == 5 and job['result']['test_model_calls'] == 3
        run = client.get('/runs/' + job['resource_id']).json()
        assert run['model_name'] == 'candidate:7b'
        assert run['test_model'] == 'reference:larger' and run['test_model_calls'] == 3
    assert models == ['reference:larger']


def test_api_does_not_construct_a_test_client_with_supplied_trusted_tests(setup):
    settings, _, _ = setup
    def forbidden_factory(*args, **kwargs):
        pytest.fail('Trusted tests must bypass construction of the auxiliary client.')
    with TestClient(create_app(settings, client_factory=lambda **kwargs: (SolverClient(), None),
                               test_client_factory=forbidden_factory)) as client:
        response = client.post('/solves', json=dict(problem=spec().problem_statement,
            tests=[case()], test_model='unused-model', max_attempts=1))
        assert response.status_code == 202, response.text
        job = finished(client, response.json()['job_id'])
        assert job['status'] == 'completed', job
        assert job['result']['test_source'] == 'trusted'
        assert job['result']['test_model'] is None and job['result']['test_model_calls'] == 0


@pytest.mark.parametrize('model', ['', '   ', 7, 'x' * 201])
def test_auxiliary_model_setting_rejects_invalid_names(model):
    with pytest.raises(ValueError):
        SolveRequest(problem='Read an integer.', test_model=model)
