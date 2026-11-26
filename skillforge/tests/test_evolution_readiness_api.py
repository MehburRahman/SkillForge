import json

from fastapi.testclient import TestClient

from api.main import create_app
from src.history import ExperimentStore
from tests.test_api import setup, finished
from tests.test_prompt_evolver import add_runs


def test_readiness_is_read_only_and_adhoc_runs_do_not_count(setup):
    settings, factory, _ = setup
    with TestClient(create_app(settings, client_factory=factory)) as client:
        response = client.get('/evolutions/readiness?catalog=problems')
        assert response.status_code == 200 and response.json()['training_runs'] == 0
        assert not settings.history_db.exists()
        with ExperimentStore(settings.history_db) as store:
            add_runs(store, split='adhoc')
        before = settings.history_db.read_bytes()
        readiness = client.get('/evolutions/readiness?catalog=problems').json()
        assert readiness['eligible'] is False and readiness['training_runs'] == 0
        assert '0/20' in readiness['reason']
        assert settings.history_db.read_bytes() == before


def test_readiness_reports_scope_mismatch_and_keeps_training_private(setup):
    settings, factory, _ = setup
    with ExperimentStore(settings.history_db) as store:
        add_runs(store)
        with store._connection:
            store._connection.execute("UPDATE runs SET language='scala', execution_mode='http'")
    with TestClient(create_app(settings, client_factory=factory)) as client:
        response = client.get('/evolutions/readiness?catalog=problems')
        assert response.json()['eligible'] is False
        assert 'different language/execution mode' in response.json()['reason']
        assert 'SECRET_TEST_INPUT' not in response.text
        assert 'run_ids' not in response.text


def test_api_context_and_model_requests_are_saved_with_run(setup):
    settings, factory, models = setup
    with TestClient(create_app(settings, client_factory=factory)) as client:
        job = client.post('/solves', json={'task_id': 'double-train', 'context_window': 4096}).json()
        result = finished(client, job['job_id'])
        assert result['status'] == 'completed', result
        run = client.get('/runs/' + result['resource_id']).json()
        logs = client.get('/model-requests/' + run['request_log_id']).json()['items']
        assert logs and logs[0]['request']['options']['num_ctx'] == 4096
        assert logs[-1]['response_text'] == run['attempts'][-1]['candidate_code']
        model_text = '\n'.join(
            text for log in logs for text in (
                log['request'].get('prompt', ''), log['request'].get('system', ''), log['response_text'] or '',
            )
        )
        assert '1998' not in model_text  # Check model-visible data, excluding random timings and IDs.
        assert client.post('/solves', json={'problem': 'Example', 'context_window': 0}).status_code == 422
