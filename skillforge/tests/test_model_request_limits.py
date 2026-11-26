"""Bound local inference waits without assuming a task, language, or model."""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
import src.agents.llm_client as module
from src.agents.llm_client import (
    LLMClient, DEFAULT_ANALYSIS_OUTPUT_TOKENS, DEFAULT_TASK_FORMAT_OUTPUT_TOKENS,
)
from src.agents.task_spec import parse_task
from src.history import ExperimentStore
from tests.test_api import setup, finished


def delivery_response():
    return {stream: dict(explicit_in_statement=False, quoted_rules='',
                         serialization='plain text', layout='One complete text record',
                         whitespace='Preserve all whitespace without adding a newline',
                         empty_values='An empty record is zero characters')
            for stream in ('stdin', 'stdout')}


@pytest.mark.parametrize('timeout', [True, 0, -1, float('nan'), float('inf'), '30'])
def test_invalid_request_timeout_is_rejected_before_model_calls(timeout):
    with pytest.raises(ValueError, match='request_timeout'):
        LLMClient(request_timeout=timeout)


def test_timeout_configuration_reaches_the_backend_and_test_client(monkeypatch):
    constructed = []
    def factory(**kwargs):
        constructed.append(kwargs['timeout'])
        return SimpleNamespace(chat=lambda **kwargs: {'message': {'content': '{}'}})
    monkeypatch.setattr(module, 'ollama', SimpleNamespace(Client=factory))
    monkeypatch.setenv('SKILLFORGE_LLM_TIMEOUT_SECONDS', '47.5')
    client = LLMClient('any-model')
    tests = client.for_tests()
    client.generate('First request')
    client.generate_json('Second request')
    tests.generate('Test request')
    assert len(constructed) == 2  # Reuse the transport for sequential requests.
    assert all(timeout.read == 47.5 and timeout.connect == 10 for timeout in constructed)
    assert tests.request_timeout == client.request_timeout == 47.5


@pytest.mark.parametrize('value', ['', 'invalid', '-1', 'nan', 'inf'])
def test_invalid_environment_timeout_is_reported(monkeypatch, value):
    monkeypatch.setenv('SKILLFORGE_LLM_TIMEOUT_SECONDS', value)
    with pytest.raises(ValueError, match='positive finite'):
        LLMClient()


def test_analysis_budget_does_not_limit_the_candidate_source_request(monkeypatch):
    requests = []
    def chat(**kwargs):
        requests.append(kwargs)
        if 'stdin' in kwargs.get('format', {}).get('required', []):
            return {'message': {'content': json.dumps(delivery_response())}}
        return {'message': {'content': json.dumps(dict(problem_type='other', constraints='',
            examples=[], selected_max_attempts=1, attempt_budget_reason='One candidate suffices.'))}}
    monkeypatch.setattr(module, 'ollama', SimpleNamespace(Client=lambda **kwargs: SimpleNamespace(chat=chat)))
    client = LLMClient('any-model')
    parse_task(client, 'Read input and return a result.')
    client.generate('Write candidate source.')
    assert requests[0]['options']['num_predict'] == DEFAULT_ANALYSIS_OUTPUT_TOKENS
    assert requests[1]['options']['num_predict'] == DEFAULT_TASK_FORMAT_OUTPUT_TOKENS
    assert 'num_predict' not in requests[2]['options']
    assert client.generation_options == {}


def test_real_transport_stops_waiting_for_a_model_that_never_returns(monkeypatch):
    native = pytest.importorskip('ollama')
    entered, release = threading.Event(), threading.Event()
    class WaitingModel(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            entered.set()
            release.wait(2)
            self.send_response(200)
            self.end_headers()
        def log_message(self, *args):
            pass
    server = HTTPServer(('127.0.0.1', 0), WaitingModel)
    worker = threading.Thread(target=lambda: server.serve_forever(poll_interval=.01), daemon=True)
    worker.start()
    monkeypatch.setattr(module, 'ollama', SimpleNamespace(Client=lambda **kwargs:
        native.Client(host=f'http://127.0.0.1:{server.server_port}', **kwargs)))
    client = LLMClient('any-model', request_timeout=.1)
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match='timed out after 0.1 seconds') as caught:
            client.generate('Any programming problem.')
        assert entered.is_set() and time.monotonic() - started < 2
        assert len(client.last_usage['infrastructure_retries']) == 2
        assert isinstance(caught.value.__cause__, httpx.ReadTimeout)
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        worker.join(timeout=1)
        if client._backend is not None:
            client._backend._client.close()


def test_analysis_timeout_is_saved_and_the_api_worker_can_process_the_next_job(setup, monkeypatch):
    settings, _, _ = setup
    calls = 0
    def chat(**kwargs):
        nonlocal calls
        calls += 1
        if calls <= 3:
            raise httpx.ReadTimeout('Model did not return')
        system = kwargs['messages'][0]['content']
        if 'stdin' in kwargs.get('format', {}).get('required', []):
            content = json.dumps(delivery_response())
        elif 'Analyze the programming task' in system:
            content = json.dumps(dict(problem_type='other', constraints='', examples=[],
                selected_max_attempts=1, attempt_budget_reason='One candidate suffices.'))
        else:
            content = 'import sys\nprint(sys.stdin.read(), end="")'
        return {'message': {'content': content}}
    monkeypatch.setattr(module, 'ollama', SimpleNamespace(Client=lambda **kwargs: SimpleNamespace(chat=chat)))
    payload = dict(problem='Read input verbatim and output it unchanged.', max_attempts=1,
                   tests=[dict(input='sample text\n', expected_output='sample text\n', category='normal')])
    with TestClient(create_app(settings)) as api:
        failed = finished(api, api.post('/solves', json=payload).json()['job_id'])
        assert failed['status'] == 'failed' and 'timed out after' in failed['error']
        assert api.get('/runs').json()['total'] == 0
        completed = finished(api, api.post('/solves', json=payload).json()['job_id'])
        assert completed['status'] == 'completed' and completed['result']['status'] == 'pass'
    with ExperimentStore(settings.history_db, readonly=True) as store:
        rows = store._connection.execute('SELECT error,stage FROM model_requests ORDER BY call_id').fetchall()
        assert len(rows) == 4 and 'timed out after' in rows[0]['error']
        assert rows[0]['stage'] == 'cached_analysis'
