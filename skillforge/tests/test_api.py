import json
import os
import threading
import time

from fastapi.testclient import TestClient
import pytest

from api.config import Settings
from api.jobs import JobManager, JobStore, TERMINAL, WorkerAlreadyRunningError
from api.main import create_app
from api.models import SolveRequest
from api.service import Service
from src.benchmark.runner import run_benchmark
from src.history import ExperimentStore
from test_benchmark_runner import BenchmarkClient, make_registry, write_dataset


@pytest.fixture
def setup(tmp_path):
    make_registry(tmp_path / "prompts.json")
    write_dataset(tmp_path / "catalog.json")
    settings = Settings(history_db=tmp_path / "history.sqlite3", registry=tmp_path / "prompts.json",
                        jobs_db=tmp_path / "jobs.sqlite3", cache_dir=tmp_path / "cache",
                        catalogs={"problems": tmp_path / "catalog.json", "mbpp": tmp_path / "catalog.json"})
    clients = []
    def factory(**kwargs):
        client = BenchmarkClient()
        clients.append(client)
        return client, None
    return settings, factory, clients


def finished(client, job_id, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in TERMINAL:
            return job
        time.sleep(0.02)
    pytest.fail("job did not finish")


def test_solve_stream_history_and_reconnect(setup):
    settings, factory, clients = setup
    with TestClient(create_app(settings, client_factory=factory)) as client:
        response = client.post("/solves", json={"task_id": "double-train", "seed": 42})
        assert response.status_code == 202, response.text
        job_id = response.json()["job_id"]
        assert response.headers["location"] == f"/jobs/{job_id}"
        job = finished(client, job_id)
        assert job["status"] == "completed", job
        assert job["result"]["attempt_count"] == 2
        assert job["result"]["attempt_budget"] == 4
        assert job["result"]["attempt_budget_source"] == "task_analysis_llm"
        assert job["result"]["reasoning_mode"] == "direct"
        run_id = job["resource_id"]
        history = client.get(f"/runs/{run_id}").json()
        assert len(history["attempts"]) == 2
        assert history["reasoning_mode"] == "direct"
        assert "print(value * 2)" in history["attempts"][1]["code_diff"]
        assert client.get("/runs?split=train&prompt_id=P0").json()["total"] == 1
        assert client.get("/runs?split=validation").json()["total"] == 0
        stream = client.get(f"/jobs/{job_id}/events")
        assert stream.headers["content-type"].startswith("text/event-stream")
        events = [block for block in stream.text.split("\n\n") if block]
        ids = [int(block.splitlines()[0][4:]) for block in events]
        assert ids == sorted(set(ids))
        stages = [json.loads(block.split("data: ")[1])["stage"] for block in events if "event: stage\n" in block]
        assert stages == ["preparing", "request_logging", "analysis", "tests_ready", "generation", "run_started",
                          "verification", "evaluation", "refinement", "verification", "evaluation", "finished"]
        assert "event: completed" in events[-1]
        replay = client.get(f"/jobs/{job_id}/events", headers={"Last-Event-ID": str(ids[-2])})
        assert replay.text == events[-1] + "\n\n"
        assert client.get(f"/jobs/{job_id}/events?after={ids[-1]}").text == ""
        for text in (stream.text, json.dumps(history), json.dumps(clients[-1].requests)):
            assert "1998" not in text  # Hidden expected answer never enters the solve path.


def test_read_endpoints_do_not_create_history_or_mutate_registry(setup):
    settings, factory, _ = setup
    original = settings.registry.read_bytes()
    with TestClient(create_app(settings, client_factory=factory)) as client:
        assert client.get("/health").json()["worker_alive"]
        for path in ("/runs", "/benchmarks", "/evolutions"):
            assert client.get(path).json() == {"items": [], "total": 0}
            assert client.get(path + "/missing").status_code == 404
        assert client.get("/jobs/missing").status_code == 404
        assert client.get("/jobs/missing/events").status_code == 404
        assert client.get("/tasks?split=validation").json()["total"] == 1
        assert "hidden_tests" not in client.get("/tasks").text
        assert "401 402" not in client.get("/tasks").text
        assert client.get("/tasks?limit=1&offset=1").json()["items"][0]["task_id"] == "sum-validation"
        assert client.get("/prompts").json()["active_prompt_id"] == "P0"
        assert "+CANDIDATE SYSTEM PROMPT" in client.get("/prompts/diff?baseline=P0&candidate=P1").json()["diff"]
        assert client.get("/prompts/diff?baseline=P0&candidate=P99").status_code == 404
        assert client.get("/runs?limit=101").status_code == 422
        assert client.get("/tasks?catalog=/etc/passwd").status_code == 404
        assert client.get("/openapi.json").status_code == 200
    assert not settings.history_db.exists()
    assert settings.registry.read_bytes() == original


@pytest.mark.parametrize("body", [
    {}, {"problem": " "}, {"problem": "x", "task_id": "double-train"},
    {"problem": "x", "catalog": "mbpp"}, {"problem": "x", "language": "cpp"},
    {"problem": "x", "max_attempts": 0}, {"problem": "x", "max_attempts": True},
    {"problem": "x", "seed": -1}, {"problem": "x", "history_db": "elsewhere"},
    {"task_id": "double-train", "language": "javascript"},
    {"task_id": "double-train", "tests": [{"input": "2", "expected_output": "4"}]},
    {"problem": "x", "execution_mode": "function", "entry_point": "f"},
    {"problem": "x", "tests": [{"input": "2", "expected_output": None}]},
    {"problem": "x", "prompt_id": "P99"},
])
def test_invalid_solve_never_queued(setup, body):
    settings, factory, clients = setup
    with TestClient(create_app(settings, client_factory=factory)) as client:
        response = client.post("/solves", json=body)
        assert response.status_code == 422, response.text
        assert client.get("/health").json()["jobs"] == {}
    assert not settings.history_db.exists()
    assert all(not c.requests for c in clients)


def test_adhoc_solve_serialization_and_origins(setup):
    settings, factory, _ = setup
    with TestClient(create_app(settings, client_factory=factory)) as client:
        body = {"problem": "Read two integers and print their sum.", "max_attempts": 1,
                "tests": [{"input": "2 3", "expected_output": "5"}]}
        assert client.post("/solves", json=body, headers={"Origin": "https://external.example"}).status_code == 403
        response = client.post("/solves", json=body, headers={"Origin": "http://localhost:5173"})
        assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
        job = finished(client, response.json()["job_id"])
        assert job["status"] == "completed", job
        assert client.get("/runs?split=adhoc").json()["total"] == 1
        own_origin = client.post("/solves", json=body, headers={"Origin": "http://testserver"})
        assert own_origin.status_code == 202
        assert finished(client, own_origin.json()["job_id"])["status"] == "completed"
        assert client.get(f"/jobs/{job['job_id']}/events", headers={"Last-Event-ID": "bad"}).status_code == 422
        assert client.get("/health", headers={"Host": "external.example"}).status_code == 400


def test_benchmark_queries_hide_checkpoint_and_match_summary(setup):
    settings, factory, _ = setup
    dataset = write_dataset(settings.catalogs["problems"])
    registry = make_registry(settings.registry)
    with ExperimentStore(settings.history_db) as store:
        expected = run_benchmark(BenchmarkClient(), dataset, prompt_registry=registry, history_store=store,
                                 max_attempts=2, run_id="sweep")
        store._connection.execute("UPDATE benchmark_tasks SET checkpoint_json=?", ('{"secret":"PRIVATE-CHECKPOINT"}',))
        store._connection.commit()
    with TestClient(create_app(settings, client_factory=factory)) as client:
        assert client.get("/benchmarks").json()["total"] == 1
        response = client.get("/benchmarks/sweep")
        assert response.json()["summary"] == expected.to_dict()
        assert response.json()["expected_tasks"] == 2
        assert "PRIVATE-CHECKPOINT" not in response.text
        assert "401 402" not in response.text
        assert "manifest" not in response.text


def test_evolution_empty_evidence_blocks_without_model_call_and_can_resume(setup):
    settings, factory, clients = setup
    with TestClient(create_app(settings, client_factory=factory)) as client:
        assert client.post("/evolutions", json={"resume": True}).status_code == 422
        response = client.post("/evolutions", json={})
        assert response.status_code == 202, response.text
        job = finished(client, response.json()["job_id"])
        assert job["status"] == "completed", job
        assert job["result"]["status"] == "blocked"
        round_id = job["resource_id"]
        assert round_id == job["result"]["round_id"]
        assert client.get("/evolutions").json()["total"] == 1
        detail = client.get(f"/evolutions/{round_id}").json()
        assert detail["status"] == "blocked"
        assert "manifest" not in detail
        resumed = client.post("/evolutions", json={"run_id": round_id, "resume": True})
        assert finished(client, resumed.json()["job_id"])["result"] == job["result"]
    assert all(not c.requests for c in clients)


def test_api_evolution_proposes_validates_and_promotes_with_isolated_models(setup):
    from tests.test_evolution_workflow import EvolutionClient, SeededMetaClient
    from tests.test_prompt_evolver import add_runs, registry
    settings, _, _ = setup
    registry(settings.registry.parent)
    catalog = json.loads(settings.catalogs["mbpp"].read_text())
    validation = catalog["tasks"][1]
    catalog["tasks"] = [catalog["tasks"][0], *[
        {**validation, "task_id": f"v{i}", "problem_statement": f"Task {i}: {validation['problem_statement']}"}
        for i in range(4)]]
    settings.catalogs["mbpp"].write_text(json.dumps(catalog))
    with ExperimentStore(settings.history_db) as store:
        add_runs(store)
    generator, meta = EvolutionClient(), SeededMetaClient()
    app = create_app(settings, client_factory=lambda **kw: (generator, None), meta_factory=lambda model: meta)
    with TestClient(app) as client:
        accepted = client.post("/evolutions", json={"meta_model": "meta-model", "max_attempts": 2})
        job = finished(client, accepted.json()["job_id"])
        assert job["status"] == "completed", job
        assert job["result"]["decision"] == "promoted"
        assert client.get("/prompts").json()["active_prompt_id"] == "P1"
        detail = client.get(f"/evolutions/{job['resource_id']}").json()
        assert detail["validation"]["comparison"]["p_value"] == 0.0625
        assert client.get("/benchmarks").json()["total"] == 2
        stream = client.get(f"/jobs/{job['job_id']}/events").text
        assert "event: progress" in stream
        assert "401 402" not in stream
        assert "SECRET_TEST_INPUT" not in stream
    assert len(meta.requests) == 1


@pytest.mark.parametrize("language", ["python", "javascript"])
def test_function_solve_with_independent_judge(setup, language):
    import shutil
    from tests.test_language_evaluation import RecordingJudge
    if language == "javascript" and shutil.which("node") is None:
        pytest.skip("Node.js is required")
    settings, _, _ = setup
    class FunctionClient(BenchmarkClient):
        def generate(self, prompt, system=None, temperature=0.2, *, seed=None):
            if "problem analyzer" in (system or "").lower():
                return super().generate(prompt, system=system, temperature=temperature, seed=seed)
            return "def solve(a, b): return a + b" if language == "python" else "function solve(a,b) {return a+b;}"
    judge = RecordingJudge()
    app = create_app(settings, client_factory=lambda **kw: (FunctionClient(), judge))
    assertion = "assert solve(2, 3) == 5" if language == "python" else "assert.strictEqual(solve(2, 3), 5);"
    with TestClient(app) as client:
        response = client.post("/solves", json={
            "problem": "Return the sum of two integers from solve(a, b).", "language": language,
            "execution_mode": "function", "entry_point": "solve", "tests": [{"assertion": assertion}],
            "max_attempts": 1, "llm_judge": True})
        assert response.status_code == 202, response.text
        job = finished(client, response.json()["job_id"])
        assert job["status"] == "completed", job
        assert job["result"]["evaluation"]["judge_grade"] is not None
        assert job["result"]["evaluation"]["functional_grade"]["score"] == 1
    assert len(judge.payloads) == 1


def test_same_family_judge_request_is_accepted(setup):
    settings, factory, _ = setup
    with TestClient(create_app(settings, client_factory=factory)) as client:
        response = client.post("/solves", json={"task_id": "double-train", "model": "qwen2.5-coder:7b",
                                                "judge_model": "qwen3:4b"})
        assert response.status_code == 202
        assert finished(client, response.json()['job_id'])['status'] == 'completed'


def test_one_worker_keeps_http_available_and_runs_jobs_in_order(setup):
    settings, factory, _ = setup
    app = create_app(settings, client_factory=factory)
    started, release = threading.Event(), threading.Event()
    order = []
    def executor(kind, payload, emit):
        order.append(payload["request"]["task_id"])
        started.set()
        assert release.wait(5)
        return {"status": "ok"}
    app.state.manager.executor = executor
    with TestClient(app) as client:
        try:
            first = client.post("/solves", json={"task_id": "double-train"}).json()
            assert started.wait(2)
            second = client.post("/solves", json={"task_id": "sum-validation"}).json()
            assert client.get(f"/jobs/{second['job_id']}").json()["status"] == "queued"
            assert client.get("/health").json()["jobs"] == {"queued": 1, "running": 1}
            assert client.get("/runs").status_code == 200
            assert order == ["double-train"]
        finally:
            release.set()
        assert finished(client, first["job_id"])["status"] == "completed"
        assert finished(client, second["job_id"])["status"] == "completed"
        assert order == ["double-train", "sum-validation"]


def test_queue_full_and_job_failure_does_not_stop_worker(setup):
    settings, factory, _ = setup
    started, release = threading.Event(), threading.Event()
    app = create_app(settings, client_factory=factory)
    calls = []
    def executor(kind, payload, emit):
        calls.append(kind)
        started.set()
        assert release.wait(5)
        if len(calls) == 1:
            raise RuntimeError("model unavailable")
        return {"status": "ok"}
    app.state.manager.executor = executor
    app.state.manager.capacity = 1
    with TestClient(app) as client:
        try:
            first = client.post("/solves", json={"task_id": "double-train"}).json()
            assert started.wait(2)
            full = client.post("/solves", json={"task_id": "double-train"})
            assert full.status_code == 503
            assert full.headers["retry-after"] == "5"
        finally:
            release.set()
        assert finished(client, first["job_id"])["error"] == "model unavailable"
        second = client.post("/solves", json={"task_id": "double-train"}).json()
        assert finished(client, second["job_id"])["status"] == "completed"


def test_verifier_infrastructure_failure_does_not_create_a_candidate_verdict(setup):
    from src.verifier.runners.http import HTTPInfrastructureError
    settings, factory, _ = setup
    app = create_app(settings,client_factory=factory)
    def executor(*args):
        raise HTTPInfrastructureError('Infrastructure error: HTTP verifier crashed')
    app.state.manager.executor = executor
    with TestClient(app) as api:
        queued = api.post('/solves',json={'task_id':'double-train'}).json()
        job = finished(api,queued['job_id'])
        assert job['status'] == 'failed'
        assert job['error'].startswith('Infrastructure error:')
        assert job['result'] is None


def test_restart_marks_active_job_interrupted_and_executes_queued_once(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    store.initialize()
    active = store.enqueue("solve", {}, 8)
    store.claim()  # Simulate a worker killed after claiming a job.
    queued = store.enqueue("solve", {}, 8)
    calls = []
    manager = JobManager(store, lambda *args: calls.append(1) or {"status": "ok"})
    manager.start()
    try:
        second = JobManager(JobStore(store.path), lambda *args: None)
        with pytest.raises(WorkerAlreadyRunningError):
            second.start()
        deadline = time.monotonic() + 3
        while store.get(queued["job_id"])["status"] not in TERMINAL and time.monotonic() < deadline:
            time.sleep(0.01)
        assert store.get(active["job_id"])["status"] == "interrupted"
        assert store.events(active["job_id"])[-1]["event"] == "interrupted"
        assert store.get(queued["job_id"])["status"] == "completed"
        assert calls == [1]
    finally:
        manager.stop()
    manager.start()
    manager.stop()
    assert calls == [1]


def test_second_start_reports_owner_without_interrupting_the_active_job(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    started, release = threading.Event(), threading.Event()

    def execute(*args):
        started.set()
        assert release.wait(5)
        return {"status": "ok"}

    manager = JobManager(store, execute)
    second = JobManager(JobStore(store.path), lambda *args: {"status": "ok"})
    manager.start()
    try:
        job = manager.submit("solve", {})
        assert started.wait(2)
        lock_path = store.path.with_suffix(".sqlite3.lock")
        owner = lock_path.read_text()
        events = store.events(job["job_id"])
        with pytest.raises(WorkerAlreadyRunningError) as error:
            second.start()
        assert f"PID {os.getpid()}" in str(error.value)
        assert str(store.path) in str(error.value)
        assert "Stop the existing server" in str(error.value)
        assert owner == lock_path.read_text()
        assert store.get(job["job_id"])["status"] == "running"
        assert store.events(job["job_id"]) == events
        assert manager.alive and not second.alive
    finally:
        release.set()
        manager.stop()
    assert store.get(job["job_id"])["status"] == "completed"
    second.start()
    second.stop()


def test_failed_initialization_releases_the_worker_lock(tmp_path, monkeypatch):
    store = JobStore(tmp_path / "jobs.sqlite3")

    def fail():
        raise RuntimeError("initialization failed")

    monkeypatch.setattr(store, "initialize", fail)
    manager = JobManager(store, lambda *args: None)
    with pytest.raises(RuntimeError, match="initialization failed"):
        manager.start()
    replacement = JobManager(JobStore(store.path), lambda *args: None)
    replacement.start()
    replacement.stop()


def test_job_restores_public_request_settings_without_internal_data(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    store.initialize()
    payload = {"context": "private-path-context", "catalog_hash": "internal-hash", "request": {
        "problem": "Add two integers.", "model": "custom-model", "max_attempts": 5,
        "tests": [{"input": "PRIVATE-FIXTURE", "expected_output": "5"}],
        "history_db": "private-path", "judge_model": "independent-judge",
    }}
    queued = store.enqueue("solve", payload, 8)
    public = store.get(queued["job_id"])
    assert public["request"] == {"problem": "Add two integers.", "model": "custom-model", "max_attempts": 5,
                                 "judge_model": "independent-judge"}
    assert "PRIVATE-FIXTURE" not in json.dumps(public)
    assert "private-path" not in json.dumps(public)
    assert "internal-hash" not in json.dumps(public)


def test_queued_jobs_reject_changed_catalog_or_configuration(setup, tmp_path):
    settings, factory, _ = setup
    service = Service(settings, client_factory=factory)
    payload = service.prepare("solve", SolveRequest(task_id="double-train"))
    settings.catalogs["problems"].write_text(settings.catalogs["problems"].read_text() + "\n")
    with pytest.raises(ValueError, match="catalog changed"):
        service.execute("solve", payload, lambda *args: None)
    payload["context"] = "different server"
    with pytest.raises(ValueError, match="paths changed"):
        service.execute("solve", payload, lambda *args: None)
    assert not settings.history_db.exists()


def test_cli_runs_one_local_server(monkeypatch, tmp_path):
    import run_api
    calls = []
    monkeypatch.setattr(run_api.uvicorn, "run", lambda *args, **kwargs: calls.append(kwargs))
    run_api.main(["--port", "8765", "--jobs-db", str(tmp_path / "jobs.sqlite3")])
    assert calls == [{"host": "127.0.0.1", "port": 8765, "workers": 1}]
    assert not (tmp_path / "jobs.sqlite3").exists()


def test_scala_http_rejects_invalid_supplied_tests_before_queue(setup, monkeypatch):
    settings, factory, _ = setup
    monkeypatch.setattr('api.service.preflight_execution', lambda *args: {})
    with TestClient(create_app(settings, client_factory=factory)) as client:
        payload = {'problem':'Implement the supplied server contract.','language':'scala','execution_mode':'http'}
        payload['tests'] = [{'requests':[{'method':'POST','path':'https://external.example','expected_status':200,'expected_body':''}]}]
        assert client.post('/solves',json=payload).status_code == 422
        assert not settings.history_db.exists()


def test_missing_scala_toolchain_is_reported_without_queueing(setup, monkeypatch):
    settings, factory, _ = setup
    def missing(*args):
        raise RuntimeError('Scala compiler cache is not prepared. Run python tools/setup_scala.py.')
    monkeypatch.setattr('api.service.preflight_execution', missing)
    with TestClient(create_app(settings, client_factory=factory)) as client:
        response = client.post('/solves',json={'problem':'Read a number.','language':'scala'})
        assert response.status_code == 422, response.text
        assert 'setup_scala.py' in response.text
        assert not settings.history_db.exists()


from tests.test_scala_execution import requires_scala

@requires_scala
def test_scala_http_api_job_executes_and_saves_evidence(setup, monkeypatch):
    from tests.test_http_execution import SERVER, request
    settings, _, _ = setup
    monkeypatch.setenv('SKILLFORGE_SCALA_HTTP_LOCAL', '1')
    class ScalaClient(BenchmarkClient):
        def generate(self,prompt,system=None,temperature=.2,*,seed=None):
            if 'problem analyzer' in (system or '').lower():
                return super().generate(prompt,system,temperature,seed=seed)
            return SERVER
    with TestClient(create_app(settings,client_factory=lambda **kwargs:(ScalaClient(),None))) as client:
        response = client.post('/solves',json={
            'problem':'Implement an HTTP service that echoes UTF-8 request bodies at POST /echo.',
            'language':'scala','execution_mode':'http','max_attempts':1,
            'tests':[{'requests':[request(body='α\n')],'category':'normal'}]})
        assert response.status_code == 202, response.text
        job = finished(client,response.json()['job_id'])
        assert job['status'] == 'completed', job
        assert job['result']['evaluation']['functional_grade']['passed'] == 1, job
        assert job['result']['evaluation']['runtime_grade']['compilation_seconds'] > 0
        saved = client.get('/runs/'+job['resource_id']).json()
        assert saved['language'] == 'scala'
        assert saved['execution_mode'] == 'http'
