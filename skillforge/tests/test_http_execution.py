from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess

import pytest

from api.models import SolveRequest
from src.agents.test_generator import build_test_suite, validate_generation_tests
from src.verifier.outcome_verifier import verify
from src.verifier.runners import scala, http
from src.verifier.runners.base import Status
from tests.test_scala_execution import requires_scala


def request(path='/echo', body='hi', status=200, expected=None, method='POST'):
    return dict(method=method,path=path,body=body,expected_status=status,
                expected_body=body if expected is None else expected)


def suite(*requests):
    return build_test_suite([{'requests':list(requests),'category':'normal'}], execution_mode='http')


SERVER = '''object Main {
  def main(args: Array[String]): Unit = {
    val server = com.sun.net.httpserver.HttpServer.create(new java.net.InetSocketAddress("127.0.0.1", 0), 0)
    var count = 0
    server.createContext("/", exchange => {
      count += 1
      val path = exchange.getRequestURI.getPath
      if(path == "/wait") Thread.sleep(10000)
      val received = new String(exchange.getRequestBody.readAllBytes(), java.nio.charset.StandardCharsets.UTF_8)
      val status = if(path == "/echo" || path == "/count") 200 else 404
      val body = (if(path == "/count") count.toString else if(status==404) "missing" else received).getBytes(java.nio.charset.StandardCharsets.UTF_8)
      exchange.getResponseHeaders.set("Content-Type","text/plain; charset=utf-8")
      exchange.sendResponseHeaders(status, if(body.length == 0) -1 else body.length)
      exchange.getResponseBody.write(body)
      exchange.close()
    })
    server.start()
    java.nio.file.Files.writeString(java.nio.file.Paths.get(sys.env("SKILLFORGE_PORT_FILE")), server.getAddress.getPort.toString)
  }
}'''


@pytest.mark.parametrize('change', [
    {'path':'https://example.com'}, {'path':'//example.com'}, {'path':'/bad\r\nHeader: x'},
    {'headers':{'Host':'remote'}}, {'headers':{'Content-Length':'0'}},
    {'expected_status':True}, {'expected_body':42}, {'unknown':'x'},
])
def test_http_contract_rejects_ambiguous_requests(change):
    with pytest.raises(ValueError):
        suite({**request(),**change})


def test_request_round_trip_and_suite_requirements():
    original = suite(request(body='  α\n'))
    parsed = SolveRequest(problem='An arbitrary service contract', language='scala', execution_mode='http',
                          tests=[asdict(c) for c in original.cases])
    rebuilt = build_test_suite([t.model_dump() for t in parsed.tests], execution_mode='http')
    assert rebuilt == original
    validate_generation_tests(None,'http')
    with pytest.raises(ValueError, match='duplicates'):
        build_test_suite([asdict(original.cases[0])]*2, execution_mode='http')


@requires_scala
def test_http_arbitrary_routes_unicode_whitespace_and_request_sequences(monkeypatch):
    monkeypatch.setenv('SKILLFORGE_SCALA_HTTP_LOCAL','1')
    cases = suite(request(body='  α😀\n'), request('/count', expected='2'), request('/unknown', status=404, expected='missing'))
    result = verify(SERVER, cases, language='scala', execution_mode='http')
    assert result.pass_rate == 1, result.failures
    # Each scenario receives a fresh process, but requests within it share state.
    another = suite(request('/count', expected='1'))
    cases.cases.extend(another.cases)
    assert verify(SERVER,cases,language='scala',execution_mode='http').pass_rate == 1


@requires_scala
def test_wrong_response_hidden_evidence_and_server_cleanup(monkeypatch):
    monkeypatch.setenv('SKILLFORGE_SCALA_HTTP_LOCAL','1')
    processes = []
    popen = http.subprocess.Popen
    def track(*args, **kwargs):
        p = popen(*args, **kwargs)
        processes.append(p)
        return p
    monkeypatch.setattr(http.subprocess,'Popen',track)
    cases = suite(request(body='private-hidden-input', expected='different'))
    result = verify(SERVER,cases,language='scala',execution_mode='http',max_failures=0)
    assert result.passed == 0 and result.execution_results[0].status == Status.FAIL
    assert result.failures == [] and result.execution_results[0].stderr == ''
    assert result.execution_results[0].http_failure == {}
    assert 'private-hidden-input' not in str(asdict(result))
    assert all(p.poll() is not None for p in processes)


@requires_scala
def test_http_repair_receives_actual_request_and_response_not_pass_fail(monkeypatch):
    monkeypatch.setenv('SKILLFORGE_SCALA_HTTP_LOCAL','1')
    result = verify(SERVER, suite(request(body='actual text',expected='different text')),
                    language='scala',execution_mode='http')
    failure = result.failures[0]
    assert json.loads(failure['input'])['request']['body'] == 'actual text'
    assert json.loads(failure['expected'])['body'] == 'different text'
    assert json.loads(failure['actual'])['body'] == 'actual text'
    assert json.loads(failure['actual'])['status'] == 200


@requires_scala
def test_slow_request_and_missing_startup_fail_with_cleanup(monkeypatch):
    monkeypatch.setenv('SKILLFORGE_SCALA_HTTP_LOCAL','1')
    result = verify(SERVER,suite(request('/wait')),language='scala',execution_mode='http',timeout=.2)
    assert result.execution_results[0].status == Status.TIMEOUT
    monkeypatch.setattr(http,'STARTUP_TIMEOUT',.2)
    no_ready = 'object Main { def main(args: Array[String]): Unit = Thread.sleep(10000) }'
    result = verify(no_ready,suite(request()),language='scala',execution_mode='http')
    assert result.execution_results[0].status == Status.TIMEOUT


@requires_scala
def test_unfinished_chunked_body_reports_headers_received_before_timeout(monkeypatch):
    monkeypatch.setenv('SKILLFORGE_SCALA_HTTP_LOCAL', '1')
    source = SERVER.replace('if(body.length == 0) -1 else body.length', '0')
    source = source.replace('exchange.getResponseBody.write(body)', '()').replace('exchange.close()', '()')
    result = verify(source, suite(request('/unknown',status=404,expected='missing')),
                    language='scala',execution_mode='http',timeout=.3)
    assert result.execution_results[0].status == Status.TIMEOUT
    assert 'Received response status 404' in result.execution_results[0].stderr
    assert 'body did not complete' in result.execution_results[0].stderr


def test_docker_runner_uses_no_network_and_always_removes_container(monkeypatch,tmp_path):
    monkeypatch.delenv('SKILLFORGE_SCALA_HTTP_LOCAL',raising=False)
    commands = []
    def execute(command,*args,**kwargs):
        commands.append(command)
        assert command[command.index('--network')+1] == 'none'
        assert '--read-only' in command and '--cap-drop' in command
        assert '-p' not in command and '--privileged' not in command
        return http.ExecutionResult(Status.PASS,json.dumps([{'status':'pass','stdout':'','stderr':'','exit_code':0}]),'',0)
    monkeypatch.setattr(http,'execute',execute)
    monkeypatch.setattr(http.subprocess,'run',lambda command,**kwargs: commands.append(command))
    results = http.run_http(tmp_path/'candidate.jar',[[request()]],'java',1)
    assert results[0].status == Status.PASS
    assert commands[-1][:3] == ['docker','rm','-f']
    assert commands[-1][-1] == commands[0][commands[0].index('--name')+1]


def test_container_failure_is_an_infrastructure_error(monkeypatch,tmp_path):
    monkeypatch.delenv('SKILLFORGE_SCALA_HTTP_LOCAL',raising=False)
    monkeypatch.setattr(http,'execute',lambda *args,**kwargs:http.ExecutionResult(Status.ERROR,'','daemon unavailable',1))
    cleanup=[]
    monkeypatch.setattr(http.subprocess,'run',lambda command,**kwargs:cleanup.append(command))
    with pytest.raises(RuntimeError,match='container failed'):
        http.run_http(tmp_path/'candidate.jar',[[request()]],'java',1)
    assert cleanup


def test_unexpected_local_harness_exception_never_becomes_a_candidate_grade(monkeypatch,tmp_path):
    monkeypatch.setenv('SKILLFORGE_SCALA_HTTP_LOCAL','1')
    def crash(*args):
        raise TypeError('internal fixture failure')
    monkeypatch.setattr(http,'_scenario',crash)
    with pytest.raises(http.HTTPInfrastructureError,match='Infrastructure error.*TypeError'):
        http.run_http(tmp_path/'candidate.jar',[[request()]],'java',1)


def test_container_harness_error_is_reported_without_candidate_results(monkeypatch,tmp_path):
    monkeypatch.delenv('SKILLFORGE_SCALA_HTTP_LOCAL',raising=False)
    monkeypatch.setattr(http,'execute',lambda *a,**k:http.ExecutionResult(Status.PASS,
        json.dumps({'infrastructure_error':'Infrastructure error: test harness crashed'}),'',0))
    monkeypatch.setattr(http.subprocess,'run',lambda *a,**k:None)
    with pytest.raises(http.HTTPInfrastructureError,match='Infrastructure error'):
        http.run_http(tmp_path/'candidate.jar',[[request()]],'java',1)


def test_stale_docker_verifier_is_detected_before_model_work(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(scala.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout='old-source-hash'))
    scala._check_container_runner.cache_clear()
    with pytest.raises(RuntimeError,match='Docker verifier is stale'):
        scala._check_container_runner('fixture-image',scala.http_runner_hash())
    monkeypatch.setattr(scala.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout=scala.http_runner_hash()))
    scala._check_container_runner('fixture-image',scala.http_runner_hash())
    scala._check_container_runner.cache_clear()


@requires_scala
def test_wrong_status_with_unchecked_body_reports_failure_without_crashing(monkeypatch):
    monkeypatch.setenv('SKILLFORGE_SCALA_HTTP_LOCAL','1')
    cases = suite(dict(method='GET',path='/unknown',expected_status=200,expected_body=None))
    result = verify(SERVER,cases,language='scala',execution_mode='http')
    assert result.passed == 0
    assert result.execution_results[0].status == Status.FAIL
    assert 'body <not checked>' in result.failures[0]['stderr']
    assert 'got status 404' in result.failures[0]['stderr']


@requires_scala
def test_streaming_response_cannot_extend_the_request_deadline(monkeypatch):
    monkeypatch.setenv('SKILLFORGE_SCALA_HTTP_LOCAL', '1')
    trickle = SERVER.replace('exchange.getResponseBody.write(body)',
        'body.foreach { byte => exchange.getResponseBody.write(byte.toInt); '
        'exchange.getResponseBody.flush(); Thread.sleep(50) }')
    result = verify(trickle, suite(request(body='abcdefghijklmnop')), language='scala',
                    execution_mode='http', timeout=.2)
    assert result.execution_results[0].status == Status.TIMEOUT
    assert result.execution_results[0].duration_seconds < 1.5


@requires_scala
@pytest.mark.skipif(os.environ.get('SKILLFORGE_TEST_DOCKER') != '1', reason='opt in with a prepared Docker image')
def test_isolated_http_image_runs_real_requests_and_leaves_no_container(monkeypatch):
    monkeypatch.delenv('SKILLFORGE_SCALA_HTTP_LOCAL', raising=False)
    cases = suite(request(body='  α😀\n'), request('/count', expected='2'),
                  request('/unknown', status=404, expected='missing'))
    result = verify(SERVER, cases, language='scala', execution_mode='http')
    assert result.pass_rate == 1, result.failures
    environment = result.execution_results[0].execution_environment
    assert environment['http_executor'] == scala.HTTP_IMAGE
    assert environment['http_image_id'].startswith('sha256:')
    assert scala.JDK_VERSION in environment['http_jvm']
    # Regression: an unchecked body combined with a real status mismatch used
    # to crash the image's failure-message formatter instead of returning FAIL.
    unchecked = suite(dict(method='GET',path='/unknown',expected_status=200,expected_body=None))
    failed = verify(SERVER, unchecked, language='scala', execution_mode='http')
    assert failed.execution_results[0].status == Status.FAIL
    assert 'body <not checked>' in failed.failures[0]['stderr']
    assert 'got status 404' in failed.failures[0]['stderr']
    containers = subprocess.run(['docker', 'ps', '-q', '--filter',
                                 'ancestor=' + environment['http_image_id']],
                                capture_output=True, text=True, check=True, timeout=15)
    assert containers.stdout.strip() == ''


@requires_scala
def test_fixed_length_response_cannot_pass_with_missing_bytes(monkeypatch):
    monkeypatch.setenv('SKILLFORGE_SCALA_HTTP_LOCAL','1')
    source=SERVER.replace('if(body.length == 0) -1 else body.length','body.length + 1')
    result=verify(source,suite(request(body='hi')),language='scala',execution_mode='http')
    assert result.passed==0
    assert result.execution_results[0].status==Status.ERROR
    assert 'IncompleteRead' in result.failures[0]['stderr']


@requires_scala
def test_repair_evidence_includes_distinct_requests_after_recoverable_error(monkeypatch):
    monkeypatch.setenv('SKILLFORGE_SCALA_HTTP_LOCAL','1')
    source=SERVER.replace('if(body.length == 0) -1 else body.length','if(path=="/count") body.length + 1 else body.length')
    cases=build_test_suite([dict(requests=[
        dict(method='GET',path='/count',expected_status=200,expected_body='1'),
        request(body='actual',expected='different')]),
        dict(requests=[dict(method='GET',path='/count',expected_status=200,expected_body='1')])],execution_mode='http')
    result=verify(source,cases,language='scala',execution_mode='http')
    assert len(result.failures)==2,result.failures
    first=json.loads(result.failures[0]['actual']);expected=json.loads(result.failures[0]['expected'])
    assert first['declared_body_bytes']==2 and first['received_body_bytes']==1
    assert expected['utf8_bytes']==1
    assert json.loads(result.failures[1]['input'])['request']['path']=='/echo'
    assert first['headers']
