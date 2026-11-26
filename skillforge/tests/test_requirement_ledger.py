from types import SimpleNamespace
import pytest
from src.evaluation.requirement_ledger import build_ledger
from src.verifier.http_requirements import extract_http_requirements,execution_ledger
from src.agents.test_generator import build_test_suite
from src.agents.task_spec import TaskSpec
from src.verifier.outcome_verifier import verify
from src.evaluation.pipeline import evaluate_candidate
from src.verifier.runners.base import ExecutionResult,Status
from tests.test_http_execution import SERVER
from tests.test_scala_execution import requires_scala


def test_static_hints_cannot_promote_unverified_and_failure_dominates_pass():
    req=dict(id='R-any',text='obligation',source_quote='obligation',capability='unclassified',selector={})
    rows=[dict(requirement_id='R-any',type='static_check',result='pass')]
    assert not build_ledger([req],rows)['coverage_complete']
    rows.append(dict(requirement_id='R-any',type='executed_test',result='pass'))
    assert build_ledger([req],rows)['all_verified']
    rows.append(dict(requirement_id='R-any',type='executed_test',result='fail'))
    ledger=build_ledger([req],rows)
    assert ledger['requirements'][0]['state']=='failed' and ledger['coverage_complete'] and not ledger['all_verified']


def test_table_inventory_and_request_count_are_separate_requirements():
    spec='GET /alpha Return value\nGET /status Return READY\n- Support at least 75 sequential requests.\n- Ensure thread-safe state.'
    reqs,endpoints=extract_http_requirements(spec)
    assert {(e['method'],e['path']) for e in endpoints}=={('GET','/alpha'),('GET','/status')}
    ledger=execution_ledger(reqs,None,SimpleNamespace(case_results=[]),'@volatile var value = 0')
    assert ledger['coverage']==0
    assert any(r['capability']=='sequential_requests' and r['selector']['minimum']==75 for r in reqs)
    assert any(e['type']=='static_check' for r in ledger['requirements'] for e in r['evidence'])


def test_endpoint_hit_without_body_assertion_is_not_verified():
    reqs,_=extract_http_requirements('GET /ready Return YES')
    row=dict(request_number=1,method='GET',path='/ready',expected_status=200,status=200,passed=True,
             body_checked=False,unicode=False,newlines=False,parallel_group=0)
    result=ExecutionResult(Status.PASS,'','',0,http_observations=[row])
    ledger=execution_ledger(reqs,None,SimpleNamespace(case_results=[SimpleNamespace(execution=result)]),'')
    assert next(r for r in ledger['requirements'] if r['capability']=='endpoint')['state']=='unverified'


@requires_scala
def test_real_parallel_requests_and_post_group_state_check(monkeypatch):
    monkeypatch.setenv('SKILLFORGE_SCALA_HTTP_LOCAL','1')
    suite=build_test_suite([dict(requests=[
        dict(method='POST',path='/echo',body='α',expected_status=200,expected_body='α',parallel_group=1),
        dict(method='POST',path='/echo',body='β',expected_status=200,expected_body='β',parallel_group=1),
        dict(method='GET',path='/count',expected_status=200,expected_body=None,expected_body_one_of=['3'])])],execution_mode='http')
    result=verify(SERVER,suite,language='scala',execution_mode='http')
    assert result.passed==1,result.failures
    reqs,_=extract_http_requirements('- Ensure thread-safe state.')
    ledger=execution_ledger(reqs,suite,result,'var value=0')
    assert next(r for r in ledger['requirements'] if r['capability']=='concurrency')['state']=='verified'
    assert result.execution_results[0].http_startup['parallel_requests']==2


@requires_scala
@pytest.mark.parametrize('dependency',[True,False])
def test_clean_environment_detects_harness_dependency(monkeypatch,dependency):
    monkeypatch.setenv('SKILLFORGE_SCALA_HTTP_LOCAL','1')
    source=SERVER.replace('("127.0.0.1", 0)','("127.0.0.1", 18081)')
    if dependency:
        source=source.replace('val server =', 'val requiredSetting = sys.env("SKILLFORGE_PORT_FILE")\n    val server =')
    else:source='\n'.join(line for line in source.splitlines() if 'SKILLFORGE_PORT_FILE' not in line)
    suite=build_test_suite([dict(clean_environment=True,port=18081,requests=[
        dict(method='GET',path='/count',expected_status=200,expected_body='1')])],execution_mode='http')
    result=verify(source,suite,language='scala',execution_mode='http')
    reqs,_=extract_http_requirements('GET /count Return request count\n- Listen on localhost:18081.')
    ledger=execution_ledger(reqs,suite,result,source)
    state=next(r for r in ledger['requirements'] if r['capability']=='clean_environment')['state']
    assert state==('failed' if dependency else 'verified'),result.failures
    if not dependency:assert ledger['all_verified'],ledger


@requires_scala
def test_same_model_judge_disabled_and_missing_endpoint_caps_confidence(monkeypatch):
    monkeypatch.setenv('SKILLFORGE_SCALA_HTTP_LOCAL','1')
    spec=TaskSpec('GET /echo Return input\nGET /other Return another value','other','')
    suite=build_test_suite([dict(requests=[dict(method='GET',path='/echo',body='x',expected_status=200,expected_body='x')])],execution_mode='http')
    verification=verify(SERVER,suite,language='scala',execution_mode='http')
    class Judge:
        model='same'
        def generate(self,*a,**k):pytest.fail('Same model cannot be an independent judge')
    report,_=evaluate_candidate(spec,SERVER,verification,'generated',judge_client=Judge(),generator_model='same',
                                language='scala',execution_mode='http',test_suite=suite)
    assert report.status.value=='review' and report.evidence_confidence<1
    assert report.judge_grade is None and 'Same-model' in report.judge_error
    assert next(r for r in report.constraint_grade.requirement_ledger['requirements'] if r['selector'].get('path')=='/other')['state']=='unverified'
