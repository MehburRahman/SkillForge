"""Generic HTTP generation and real execution, with isolated model fixtures."""
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from src.agents.http_test_generator import generate_http_tests
from src.agents.test_generator import generate_tests
from src.agents.task_spec import TaskSpec
from src.benchmark.execution_mode import suggested_execution_mode
from src.verifier.outcome_verifier import verify
from tests.test_api import setup, finished
from tests.test_http_execution import SERVER
from tests.test_scala_execution import requires_scala
from tests.test_benchmark_runner import BenchmarkClient

BUILDER = '''def build_scenarios():
    return [dict(category='stress', requests=[dict(method='GET',path='/count') for _ in range(100)]),
            dict(category='normal',requests=[dict(method='POST',path='/echo',body='  α😀\\n'),
                                            dict(method='GET',path='/count')])]
'''
REFERENCE = '''def expected_responses(requests):
    responses = []
    count = 0
    for request in requests:
        count += 1
        body = str(count) if request['path'] == '/count' else request['body']
        responses.append(dict(status=200,body=body,headers={}))
    return responses
'''

class TestModel:
    __test__ = False
    model = 'test-reference-fixture'
    def __init__(self, responses=None):
        self.responses = iter(responses or [BUILDER, REFERENCE])
        self.requests = []
    def generate(self, prompt, **kwargs):
        self.requests.append(dict(prompt=prompt, **kwargs))
        return next(self.responses)

def test_http_inputs_have_executed_reference_answers_and_100_request_scenario():
    model = TestModel()
    suite = generate_tests(model, TaskSpec('The task-specific service contract.', 'other', ''), execution_mode='http')
    assert len(suite.cases) == 2
    assert len(suite.cases[0].requests) == 100
    assert suite.cases[0].requests[-1]['expected_body'] == '100'
    assert suite.cases[1].requests[0]['expected_body'] == '  α😀\n'
    assert suite.cases[1].requests[1]['expected_body'] == '2'
    assert suite.validation['answer_source'] == 'executed_model_generated_http_reference'
    assert not suite.validation['independent_verification']
    assert suite.validation['request_count'] == 102
    assert len(model.requests) == 2
    assert 'α😀' not in model.requests[1]['prompt']


def test_missing_sequence_coverage_is_retried_then_reported_without_blocking_execution():
    short = "def build_scenarios():\n    return [dict(category='normal', requests=[dict(method='GET',path='/anything')])]"
    reference = 'def expected_responses(requests):\n    return [dict(status=200,body=None) for request in requests]'
    model = TestModel([short,short,reference])
    suite = generate_http_tests(model, 'Support at least 75 sequential requests.', max_attempts=2)
    assert len(suite.cases[0].requests) == 1
    assert not suite.validation['coverage_complete']
    assert 'longest generated sequence has 1' in suite.validation['coverage_warning']
    assert 'repeating text' in model.requests[1]['prompt']
    suite = generate_http_tests(TestModel(), 'Support at least 75 sequential requests.')
    assert suite.validation['minimum_sequence_requests'] == 75
    assert suite.validation['sequence_coverage_complete']
    assert not suite.validation['coverage_complete']  # No execution evidence yet.
    assert suite.validation['coverage_quotes'] == ['at least 75 sequential requests']

@pytest.mark.parametrize('reference', [
    'def expected_responses(requests):\n    return []',
    '''def expected_responses(requests):
    expected_responses.n = getattr(expected_responses,'n',0) + 1
    return [dict(status=200,body=str(expected_responses.n),headers={}) for _ in requests]
''',
])
def test_invalid_or_state_leaking_references_do_not_certify_tests(reference):
    with pytest.raises(ValueError, match='Could not generate HTTP tests') as caught:
        generate_http_tests(TestModel([BUILDER, reference]), 'Task contract', max_attempts=1)
    assert caught.value.suite.validation == {}

def test_input_builder_answers_are_discarded_and_reference_still_computes_them():
    source = BUILDER.replace("dict(method='GET',path='/count')", "dict(method='GET',path='/count',expected_body='fake')")
    model = TestModel([source, REFERENCE])
    suite = generate_http_tests(model, 'Task contract', max_attempts=1)
    assert suite.cases[0].requests[-1]['expected_body'] == '100'
    assert suite.validation['discarded_input_answers'] == 101
    assert 'fake' not in model.requests[1]['prompt']


def test_reference_repairs_reuse_frozen_inputs_and_receive_the_actual_error():
    invalid = 'def expected_responses(requests):\n    value = 0\n    nonlocal value\n    return []'
    model = TestModel([BUILDER, invalid, REFERENCE])
    suite = generate_http_tests(model, 'Task contract')
    assert len(model.requests) == 3
    assert 'nonlocal' in model.requests[-1]['prompt']
    assert invalid in model.requests[-1]['prompt']
    assert suite.validation['input_generation_attempts'] == 1
    assert suite.validation['reference_generation_attempts'] == 2
    assert suite.cases[0].requests[-1]['expected_body'] == '100'


def test_optional_headers_and_unstated_header_assertions_do_not_block_the_suite():
    reference = REFERENCE.replace('headers={}', "headers={'X-Guess':'invented'}")
    suite = generate_http_tests(TestModel([BUILDER, reference]), 'Task contract')
    assert suite.validation['ignored_unstated_headers'] == 102
    assert all(not r['expected_headers'] for c in suite.cases for r in c.requests)
    suite = generate_http_tests(TestModel([BUILDER, REFERENCE.replace(',headers={}', '')]), 'Task contract')
    assert suite.validation['request_count'] == 102


def test_reference_cannot_use_network_to_copy_a_candidate_response():
    reference = 'import http.client\ndef expected_responses(requests):\n    return []'
    with pytest.raises(ValueError, match='without network or process IO'):
        generate_http_tests(TestModel([BUILDER, reference]), 'Task contract', max_attempts=1)

def test_http_mode_hint_has_no_application_specific_routes():
    assert suggested_execution_mode('Develop a small HTTP server in Scala.', 'scala') == 'http'
    assert suggested_execution_mode('Sort these strings.', 'scala') == 'stdio'
    assert suggested_execution_mode('An HTTP service.', 'python') == 'stdio'

@requires_scala
def test_generated_scenarios_run_against_real_server_and_detect_wrong_reference(monkeypatch):
    monkeypatch.setenv('SKILLFORGE_SCALA_HTTP_LOCAL','1')
    suite = generate_http_tests(TestModel(), 'Task contract')
    assert verify(SERVER, suite, language='scala', execution_mode='http').pass_rate == 1
    suite.cases[0].requests[-1]['expected_body'] = 'incorrect'
    assert verify(SERVER, suite, language='scala', execution_mode='http').pass_rate == .5

@requires_scala
def test_unspecified_response_body_is_distinct_from_required_empty_body(monkeypatch):
    from src.agents.test_generator import build_test_suite
    monkeypatch.setenv('SKILLFORGE_SCALA_HTTP_LOCAL','1')
    suite = build_test_suite([dict(category='edge', requests=[dict(method='GET',path='/unknown',
        expected_status=404,expected_body=None)])], execution_mode='http')
    assert verify(SERVER, suite, language='scala', execution_mode='http').passed == 1
    suite.cases[0].requests[0]['expected_body'] = ''
    assert verify(SERVER, suite, language='scala', execution_mode='http').passed == 0

@requires_scala
@pytest.mark.parametrize('incomplete', [False, True])
def test_api_generates_http_cases_without_user_tests_and_saves_results(setup, monkeypatch, incomplete):
    settings, _, _ = setup
    monkeypatch.setenv('SKILLFORGE_SCALA_HTTP_LOCAL','1')
    class Solver(BenchmarkClient):
        def generate(self,prompt,system='',**kwargs):
            if 'problem analyzer' in system:
                return super().generate(prompt,system,kwargs.get('temperature',.2),seed=kwargs.get('seed'))
            return SERVER
    tests = TestModel([BUILDER,BUILDER,BUILDER,REFERENCE,BUILDER] if incomplete else None)
    with TestClient(create_app(settings,client_factory=lambda **kw:(Solver(),None),
            test_client_factory=lambda name: tests)) as api:
        problem = 'Develop an HTTP server for this contract.' + (' Support at least 110 sequential requests.' if incomplete else '')
        response = api.post('/solves',json=dict(problem=problem,
            language='scala',model='fixture-solver',test_model=tests.model,max_attempts=1))
        assert response.status_code == 202, response.text
        # This integration check compiles Scala twice and launches real JVMs.
        job = finished(api,response.json()['job_id'],timeout=60)
        assert job['status'] == 'completed', job
        result = job['result']
        assert result['test_source'] == 'generated' and result['status'] == 'review', result
        assert result['test_model_calls'] == (5 if incomplete else 2)
        if incomplete:
            ledger=result['evaluation']['constraint_grade']['requirement_ledger']
            assert any(r['capability']=='sequential_requests' and r['state']=='unverified' for r in ledger['requirements'])
        assert result['test_validation']['request_count'] == 102
        assert result['evaluation']['functional_grade']['passed'] == 2
        saved = api.get('/runs/'+result['history_run_id']).json()
        assert saved['execution_mode'] == 'http'

@requires_scala
def test_missing_endpoint_triggers_targeted_generation_and_saved_execution_ledger(setup,monkeypatch):
    settings,_,_=setup
    monkeypatch.setenv('SKILLFORGE_SCALA_HTTP_LOCAL','1')
    class Solver(BenchmarkClient):
        def generate(self,prompt,system='',**kwargs):
            if 'problem analyzer' in system:
                return super().generate(prompt,system,kwargs.get('temperature',.2),seed=kwargs.get('seed'))
            return SERVER
    echo="def build_scenarios():\n    return [dict(category='normal',requests=[dict(method='POST',path='/echo',body='test')])]"
    count="def build_scenarios():\n    return [dict(category='normal',requests=[dict(method='GET',path='/count')])]"
    model=TestModel([echo,REFERENCE,count])
    statement='POST /echo Return the supplied text\nGET /count Return the request count'
    with TestClient(create_app(settings,client_factory=lambda **kw:(Solver(),None),test_client_factory=lambda name:model)) as api:
        response=api.post('/solves',json=dict(problem=statement,language='scala',execution_mode='http',test_model=model.model,max_attempts=1))
        assert response.status_code==202,response.text
        result=finished(api,response.json()['job_id'],timeout=60)['result']
        assert result['evaluation']['functional_grade']['total']==2,result
        assert result['test_model_calls']==3
        rows=result['evaluation']['constraint_grade']['requirement_ledger']['requirements']
        assert all(r['state']=='verified' for r in rows if r['capability']=='endpoint')
        assert next(r for r in rows if r['capability']=='clean_environment')['state']=='unverified'
        assert result['status']=='review' and not result['test_validation']['coverage_complete']
        assert 'Requirement IDs and capabilities to exercise' in model.requests[-1]['prompt']
        saved=api.get('/runs/'+result['history_run_id']).json()
        assert 'requirement_ledger' in str(saved)


def test_generated_overlong_workloads_keep_bounded_prefix_before_deriving_answers():
    builder=BUILDER.replace('range(100)','range(200)')
    model=TestModel([builder,REFERENCE])
    suite=generate_http_tests(model,'Support at least 100 sequential requests.')
    assert len(suite.cases[0].requests)==128
    assert suite.cases[0].requests[-1]['expected_body']=='128'
    assert suite.validation['budget_trimmed_requests']==72
    assert suite.validation['sequence_coverage_complete']
    assert not suite.validation['coverage_complete']
    assert len(model.requests)==2


def test_large_generated_scenarios_are_bounded_without_changing_model_inputs():
    builder='''def build_scenarios():
    return [dict(category='stress',requests=[dict(method='POST',path='/echo',body='x'*60000) for _ in range(8)]),
            dict(category='edge',requests=[dict(method='GET',path='/count')])]
'''
    suite=generate_http_tests(TestModel([builder,REFERENCE]),'Task contract')
    assert 1<=len(suite.cases[0].requests)<8
    assert all(r['body']=='x'*60000 for r in suite.cases[0].requests)
    assert suite.validation['budget_trimmed_requests']>0
    assert suite.cases[-1].requests[0]['expected_body']=='1'


def test_model_category_labels_do_not_block_valid_requests_or_certify_concurrency():
    builder=BUILDER.replace("category='stress'","category='concurrency'")
    suite=generate_http_tests(TestModel([builder,REFERENCE]),'Task contract')
    assert suite.cases[0].category=='normal'
    assert suite.validation['normalized_categories'][0]['original']=='concurrency'
    assert all(not request.get('parallel_group') for request in suite.cases[0].requests)
    assert not suite.validation['coverage_complete']


def test_native_http_inputs_use_structured_data_and_bounded_repeat_not_python():
    import json
    class Native:
        def __init__(self):self.json_calls=[]
        def generate_json(self,prompt,**kwargs):
            self.json_calls.append((prompt,kwargs))
            assert 'Return only a JSON array' in prompt
            return json.dumps([dict(category='stress',repeat=100,requests=[dict(method='GET',path='/count',body='')])])
        def generate(self,*args,**kwargs):return REFERENCE
    model=Native()
    suite=generate_http_tests(model,'Support at least 100 sequential requests.')
    assert len(suite.cases[0].requests)==100
    assert suite.cases[0].requests[-1]['expected_body']=='100'
    assert suite.validation['input_generation_format']=='json'
    assert len(model.json_calls)==1
