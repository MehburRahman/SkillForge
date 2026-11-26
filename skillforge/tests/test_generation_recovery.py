"""Generic draft retention, bounded construction and independent reference checks."""
from dataclasses import asdict
import json

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.jobs import JobStore
from api.models import SolveRequest
from src.agents.code_prompts import task_context
from src.agents.task_spec import parse_task, execution_problem_statement
from src.agents.task_spec_cache import task_spec_from_dict
from src.agents.test_builder import build_generated_tests
from src.agents.test_generator import build_test_suite, generate_tests
from src.agents.test_oracle import ReferenceAnchorError
from src.history import ExperimentStore
from src.reinforcement_loop import run_candidate_refinement_loop
from tests.test_api import setup, finished
from tests.test_executable_test_builder import PlanningClient, INPUT_BUILDER, program_responses, reviews, spec
from tests.test_executable_test_oracle import COUNTED_SUM_REFERENCE, repair
from tests.test_separate_test_model import SolverClient
from tests.test_task_spec import TaskAnalysisClient


def anchors():
    return build_test_suite([dict(input='2 157 251', expected_output='408', category='edge')])


def uncertain_reviews():
    return [dict(item, output_verified=False, expected_output='', reason='Cannot derive this answer confidently.')
            for item in reviews()]


def deferred_client(code=COUNTED_SUM_REFERENCE):
    return PlanningClient([*program_responses()[:3], uncertain_reviews(), code])


def test_stated_formats_use_the_verbatim_grammar_instead_of_a_shortened_paraphrase():
    excerpt = 'Read a count, then records: PUT key value or ERASE key. Any whitespace separates tokens.'
    problem = 'Process these records. ' + excerpt
    response = dict(input_format='Count followed by numbers', input_format_origin='stated',
                    input_format_source_excerpt=excerpt, output_format='One JSON integer',
                    output_format_origin='selected', output_format_source_excerpt='')
    task = parse_task(TaskAnalysisClient(response), problem)
    assert task.input_format == excerpt
    assert task_spec_from_dict(asdict(task), problem) == task
    for text in (execution_problem_statement(task), task_context(task)):
        assert excerpt in text and 'Count followed by numbers' not in text
        assert 'One JSON integer' in text
        assert text.count(excerpt) == 1


def test_unquoted_stated_formats_are_rejected_instead_of_overriding_the_problem():
    response = dict(input_format='A made-up encoding', input_format_origin='stated',
                    input_format_source_excerpt='This is not in the statement.')
    with pytest.raises(ValueError, match='quote the original format'):
        parse_task(TaskAnalysisClient(response), 'Read the supplied values.')


def test_uncertainty_is_resolved_by_a_reference_checked_against_withheld_verified_cases():
    client = deferred_client()
    events = []
    suite = generate_tests(client, spec(), method='program', reference_test_suite=anchors(),
        on_progress=lambda name, data: events.append((name, data)))
    assert [case.expected_output for case in suite.cases] == ['6', '9', '0']
    assert suite.validation == dict(status='reference_checked', uncertain_small_cases=3,
        independent_anchor_count=1, reference_anchors_verified=True)
    assert len(client.requests) == 5
    assert sum(name == 'reference_anchors_verified' for name, _ in events) == 1
    for request in client.requests:
        assert '157' not in request['prompt'] and '251' not in request['prompt']
        assert '408' not in request['prompt']


def test_unsupported_input_objection_cannot_turn_reference_outputs_into_verified_tests():
    objections = uncertain_reviews()
    objections[0].update(input_valid=False, input_rule_quote='All integers must be positive.')
    client = PlanningClient([*program_responses()[:3], objections, COUNTED_SUM_REFERENCE])
    suite = generate_tests(client, spec(), method='program', reference_test_suite=anchors())
    assert suite.validation['status'] == 'needs_review'
    assert suite.validation['unsupported_input_objections'] == [1]
    assert suite.validation['reference_anchors_verified']
    assert [case.expected_output for case in suite.cases] == ['6', '9', '0']


def test_reference_disagreement_with_verified_cases_stops_without_answer_driven_repairs():
    wrong = COUNTED_SUM_REFERENCE.replace('str(sum(values[1:]))', 'str(sum(values[1:]) + 1)')
    client = deferred_client(wrong)
    with pytest.raises(ReferenceAnchorError, match='independently verified case'):
        generate_tests(client, spec(), method='program', reference_test_suite=anchors())
    assert len(client.requests) == 5


def test_repaired_reference_is_rechecked_against_the_same_verified_cases():
    broken = COUNTED_SUM_REFERENCE.replace('return str(sum(values[1:]))',
        'return str(sum(values[1:])) if count else str(values[1])')
    wrong_repair = COUNTED_SUM_REFERENCE.replace('str(sum(values[1:]))', 'str(sum(values[1:]) + 1)')
    client = PlanningClient([*program_responses()[:4], broken, repair(broken, wrong_repair)])
    with pytest.raises(ReferenceAnchorError, match='independently verified case'):
        generate_tests(client, spec(), method='program', reference_test_suite=anchors())
    assert len(client.requests) == 6
    assert '408' not in client.requests[-1]['prompt']


def test_grounded_statement_examples_can_check_the_reference_without_invented_anchors():
    task = spec()
    excerpt = 'Example input: 1 13. Output: 13.'
    task.problem_statement += ' ' + excerpt
    task.examples = [dict(input='1 13', output='13', source_excerpt=excerpt)]
    suite = generate_tests(deferred_client(), task, method='program')
    assert suite.validation['reference_anchors_verified']
    assert suite.validation['independent_anchor_count'] == 1


def test_uncertainty_without_verified_cases_retains_a_draft_without_grading_or_training(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Unverified test answers must never grade a draft')
    monkeypatch.setattr('src.reinforcement_loop.verify', forbidden)
    monkeypatch.setattr('src.reinforcement_loop.evaluate_candidate', forbidden)
    stages = []
    with ExperimentStore(tmp_path / 'history.sqlite3') as store:
        result = run_candidate_refinement_loop(SolverClient(), spec().problem_statement,
            test_client=deferred_client(), test_generation_method='program', max_attempts=1,
            history_store=store, on_stage=lambda name, data: stages.append((name, data)))
        assert store._connection.execute('SELECT COUNT(*) FROM runs').fetchone()[0] == 0
        assert store._connection.execute('SELECT COUNT(*) FROM attempts').fetchone()[0] == 0
        records = store._connection.execute('SELECT stage,response_text FROM model_requests ORDER BY call_id').fetchall()
    assert not result.success and result.attempts == [] and result.best_reward == 0
    assert result.best_attempt is None and result.best_code == records[1]['response_text']
    assert result.validation_error and result.test_suite.validation['status'] == 'needs_review'
    names = [name for name, _ in stages]
    assert names.index('draft_ready') < names.index('test_generation')
    assert result.model_calls == len(records) == 7


@pytest.mark.parametrize('code, message', [
    ('def build_inputs():\n    return "[]"', 'non-empty list'),
    ('def build_inputs():\n    return [dict(input="x" * 2049, category="normal")]', 'construction budget'),
    ('def build_inputs():\n    return [dict(input="😀" * 800 + str(i), category="stress") for i in range(6)]', '16 KiB'),
    ('def build_inputs():\n    print("[]")\n    return []', 'without print'),
])
def test_host_serialization_rejects_bad_or_oversized_builder_results_without_truncating(code, message):
    with pytest.raises(ValueError, match=message):
        build_generated_tests(PlanningClient([code]), spec().problem_statement, None, '', input_only=True)


def test_api_preserves_an_unverified_draft_after_test_generation_fails(setup):
    settings, _, _ = setup
    class OfflineTests:
        model = 'offline-test-model'
        def generate(self, *args, **kwargs):
            raise RuntimeError('Test model unavailable')
    with TestClient(create_app(settings, client_factory=lambda **kwargs: (SolverClient(), None),
                               test_client_factory=lambda name: OfflineTests())) as client:
        response = client.post('/solves', json=dict(problem=spec().problem_statement,
            model='candidate:7b', test_model='offline-test-model', max_attempts=1))
        job = finished(client, response.json()['job_id'])
        assert job['status'] == 'completed' and job['result']['status'] == 'unverified'
        assert job['result']['evaluation'] is None and job['result']['attempt_count'] == 0
        assert job['result']['code'] and 'Test model unavailable' in job['result']['validation_error']
        assert job['resource_id'] is None and client.get('/runs').json()['total'] == 0
        saved = client.get('/jobs/' + job['job_id']).json()
        assert saved['result'] == job['result']
        assert saved['draft_code'] == job['result']['code']
        assert 'draft_ready' in client.get('/jobs/' + job['job_id'] + '/events').text


def test_job_snapshot_retains_the_latest_draft_when_the_worker_is_interrupted(tmp_path):
    path = tmp_path / 'jobs.sqlite3'
    store = JobStore(path)
    store.initialize()
    job = store.enqueue('solve', {}, 8)
    store.claim()
    store.emit(job['job_id'], 'stage', dict(stage='draft_ready', code='old draft'))
    store.emit(job['job_id'], 'stage', dict(stage='draft_ready', code='saved draft'))
    store.emit(job['job_id'], 'stage', dict(stage='test_generation'))
    restarted = JobStore(path)
    restarted.initialize()
    saved = restarted.get(job['job_id'])
    assert saved['status'] == 'interrupted' and saved['result'] is None
    assert saved['draft_code'] == 'saved draft' and saved['resource_id'] is None


def test_api_falls_back_to_verified_cases_when_generated_reference_disagrees(setup):
    settings, _, _ = setup
    wrong = COUNTED_SUM_REFERENCE.replace('str(sum(values[1:]))', 'str(sum(values[1:]) + 1)')
    tests_client = deferred_client(wrong)
    with TestClient(create_app(settings, client_factory=lambda **kwargs: (SolverClient(), None),
                               test_client_factory=lambda name: tests_client)) as client:
        response = client.post('/solves', json=dict(problem=spec().problem_statement,
            model='candidate:7b', test_model='reference-model', max_attempts=1,
            reference_tests=[asdict(case) for case in anchors().cases]))
        assert response.status_code == 202, response.text
        job = finished(client, response.json()['job_id'])
        assert job['status'] == 'completed' and job['result']['status'] == 'pass'
        validation = job['result']['test_validation']
        assert validation['status'] == 'trusted_samples_only'
        assert 'independently verified case' in validation['generated_tests_error']
        assert job['draft_code'] == job['result']['code']
        assert job['result']['evaluation']['functional_grade']['total'] == 1
        assert job['result']['attempt_count'] == 1 and job['result']['test_source'] == 'trusted'
        assert client.get('/runs').json()['total'] == 1
    assert len(tests_client.requests) == 5


@pytest.mark.parametrize('options', [dict(task_id='catalog-task'), dict(tests=[dict(input='x', expected_output='y')]),
                                    dict(execution_mode='function', entry_point='solve')])
def test_reference_cases_cannot_be_mixed_with_incompatible_execution_contracts(options):
    payload = dict(reference_tests=[dict(input='x', expected_output='y')], **options)
    if 'task_id' not in options:
        payload['problem'] = 'Process input.'
    with pytest.raises(ValueError, match='reference_tests require'):
        SolveRequest(**payload)
