"""Generated suites must survive a blind derivation before influencing repairs."""

import json

import pytest

from src.agents.task_spec import TaskSpec
from src.agents.test_generator import generate_tests
from src.agents.test_review import test_review_schema as review_schema
from src.agents.test_review import review_generated_tests
from src.agents.test_generator import build_test_suite
from src.agents.llm_client import CallCountingClient, RecordingClient
from src.history import ExperimentStore
from src.reinforcement_loop import run_candidate_refinement_loop


def case(stdin='8 3', answer='5', category='normal'):
    return dict(input=stdin, expected_output=answer, category=category)


def review(number=1, answer='5', **changes):
    return dict(test_number=number, input_valid=True, output_verified=True,
                expected_output=answer, category_valid=True,
                reason='Eight minus three equals five.', **changes)


class SequenceClient:
    model = 'fixture-model'

    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def generate_json(self, prompt, **kwargs):
        self.requests.append(dict(prompt=prompt, **kwargs))
        return json.dumps(next(self.responses))


def spec():
    return TaskSpec('Read two integers and print the first minus the second.', 'math', '')


def builder_response(cases):
    return dict(approach='Materialize the fixture cases.',
                code='import json; print(json.dumps(' + repr(cases) + '))')


def test_disagreement_regenerates_suite_without_leaking_targets_into_review():
    client = SequenceClient([[case(answer='999999')], [review()], [case()], [review()]])
    stages = []
    suite = generate_tests(client, spec(), on_progress=lambda stage, data: stages.append((stage, data)))
    assert suite.cases[0].expected_output == '5'
    assert len(client.requests) == 4
    first_review = client.requests[1]
    assert first_review['schema'] == review_schema(1)
    assert '999999' not in first_review['prompt']
    assert 'Previous solution' not in first_review['prompt']
    assert 'disagrees with blind derivation' in client.requests[2]['prompt']
    assert [stage for stage, _ in stages].count('tests_rejected') == 1


@pytest.mark.parametrize(('changes', 'message'), [
    ({'input_valid': False}, 'input rejected'),
    ({'category_valid': False}, 'category rejected'),
])
def test_review_rejections_stop_before_any_suite_is_accepted(changes, message):
    bad_review = review()
    bad_review.update(changes)
    bad_review['input_rule_quote'] = spec().problem_statement
    client = SequenceClient([[case()], [bad_review]])
    with pytest.raises(ValueError, match=message):
        generate_tests(client, spec(), max_generation_attempts=1)
    assert len(client.requests) == 2


def test_incomplete_or_duplicate_review_cannot_skip_a_case():
    for reviews in ([review()], [review(), review()]):
        client = SequenceClient([[case(), case('9 2', '7')], reviews])
        with pytest.raises(ValueError, match='every.*case exactly once'):
            generate_tests(client, spec(), max_generation_attempts=1)


def test_schema_shaped_transport_wrapper_still_validates_every_review():
    wrapper = dict(type='array', minItems=1, items=[review()])
    result = review_generated_tests(SequenceClient([wrapper]), spec().problem_statement,
                                    build_test_suite([case()]))
    assert result[0]['expected_output'] == '5'
    wrapper['items'] = []
    with pytest.raises(ValueError, match='every generated case'):
        review_generated_tests(SequenceClient([wrapper]), spec().problem_statement,
                               build_test_suite([case()]))


@pytest.mark.parametrize('changes', [
    {'test_number': True}, {'input_valid': 'true'}, {'output_verified': 1},
    {'expected_output': None}, {'reason': ''}, {'unexpected': 'extra'},
])
def test_malformed_review_is_not_treated_as_approval(changes):
    malformed = review()
    malformed.update(changes)
    client = SequenceClient([[case()], [malformed]])
    with pytest.raises(ValueError):
        generate_tests(client, spec(), max_generation_attempts=1)


def test_valid_empty_output_is_distinct_from_an_unverified_answer():
    task = TaskSpec('Read an integer; produce no output.', 'math', '')
    client = SequenceClient([[case('8', '')], [review(answer='')]])
    assert generate_tests(client, task).cases[0].expected_output == ''


@pytest.mark.parametrize('derived', [True, False])
def test_native_review_preserves_empty_answer_derivation_and_uncertainty(derived):
    response = dict(test_number=1, derivation='The specified result is empty.',
                    input_obeys_stated_rules=True, derived_stdout='',
                    answer_is_derived=derived, workload_category='normal')
    client = SequenceClient([[response], [response]])
    client.structured_test_review = True
    suite = build_test_suite([case('8', '')])
    reviews = review_generated_tests(client, 'Read an integer; produce no output.', suite,
                                     answers_supplied=False)
    assert reviews[0]['expected_output'] == ''
    assert reviews[0]['output_verified'] is derived
    assert 'answer_is_derived' in client.requests[0]['schema']['items']['required']
    assert '"expected_output"' not in client.requests[0]['prompt']
    if not derived:
        with pytest.raises(ValueError, match='small-case answer unverified'):
            review_generated_tests(client, 'Read an integer; produce no output.', suite,
                                   answers_supplied=False, require_blind_answers=True)


def test_native_review_cannot_mix_wire_names_or_ignore_invalid_inputs():
    client = SequenceClient([[review()]])
    client.structured_test_review = True
    suite = build_test_suite([case()])
    with pytest.raises(ValueError, match='fields do not match'):
        review_generated_tests(client, spec().problem_statement, suite)
    response = dict(test_number=1, derivation='The input violates the stated rule.',
                    input_obeys_stated_rules=False, derived_stdout='',
                    answer_is_derived=False, workload_category='normal',
                    input_rule_quote=spec().problem_statement)
    client = SequenceClient([[response]])
    client.structured_test_review = True
    with pytest.raises(ValueError, match='input rejected'):
        review_generated_tests(client, spec().problem_statement, suite)


def test_native_category_classification_cannot_accept_a_trivial_stress_case():
    response = dict(test_number=1, derivation='This input requires only one operation.',
                    input_obeys_stated_rules=True, derived_stdout='5',
                    answer_is_derived=True, workload_category='edge')
    client = SequenceClient([[response]])
    client.structured_test_review = True
    suite = build_test_suite([case(category='stress')])
    with pytest.raises(ValueError, match='Assigned stress; reviewer classified edge'):
        review_generated_tests(client, spec().problem_statement, suite)
    assert suite.cases[0].category == 'stress'


def test_line_boundaries_in_answers_remain_significant():
    client = SequenceClient([[case(answer='5\n7')], [review(answer='5 7')]])
    with pytest.raises(ValueError, match='disagrees'):
        generate_tests(client, spec(), max_generation_attempts=1)


@pytest.mark.parametrize('quote', ['', 'Both numbers must be positive.'])
def test_invented_input_rule_is_advisory_and_never_verifies_an_answer(quote):
    objection = review()
    objection.update(input_valid=False, input_rule_quote=quote)
    reviews = review_generated_tests(SequenceClient([[objection]]), spec().problem_statement,
                                    build_test_suite([case()]), answers_supplied=False)
    assert reviews[0]['unsupported_input_objection']
    assert not reviews[0]['output_verified']


def test_rejected_answers_preserve_inputs_for_execution_without_grading():
    client = SequenceClient([[case(answer='wrong')], [review()]])
    with pytest.raises(ValueError, match='disagrees') as failure:
        generate_tests(client, spec(), max_generation_attempts=1)
    assert failure.value.suite.cases[0].input == '8 3'
    assert failure.value.suite.validation == {}


def test_rejected_suite_still_drives_bounded_runtime_repair(tmp_path):
    class Solver:
        model = 'fixture-solver'
        def __init__(self):
            self.repairs = []
        def generate(self, prompt, **kwargs):
            if 'problem analyzer' in kwargs.get('system', ''):
                return json.dumps(dict(selected_max_attempts=2, attempt_budget_reason='Fixture.'))
            if 'Previous solution:' in prompt:
                self.repairs.append(prompt)
                return 'print(5)'
            return 'raise RuntimeError("visible diagnostic")'
    solver = Solver()
    # Source validation succeeds, but execution of the rejected suite crashes.
    tests = SequenceClient([[case(answer='wrong')], [review()]] * 2 +
                           [builder_response([case(answer='wrong')]), [review()]])
    with ExperimentStore(tmp_path / 'history.db') as store:
        result = run_candidate_refinement_loop(solver, spec().problem_statement,
            test_client=tests, max_attempts=2, history_store=store)
        assert result.validation_error and not result.attempts
        assert result.best_code == 'print(5)'
        assert len(solver.repairs) == 1
        assert 'visible diagnostic' in solver.repairs[0]
        assert store._connection.execute('SELECT COUNT(*) FROM runs').fetchone()[0] == 0
        assert any(d['phase'] == 'unverified_tests' for d in result.diagnostics)


def test_semantic_review_retry_budget_is_bounded():
    client = SequenceClient([[case(answer='wrong')], [review()]] * 2 +
                            [builder_response([case(answer='wrong')]), [review()]])
    with pytest.raises(ValueError, match='after 3 attempts'):
        generate_tests(client, spec())
    assert len(client.requests) == 6


def test_inferred_examples_and_constraints_do_not_override_the_original_question():
    task = spec()
    task.examples = [{'input': 'invented input', 'output': 'invented output'}]
    task.constraints = 'invented constraints'
    client = SequenceClient([[case()], [review()]])
    generate_tests(client, task)
    for request in client.requests:
        assert task.problem_statement in request['prompt']
        assert 'invented' not in request['prompt']


def test_generation_and_review_have_distinct_audited_seeded_stages(tmp_path):
    raw = SequenceClient([[case()], [review()]])
    with ExperimentStore(tmp_path / 'history.db') as store:
        client = CallCountingClient(RecordingClient(raw, store, 'tests-owner', 'generator'), seed=42)
        generate_tests(client, spec())
        rows = store._connection.execute('SELECT * FROM model_requests ORDER BY call_id').fetchall()
    assert [row['stage'] for row in rows] == ['tests:1', 'test_validation:1']
    assert len({json.loads(row['request_json'])['seed'] for row in rows}) == 2
    assert client.call_count == 2


def test_rejected_suite_retains_an_ungraded_draft_without_creating_a_training_run(tmp_path):
    class BadTestsClient(SequenceClient):
        def generate_json(self, prompt, **kwargs):
            if 'problem analyzer' in kwargs.get('system', ''):
                return json.dumps(dict(selected_max_attempts=2, attempt_budget_reason='Fixture budget.'))
            return super().generate_json(prompt, **kwargs)

    client = BadTestsClient([[case(answer='wrong')], [review()]] * 2 +
                           [builder_response([case(answer='wrong')]), [review()]])
    class DraftClient:
        model = 'draft-generator'
        def generate(self, prompt, **kwargs):
            if 'problem analyzer' in kwargs.get('system', ''):
                return json.dumps(dict(selected_max_attempts=2, attempt_budget_reason='Fixture budget.'))
            return 'def draft():\n    return None'
    with ExperimentStore(tmp_path / 'history.db') as store:
        result = run_candidate_refinement_loop(DraftClient(), spec().problem_statement,
            test_client=client, task_spec_cache=None, history_store=store)
        assert not result.success and result.attempts == []
        assert 'disagrees' in result.validation_error
        assert result.best_code == 'def draft():\n    return None'
        assert store._connection.execute('SELECT COUNT(*) FROM runs').fetchone()[0] == 0
        assert store._connection.execute('SELECT COUNT(*) FROM attempts').fetchone()[0] == 0
        assert store._connection.execute('SELECT COUNT(*) FROM model_requests').fetchone()[0] == 8
    assert all('Previous solution' not in request['prompt'] for request in client.requests)
