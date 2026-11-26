from types import SimpleNamespace
import pytest
from src.agents import test_generator as generator


def setup_dispute(monkeypatch, outputs):
    created = []
    class Oracle:
        def __init__(self, client, statement, **kwargs):
            created.append((statement, kwargs))
        def derive(self, suite):
            return [{'expected_output': answer} for answer in outputs]
    monkeypatch.setattr(generator, 'ExecutableTestOracle', Oracle)
    suite = generator.TestSuite([generator.TestCase('opaque input', 'correct', 'normal')])
    reviews = [dict(test_number=1, output_verified=True, expected_output='incorrect')]
    return suite, reviews, SimpleNamespace(anchor_suite=None), created


def test_executable_agreement_resolves_text_dispute_without_revealing_answers(monkeypatch):
    suite, reviews, oracle, created = setup_dispute(monkeypatch, ['correct'])
    audit = generator.reconcile_program_answers(None, 'Task contract', suite, reviews, oracle)
    assert audit[0]['textual_answer'] == 'incorrect'
    assert reviews[0]['answer_source'] == 'executable_agreement'
    assert suite.cases[0].expected_output == 'correct'
    assert 'opaque input' not in created[0][0]
    assert 'incorrect' not in created[0][0]
    assert 'correct' not in created[0][0]


def test_executable_disagreement_does_not_take_textual_majority(monkeypatch):
    suite, reviews, oracle, _ = setup_dispute(monkeypatch, ['incorrect'])
    with pytest.raises(generator.UnresolvedAnswerDispute, match='references disagree'):
        generator.reconcile_program_answers(None, 'Task contract', suite, reviews, oracle)
    assert suite.cases[0].expected_output == 'correct'
    assert reviews[0]['expected_output'] == 'incorrect'


def test_missing_adjudicator_results_remain_unverified(monkeypatch):
    suite, reviews, oracle, _ = setup_dispute(monkeypatch, [])
    with pytest.raises(generator.UnresolvedAnswerDispute):
        generator.reconcile_program_answers(None, 'Task contract', suite, reviews, oracle)


def test_dispute_adjudication_runs_fresh_programs_without_answer_feedback():
    from tests.test_executable_test_builder import PlanningClient, program_responses, spec, reviews
    from tests.test_executable_test_oracle import COUNTED_SUM_REFERENCE
    responses = program_responses()
    responses[3] = reviews(('wrong', '9', '0'))
    client = PlanningClient([*responses, *program_responses()[:2], COUNTED_SUM_REFERENCE])
    suite = generator.generate_tests(client, spec(), method='program')
    assert suite.validation['status'] == 'model_agreement'
    assert suite.validation['answer_disputes'][0]['textual_answer'] == 'wrong'
    assert [c.expected_output for c in suite.cases] == ['6', '9', '0']
    assert len(client.requests) == 8
    assert all('wrong' not in r['prompt'] for r in client.requests[5:])


def test_repeated_program_inputs_stop_before_another_review():
    from tests.test_executable_test_builder import PlanningClient, program_responses, spec, INPUT_BUILDER, reviews
    bad_builder = INPUT_BUILDER.replace('len(values)', 'len(values) + 1')
    responses = program_responses()
    client = PlanningClient([*responses[:2], bad_builder, reviews(), responses[4], bad_builder])
    with pytest.raises(generator.OracleBudgetExhausted, match='repeated a rejected suite'):
        generator.generate_tests(client, spec(), method='program')
    assert len(client.requests) == 6


def test_compile_repairs_have_separate_budget_and_unchanged_source_escalates(monkeypatch):
    import json
    from src.reinforcement_loop import run_candidate_refinement_loop
    suite = generator.build_test_suite([dict(input='2 3', expected_output='5', category='normal')])
    monkeypatch.setattr('src.reinforcement_loop.generate_tests', lambda *a, **k: suite)
    class Client:
        model = 'fixture'
        def __init__(self):
            self.responses = iter(['def broken(', 'def broken(', 'print(0)', 'print(5)'])
            self.prompts = []
        def generate(self, prompt, **kwargs):
            if 'problem analyzer' in kwargs.get('system', '').lower():
                return json.dumps(dict(selected_max_attempts=2, attempt_budget_reason='Fixture.'))
            self.prompts.append((prompt, kwargs))
            return next(self.responses)
    client = Client()
    stages = []
    result = run_candidate_refinement_loop(client, 'Read two integers and print their sum.',
        max_attempts=2, on_stage=lambda name, data: stages.append((name, data)))
    assert result.success
    assert result.diagnostic_repairs == 2
    assert len(result.attempts) == 2
    assert any(name == 'diagnostic_duplicate' for name, _ in stages)
    assert len(client.prompts) == 4
    assert 'SyntaxError' in client.prompts[1][0]


def test_native_program_inputs_are_validated_before_answer_review_and_repaired():
    from tests.test_executable_test_builder import PlanningClient, program_responses, spec, INPUT_BUILDER, reviews
    from tests.test_executable_test_oracle import COUNTED_SUM_REFERENCE
    validator = '''def solve_input(input_text: str) -> str:
    values = list(map(int, input_text.split()))
    if not values or values[0] < 0 or values[0] > 20 or len(values) != values[0] + 1:
        raise InvalidInput("record count does not match")
    return ""
'''
    bad_builder = INPUT_BUILDER.replace('len(values)', 'len(values) + 1')
    responses = program_responses()
    class Client(PlanningClient):
        executable_input_validation = True
    client = Client([*responses[:2], bad_builder, validator, INPUT_BUILDER, reviews(), COUNTED_SUM_REFERENCE])
    suite = generator.generate_tests(client, spec(), method='program')
    assert [c.expected_output for c in suite.cases] == ['6', '9', '0']
    assert suite.validation['input_validation'] == 'model_generated_executable'
    assert 'record count does not match' in client.requests[4]['prompt']
    assert 'INPUT VALIDATOR ONLY' in client.requests[3]['system']
    assert len(client.requests) == 7


def test_invalid_input_cannot_escape_as_candidate_diagnostic_feedback():
    from tests.test_executable_test_builder import PlanningClient, program_responses, spec, INPUT_BUILDER
    validator = '''def solve_input(input_text: str) -> str:
    values = input_text.split()
    if len(values) != int(values[0]) + 1:
        raise InvalidInput("wrong count")
    return ""
'''
    class Client(PlanningClient):
        executable_input_validation = True
    bad_builder = INPUT_BUILDER.replace('len(values)', 'len(values) + 1')
    client = Client([*program_responses()[:2], bad_builder, validator])
    with pytest.raises(ValueError) as failure:
        generator.generate_tests(client, spec(), method='program', max_generation_attempts=1)
    assert failure.value.suite is None


def test_one_invalid_input_does_not_discard_valid_cases():
    from tests.test_executable_test_builder import PlanningClient, program_responses, spec, INPUT_BUILDER, reviews
    from tests.test_executable_test_oracle import COUNTED_SUM_REFERENCE
    validator = '''def solve_input(input_text: str) -> str:
    values = list(map(int, input_text.split()))
    if not values or len(values) != values[0] + 1:
        raise InvalidInput("record count mismatch")
    return ""
'''
    class Client(PlanningClient):
        executable_input_validation = True
    builder = INPUT_BUILDER.replace('len(values)', 'len(values) + (1 if len(values) == 3 else 0)')
    client = Client([*program_responses()[:2], builder, validator, reviews(('9', '0')), COUNTED_SUM_REFERENCE])
    # One attempt: a final attempt keeps a suite below the minimum-size retry rule.
    suite = generator.generate_tests(client, spec(), method='program', max_generation_attempts=1)
    assert [c.expected_output for c in suite.cases] == ['9', '0']
    assert suite.validation['discarded_inputs'][0]['reason'] == 'record count mismatch'
    assert suite.validation['status'] == 'model_agreement'


def test_bad_review_cardinality_uses_executable_adjudication_without_regenerating_inputs():
    from tests.test_executable_test_builder import PlanningClient, program_responses, spec, INPUT_BUILDER, reviews
    from tests.test_executable_test_oracle import COUNTED_SUM_REFERENCE
    validator = '''def solve_input(input_text: str) -> str:
    values = list(map(int, input_text.split()))
    if not values or len(values) != values[0] + 1:
        raise InvalidInput("record count mismatch")
    return ""
'''
    class Client(PlanningClient):
        executable_input_validation = True
    plans = program_responses()[:2]
    client = Client([*plans, INPUT_BUILDER, validator, reviews()[:1], COUNTED_SUM_REFERENCE,
                     *plans, COUNTED_SUM_REFERENCE])
    suite = generator.generate_tests(client, spec(), method='program')
    assert [c.expected_output for c in suite.cases] == ['6', '9', '0']
    assert 'cover every' in suite.validation['review_transport_error']
    assert suite.validation['status'] == 'model_agreement'
    assert len(suite.validation['answer_disputes']) == 3
    assert len(client.requests) == 9
