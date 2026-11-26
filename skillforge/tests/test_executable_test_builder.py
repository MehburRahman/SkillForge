"""Execute suite builders without relaxing structural or blind review checks."""

import json

import pytest

from src.agents.llm_client import CallCountingClient, RecordingClient
from src.agents.task_spec import TaskSpec
from src.agents.test_builder import TEST_BUILDER_RESPONSE_SCHEMA
from src.agents.test_generator import generate_tests, TEST_SUITE_RESPONSE_SCHEMA
from src.agents.test_review import test_review_schema as review_schema
from src.history import ExperimentStore
from src.verifier.runners.base import ExecutionResult, Status
from tests.test_generated_test_review import SequenceClient
from tests.test_executable_test_oracle import COUNTED_SUM_REFERENCE, reference


SUM_BUILDER = '''import json
cases = []
for values in ([1, 2, 3], [4, 5], []):
    text = " ".join(map(str, [len(values), *values]))
    cases.append(dict(input=text, expected_output=str(sum(values)), category="normal"))
print(json.dumps(cases))
'''


def spec():
    return TaskSpec('Read a count k (0 <= k <= 20) followed by exactly k integers. '
                    'Print their sum.', 'math', '')


def duplicate_suite():
    return [dict(input='3 1 2 3', expected_output='6', category='normal')] * 2


def builder(code=SUM_BUILDER):
    return dict(approach='Build counted sequences and calculate their sums.', code=code)


INPUT_BUILDER = '''def build_inputs():
    cases = []
    for values in ([1, 2, 3], [4, 5], []):
        text = " ".join(map(str, [len(values), *values]))
        cases.append(dict(input=text, category="normal"))
    return cases
'''


class PlanningClient(SequenceClient):
    def generate(self, prompt, **kwargs):
        self.requests.append(dict(prompt=prompt, **kwargs))
        return next(self.responses)


def program_responses(*, code=COUNTED_SUM_REFERENCE):
    return ['Validate the count and sum the values.',
            'Check count bounds and exact length, then sum the values.', INPUT_BUILDER, reviews(), code]


def reviews(answers=('6', '9', '0')):
    return [dict(test_number=i, input_valid=True, output_verified=True,
                 expected_output=answer, category_valid=True,
                 reason='Validated the count and summed the integers.')
            for i, answer in enumerate(answers, 1)]


def test_repeated_structural_failure_materializes_then_blindly_reviews_all_cases():
    client = SequenceClient([duplicate_suite(), duplicate_suite(), builder(), reviews()])
    stages = []
    suite = generate_tests(client, spec(), on_progress=lambda name, data: stages.append((name, data)))
    assert [case.input for case in suite.cases] == ['3 1 2 3', '2 4 5', '0']
    assert [request['schema'] for request in client.requests] == [
        TEST_SUITE_RESPONSE_SCHEMA, TEST_SUITE_RESPONSE_SCHEMA,
        TEST_BUILDER_RESPONSE_SCHEMA, review_schema(3),
    ]
    result = next(data for name, data in stages if name == 'test_builder_result')
    assert result['status'] == 'pass' and result['duration_seconds'] > 0
    assert 'duplicates an earlier input' in client.requests[2]['prompt']
    assert 'Previous solution' not in client.requests[2]['prompt']
    assert all(spec().problem_statement in request['prompt'] for request in client.requests)


def test_program_mode_starts_with_a_builder_and_keeps_blind_output_review():
    client = PlanningClient(program_responses())
    stages = []
    suite = generate_tests(client, spec(), method='program', on_progress=lambda name, data: stages.append((name, data)))
    assert [case.expected_output for case in suite.cases] == ['6', '9', '0']
    assert all('schema' not in request for request in (*client.requests[:3], client.requests[-1]))
    assert client.requests[3]['schema'] == review_schema(3)
    for request in (*client.requests[:3], client.requests[-1]):
        assert '3 1 2 3' not in request['prompt']
        assert 'expected_output' not in request['prompt']
    assert 'No solution or expected answers are needed' in client.requests[2]['prompt']
    assert ('test_generation', dict(test_generation_attempt=1, method='program')) in stages


def test_chosen_io_formats_reach_builder_blind_review_and_reference_requests():
    task = spec()
    task.input_format = 'Whitespace-separated count followed by exactly that many values'
    task.output_format = 'One decimal sum on stdout'
    client = PlanningClient(program_responses())
    suite = generate_tests(client, task, method='program')
    assert [case.expected_output for case in suite.cases] == ['6', '9', '0']
    for request in client.requests:
        assert task.problem_statement in request['prompt']
        assert task.input_format in request['prompt']
        assert task.output_format in request['prompt']


def test_program_mode_rejects_builder_answers_before_reference_generation():
    wrong = INPUT_BUILDER.replace('category="normal"', 'category="normal", expected_output="999999"')
    client = PlanningClient([*program_responses()[:2], wrong])
    with pytest.raises(ValueError, match='answers come from the separate reference'):
        generate_tests(client, spec(), method='program', max_generation_attempts=1)
    assert len(client.requests) == 3


def test_program_mode_rejects_executable_answers_that_disagree_with_blind_review():
    wrong_reference = COUNTED_SUM_REFERENCE.replace('str(sum(values[1:]))', '"999999"')
    client = PlanningClient([*program_responses(code=wrong_reference),
                             *program_responses()[:2], COUNTED_SUM_REFERENCE])
    with pytest.raises(ValueError, match='Executable references disagree'):
        generate_tests(client, spec(), method='program', max_generation_attempts=1)
    assert '999999' not in client.requests[3]['prompt']


def test_program_mode_rejects_invalid_inputs_and_reuses_the_frozen_reference():
    invalid_builder = INPUT_BUILDER.replace('len(values)', 'len(values) + 1')
    responses = program_responses()
    client = PlanningClient([*responses[:2], invalid_builder, reviews(), responses[4], INPUT_BUILDER, reviews()])
    suite = generate_tests(client, spec(), method='program')
    assert [case.expected_output for case in suite.cases] == ['6', '9', '0']
    assert len(client.requests) == 7
    assert 'record count' in client.requests[5]['prompt']


def test_program_mode_works_for_verbatim_text_and_unicode():
    task = TaskSpec('Read input verbatim and print it twice.', 'string', '')
    code = 'def build_inputs():\n    return [dict(input="α😀\\n", category="normal")]'
    oracle = 'def solve_input(input_text: str) -> str:\n    return input_text * 2'
    client = PlanningClient(['Preserve input verbatim.', 'Return twice the exact string.', code,
                             reviews(('α😀\nα😀\n',)), oracle])
    suite = generate_tests(client, task, method='program', max_generation_attempts=1)
    assert suite.cases[0].expected_output == 'α😀\nα😀\n'


def test_program_mode_bounds_materialized_inputs_before_reference_generation():
    oversized = 'def build_inputs():\n    return [dict(input="1 " * 20000, category="stress")]'
    client = PlanningClient([*program_responses()[:2], oversized])
    with pytest.raises(ValueError, match='construction budget'):
        generate_tests(client, spec(), method='program', max_generation_attempts=1)
    assert len(client.requests) == 3


def test_reference_plan_does_not_forward_placeholder_source_to_generation():
    responses = program_responses()
    outline = '\n```python\ndef unfinished():\n    pass\n```'
    responses[0] += outline
    responses[1] += outline
    client = PlanningClient(responses)
    assert len(generate_tests(client, spec(), method='program').cases) == 3
    assert all('def unfinished' not in request['prompt'] for request in client.requests)


def test_program_mode_derives_uncertain_answers_but_marks_them_for_review_without_anchors():
    uncertain = reviews()
    for review in uncertain:
        review.update(output_verified=False, expected_output='', reason='Need candidate execution.')
    client = PlanningClient([*program_responses()[:3], uncertain, COUNTED_SUM_REFERENCE])
    suite = generate_tests(client, spec(), method='program', max_generation_attempts=1)
    assert [case.expected_output for case in suite.cases] == ['6', '9', '0']
    assert suite.validation['status'] == 'needs_review'
    assert not suite.validation['reference_anchors_verified']
    assert len(client.requests) == 5


def test_program_mode_still_rejects_malformed_input_before_reference_generation():
    invalid = reviews()
    invalid[0].update(input_valid=False, reason='Declared count does not match the input.',
                      input_rule_quote=spec().problem_statement)
    client = PlanningClient([*program_responses()[:3], invalid])
    with pytest.raises(ValueError, match='input rejected by blind review'):
        generate_tests(client, spec(), method='program', max_generation_attempts=1)
    assert len(client.requests) == 4


def test_builder_cannot_approve_its_own_incorrect_answers():
    wrong = SUM_BUILDER.replace('str(sum(values))', '"999999"')
    client = SequenceClient([duplicate_suite(), duplicate_suite(), builder(wrong), reviews()])
    with pytest.raises(ValueError, match='disagrees with blind derivation'):
        generate_tests(client, spec())
    assert len(client.requests) == 4
    assert '999999' not in client.requests[-1]['prompt']


def test_repeated_answer_disagreement_also_switches_to_executable_generation():
    wrong_case = dict(input='3 1 2 3', expected_output='999999', category='normal')
    client = SequenceClient([[wrong_case], reviews(('6',)), [wrong_case], reviews(('6',)),
                             builder(), reviews()])
    suite = generate_tests(client, spec())
    assert [case.expected_output for case in suite.cases] == ['6', '9', '0']
    assert len(client.requests) == 6
    assert client.requests[4]['schema'] == TEST_BUILDER_RESPONSE_SCHEMA


def test_builder_output_still_rejects_duplicate_inputs_before_review():
    code = 'import json; print(json.dumps(' + repr(duplicate_suite()) + '))'
    client = SequenceClient([duplicate_suite(), duplicate_suite(), builder(code)])
    with pytest.raises(ValueError, match='duplicates an earlier input'):
        generate_tests(client, spec())
    assert len(client.requests) == 3


@pytest.mark.parametrize('code', [
    'def broken(', 'raise RuntimeError("builder failed")',
    'print("not JSON")', 'print("[]")',
    'print("x" * 300000)',
])
def test_builder_execution_or_invalid_output_never_freezes_tests(code):
    client = SequenceClient([duplicate_suite(), duplicate_suite(), builder(code)])
    with pytest.raises(ValueError, match='after 3 attempts'):
        generate_tests(client, spec())
    assert len(client.requests) == 3


def test_builder_timeout_remains_a_failure_within_the_existing_attempt_budget(monkeypatch):
    import src.agents.test_builder as module
    monkeypatch.setattr(module, 'execute', lambda *args, **kwargs:
                        ExecutionResult(Status.TIMEOUT, '', 'Execution timed out', -1))
    client = SequenceClient([duplicate_suite(), duplicate_suite(), builder()])
    with pytest.raises(ValueError, match='test builder execution failed: timeout'):
        generate_tests(client, spec())
    assert len(client.requests) == 3


def test_builder_generation_is_audited_counted_and_seeded(tmp_path):
    model = SequenceClient([duplicate_suite(), duplicate_suite(), builder(), reviews()])
    with ExperimentStore(tmp_path / 'history.db') as store:
        client = CallCountingClient(RecordingClient(model, store, 'builder-owner', 'generator'), seed=42)
        generate_tests(client, spec())
        rows = store._connection.execute('SELECT * FROM model_requests ORDER BY call_id').fetchall()
    assert client.call_count == 4
    assert [row['stage'] for row in rows] == ['tests:1', 'tests:2', 'tests:3', 'test_validation:3']
    assert len({json.loads(row['request_json'])['seed'] for row in rows}) == 4
    assert 'Build counted sequences' in rows[2]['response_text']


def test_native_structured_inputs_do_not_require_a_python_builder():
    from src.agents.test_builder import build_generated_tests
    class Client:
        structured_test_inputs = True
        def generate(self, *args, **kwargs):
            pytest.fail('Native input generation should request structured data')
        def generate_json(self, prompt, **kwargs):
            assert 'expected_output' not in kwargs['schema']['items']['properties']
            assert kwargs['schema']['maxItems'] == 6
            return json.dumps([dict(input='2 4 7', category='normal')])
    result = json.loads(build_generated_tests(Client(), 'Read a counted sequence.', None, '', input_only=True))
    assert result == [dict(input='2 4 7', category='normal')]


@pytest.mark.parametrize('payload', [
    {'type': 'array', 'items': [{'input': '1 8', 'category': 'normal'}]},
    {'input': '1 8', 'category': 'normal'},
])
def test_native_input_envelopes_preserve_actual_case_data(payload):
    from src.agents.test_builder import build_generated_tests
    class Client:
        structured_test_inputs = True
        def generate_json(self, *args, **kwargs): return json.dumps(payload)
    assert json.loads(build_generated_tests(Client(), 'Read counted records.', None, '', input_only=True)) == [
        {'input': '1 8', 'category': 'normal'}]


def test_program_mode_retries_a_suite_with_too_few_inputs():
    single = 'def build_inputs():\n    return [dict(input="2 4 5\\n", category="normal")]'
    client = PlanningClient([*program_responses()[:2], single, *program_responses()[2:]])
    suite = generate_tests(client, spec(), method='program')
    assert len(suite.cases) == 3
    assert any('at least 3 distinct valid inputs' in request['prompt'] for request in client.requests)


def test_program_mode_keeps_a_repeated_small_suite_instead_of_stopping():
    single = 'def build_inputs():\n    return [dict(input="2 4 5\\n", category="normal")]'
    oracle = 'def solve_input(input_text: str) -> str:\n    values = list(map(int, input_text.split()))\n    return str(sum(values[1:]))'
    client = PlanningClient([*program_responses()[:2], single, single, reviews(('9',)), oracle])
    suite = generate_tests(client, spec(), method='program')
    assert [case.input for case in suite.cases] == ['2 4 5\n']
