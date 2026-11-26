"""Execute model-supplied references for varied inputs; no live model calls."""

import json
import re

import pytest

from src.agents.llm_client import CallCountingClient, RecordingClient
from src.agents.task_spec import TaskSpec
from src.agents.test_generator import generate_tests, TEST_SUITE_RESPONSE_SCHEMA
from src.agents.test_oracle import ExecutableTestOracle, OracleBudgetExhausted, TEST_ORACLE_RESPONSE_SCHEMA
from src.agents.source_edits import line_edit_schema
from src.agents.test_review import test_review_schema as review_schema
from src.history import ExperimentStore
from tests.test_generated_test_review import SequenceClient


COUNTED_SUM_REFERENCE = '''def solve_input(input_text: str) -> str:
    tokens = input_text.split()
    try:
        values = [int(token) for token in tokens]
        if not values:
            raise ValueError("missing count")
        count = values[0]
        if not 0 <= count <= 20:
            raise ValueError("count out of bounds")
        if len(values) != count + 1:
            raise ValueError("record count does not match the actual input")
    except ValueError as error:
        raise InvalidInput(str(error))
    return str(sum(values[1:]))
'''


def spec():
    return TaskSpec('Read a count k (0 <= k <= 20) followed by exactly k integers. '
                    'Print their sum. Any whitespace may separate the tokens.', 'math', '')


def case(stdin='3 1 2 3', answer='6'):
    return dict(input=stdin, expected_output=answer, category='normal')


def unsure(number=1):
    return dict(test_number=number, input_valid=True, output_verified=False,
                expected_output='', category_valid=True,
                reason='The output cannot be verified without running a program.')


def reference(code=COUNTED_SUM_REFERENCE):
    return dict(approach='Validate a counted sequence and sum its elements.', code=code)


def test_reference_schema_accepts_multiline_unicode_python_and_rejects_another_language():
    pattern = TEST_ORACLE_RESPONSE_SCHEMA['properties']['code']['pattern']
    assert re.fullmatch(pattern, "def solve_input(input_text: str) -> str:\n    return input_text + 'α'")
    assert re.fullmatch(pattern, COUNTED_SUM_REFERENCE)
    assert not re.fullmatch(pattern, 'object Main { def main(args: Array[String]): Unit = () }')


def repair(source, replacement=COUNTED_SUM_REFERENCE):
    source = source.strip()
    return dict(diagnosis='Correct the reference against the original input/output contract.',
                source_sha256=line_edit_schema(source)['properties']['source_sha256']['const'],
                edits=[dict(start_line=1, end_line=len(source.splitlines()),
                            replacement=replacement.strip())])


def test_review_refusal_uses_a_reference_and_proceeds_after_execution():
    client = SequenceClient([[case()], [unsure()], reference()])
    stages = []
    suite = generate_tests(client, spec(), on_progress=lambda name, data: stages.append((name, data)))
    assert suite.cases[0].expected_output == '6'
    assert [request['schema'] for request in client.requests] == [
        TEST_SUITE_RESPONSE_SCHEMA, review_schema(1), TEST_ORACLE_RESPONSE_SCHEMA,
    ]
    oracle_prompt = client.requests[-1]['prompt']
    assert spec().problem_statement in oracle_prompt
    assert '3 1 2 3' not in oracle_prompt
    assert 'proposed=' not in oracle_prompt and 'Previous solution' not in oracle_prompt
    assert any(name == 'test_oracle_execution' for name, _ in stages)
    result = next(data for name, data in stages if name == 'test_oracle_result')
    assert result['input_valid'] and result['status'] == 'pass'


def test_reference_checks_all_cases_including_ones_the_reviewer_claimed_to_verify():
    reviewed = unsure(2)
    reviewed.update(output_verified=True, expected_output='0')
    client = SequenceClient([
        [case(), case('2 1', '0')], [unsure(), reviewed], reference(),
    ])
    with pytest.raises(ValueError, match='test 2 input rejected by executable reference'):
        generate_tests(client, spec(), max_generation_attempts=1)


def test_invalid_counts_are_rejected_and_corrected_suite_reuses_frozen_reference():
    client = SequenceClient([
        [case('4 1 2 3')], [unsure()], reference(), [case()], [unsure()],
    ])
    suite = generate_tests(client, spec())
    assert suite.cases[0].input == '3 1 2 3'
    assert len(client.requests) == 5
    assert sum(request['schema'] == TEST_ORACLE_RESPONSE_SCHEMA for request in client.requests) == 1
    assert 'record count does not match' in client.requests[3]['prompt']


def test_output_disagreement_does_not_select_a_more_agreeable_reference():
    client = SequenceClient([
        [case(answer='999999')], [unsure()], reference(), [case()], [unsure()],
    ])
    suite = generate_tests(client, spec())
    assert suite.cases[0].expected_output == '6'
    assert len(client.requests) == 5
    assert '999999' not in client.requests[2]['prompt']
    assert 'disagrees with executable reference' in client.requests[3]['prompt']


def test_reference_syntax_failure_gets_a_bounded_compiler_diagnostic_repair():
    client = SequenceClient([[case()], [unsure()], reference('def broken('), reference()])
    suite = generate_tests(client, spec())
    assert suite.cases[0].expected_output == '6'
    assert len(client.requests) == 4
    assert 'SyntaxError' in client.requests[-1]['prompt']
    assert 'Previous reference program (line numbers are labels' in client.requests[-1]['prompt']
    assert '3 1 2 3' not in client.requests[-1]['prompt']


@pytest.mark.parametrize('bad_code', [
    'def solve_input(input_text: str) -> str:\n    input_text.split()\n    raise RuntimeError("reference implementation failed")',
    'def solve_input(input_text: str) -> str:\n    input_text.split()\n    print("not JSON")\n    return ""',
    'def solve_input(input_text: str) -> str:\n    input_text.split()\n    return 5',
    'def solve_input(input_text: str) -> str:\n    input_text.split()\n    return dict(input_valid="true", expected_output="6", reason="invalid flag")',
])
def test_reference_runtime_and_schema_failures_never_approve_the_suite(bad_code):
    client = SequenceClient([[case()], [unsure()], reference(bad_code),
                             repair(bad_code, bad_code + '\n# Changed but still invalid'), reference(bad_code)])
    with pytest.raises(ValueError, match='reference failed after 3 attempts'):
        generate_tests(client, spec(), max_generation_attempts=1)
    assert len(client.requests) == 5


def test_reference_timeout_is_an_error_with_a_bounded_retry(monkeypatch):
    from src.verifier.runners.base import ExecutionResult, Status
    import src.agents.test_oracle as oracle_module
    monkeypatch.setattr(oracle_module, 'run_code_many', lambda *args, **kwargs: [
        ExecutionResult(Status.TIMEOUT, '', 'Execution timed out', -1),
    ])
    client = SequenceClient([[case()], [unsure()], reference(),
                             repair(COUNTED_SUM_REFERENCE, COUNTED_SUM_REFERENCE + '\n# Retry'), reference()])
    with pytest.raises(ValueError, match='timeout'):
        generate_tests(client, spec(), max_generation_attempts=1)
    assert len(client.requests) == 5


def test_exhausted_reference_stops_suite_regeneration_without_spending_more_calls():
    code = 'def solve_input(input_text: str) -> str:\n    input_text.split()\n    raise RuntimeError("unusable reference")'
    broken = reference(code)
    client = SequenceClient([
        [case()], [unsure()], broken, repair(code, code + '\n# Retry'), broken,
        [case('2 4 5', '9')], [unsure()],
    ])
    with pytest.raises(OracleBudgetExhausted, match='Generated test validation stopped'):
        generate_tests(client, spec())
    assert len(client.requests) == 5
    assert sum(request['schema'] == TEST_ORACLE_RESPONSE_SCHEMA for request in client.requests) == 2
    assert sum(request['schema'] == line_edit_schema(code) for request in client.requests) == 1


def test_invalid_input_or_category_does_not_invoke_a_reference():
    for flag in ('input_valid', 'category_valid'):
        verdict = unsure()
        verdict[flag] = False
        verdict['input_rule_quote'] = spec().problem_statement
        client = SequenceClient([[case()], [verdict]])
        with pytest.raises(ValueError, match='rejected by blind review'):
            generate_tests(client, spec(), max_generation_attempts=1)
        assert len(client.requests) == 2


def test_reference_works_for_a_different_problem_and_preserves_unicode_and_newlines():
    task = TaskSpec('Read the entire input verbatim. Print it twice without adding characters.', 'string', '')
    code = 'def solve_input(input_text: str) -> str:\n    return input_text*2'
    client = SequenceClient([[case('α😀\n', 'α😀\nα😀\n')], [unsure()], reference(code)])
    suite = generate_tests(client, task)
    assert suite.cases[0].expected_output == 'α😀\nα😀\n'


def test_reference_can_verify_legitimately_empty_task_output():
    task = TaskSpec('Read an integer and produce no output.', 'math', '')
    code = 'def solve_input(input_text: str) -> str:\n    int(input_text)\n    return ""'
    client = SequenceClient([[case('7', '')], [unsure()], reference(code)])
    assert generate_tests(client, task).cases[0].expected_output == ''


def test_reference_generation_and_repair_are_audited_counted_and_seeded(tmp_path):
    model = SequenceClient([[case()], [unsure()], reference('bad syntax('), reference()])
    with ExperimentStore(tmp_path / 'history.db') as store:
        client = CallCountingClient(RecordingClient(model, store, 'oracle-owner', 'generator'), seed=42)
        generate_tests(client, spec())
        rows = store._connection.execute('SELECT * FROM model_requests ORDER BY call_id').fetchall()
    assert client.call_count == 4
    assert [row['stage'] for row in rows] == ['tests:1', 'test_validation:1', 'test_oracle:1', 'test_oracle:2']
    assert len({json.loads(row['request_json'])['seed'] for row in rows}) == 4


def test_oracle_execution_rejects_extra_tokens_and_out_of_bound_counts():
    from src.agents.test_generator import build_test_suite
    for stdin in ('2 1 2 3', '21 ' + '1 ' * 21):
        client = SequenceClient([reference()])
        oracle = ExecutableTestOracle(client, spec().problem_statement)
        with pytest.raises(ValueError, match='input rejected by executable reference'):
            oracle.check(build_test_suite([case(stdin)]))
        assert len(client.requests) == 1  # Invalid cases never trigger reference selection.


def test_wrong_return_container_gets_specific_feedback_without_exposing_input_or_answers():
    code = 'def solve_input(input_text: str) -> str:\n    input_text.split()\n    return []'
    client = SequenceClient([[case()], [unsure()], reference(code), repair(code)])
    suite = generate_tests(client, spec())
    assert suite.cases[0].expected_output == '6'
    prompt = client.requests[-1]['prompt']
    assert 'must return the complete task output as a string' in prompt
    assert client.requests[-1]['schema'] == line_edit_schema(code)
    assert '1: def solve_input(input_text: str) -> str:' in prompt
    assert '3:     return []' in prompt
    assert '3 1 2 3' not in prompt
    assert "proposed=" not in prompt


def test_syntax_is_checked_before_execution_and_the_model_repairs_it(monkeypatch):
    import src.agents.test_oracle as oracle_module
    execute = oracle_module.run_code_many
    sources = []

    def record(source, *args, **kwargs):
        sources.append(source)
        return execute(source, *args, **kwargs)

    monkeypatch.setattr(oracle_module, 'run_code_many', record)
    invalid = "def solve_input(input_text: str) -> str:\n    print('trace')if __name__ == '__main__':\n        pass"
    client = SequenceClient([[case()], [unsure()], reference(invalid), reference()])
    assert generate_tests(client, spec()).cases[0].expected_output == '6'
    assert len(sources) == 1
    assert 'SyntaxError' in client.requests[-1]['prompt']
    assert 'solve_input(_skillforge_sys.stdin.read())' in sources[0]


def test_rechecking_exhausted_reference_does_not_repeat_calls_or_nest_errors():
    from src.agents.test_generator import build_test_suite
    code = 'def solve_input(input_text: str) -> str:\n    input_text.split()\n    return []'
    client = SequenceClient([reference(code), repair(code, code + '\n# Retry'), reference(code)])
    oracle = ExecutableTestOracle(client, spec().problem_statement)
    suite = build_test_suite([case()])
    with pytest.raises(OracleBudgetExhausted) as first:
        oracle.check(suite)
    original_error = oracle.last_error
    with pytest.raises(OracleBudgetExhausted) as second:
        oracle.check(suite)
    assert oracle.calls == len(client.requests) == 3
    assert oracle.last_error == original_error
    assert str(second.value) == str(first.value)
    assert str(second.value).count('failed after 3 attempts') == 1


@pytest.mark.parametrize('code', [
    'def solve(stdin):\n    return {}',
    'async def solve_input(input_text: str) -> str:\n    return {}',
    'def solve_input(input_text: str, extra) -> str:\n    return {}',
    COUNTED_SUM_REFERENCE + '\nsolve_input("0")',
])
def test_incorrect_function_interface_is_repaired_without_executing_a_driver(code):
    client = SequenceClient([[case()], [unsure()], reference(code), reference()])
    assert generate_tests(client, spec()).cases[0].expected_output == '6'
    assert len(client.requests) == 4


@pytest.mark.parametrize('changes', [
    dict(source_sha256='wrong-source'),
    dict(edits=[dict(start_line=0, end_line=2, replacement='return ""')]),
    dict(edits=[dict(start_line=2, end_line=4, replacement='return ""')]),
    dict(edits=[dict(start_line=3, end_line=3, replacement='    return []')]),
    dict(edits=[dict(start_line=1, end_line=2, replacement=COUNTED_SUM_REFERENCE),
                dict(start_line=2, end_line=2, replacement='    return ""')]),
])
def test_invalid_runtime_repairs_keep_accepted_source_and_request_a_fresh_function(changes):
    from src.agents.test_generator import build_test_suite
    code = 'def solve_input(input_text: str) -> str:\n    input_text.split()\n    return []'
    response = repair(code)
    response.update(changes)
    client = SequenceClient([reference(code), response, reference()])
    rejected = []
    def progress(name, data):
        if name == 'test_oracle_rejected' and data['oracle_attempt'] == 2:
            rejected.append(oracle.code)
    oracle = ExecutableTestOracle(client, spec().problem_statement, on_progress=progress)
    assert oracle.check(build_test_suite([case()]))[0]['expected_output'] == '6'
    assert rejected == [code]
    assert oracle.calls == len(client.requests) == 3
    assert client.requests[-1]['schema'] == TEST_ORACLE_RESPONSE_SCHEMA


def test_line_range_repair_fixes_runtime_parsing_and_executes_all_inputs():
    from src.agents.test_generator import build_test_suite
    code = COUNTED_SUM_REFERENCE.replace('input_text.split()', 'input_text.split(1)').strip()
    response = repair(code)
    response['edits'] = [dict(start_line=2, end_line=2, replacement='    tokens = input_text.split()')]
    client = SequenceClient([reference(code), response])
    oracle = ExecutableTestOracle(client, spec().problem_statement)
    suite = build_test_suite([case(), case('0', '0'), case('2 -4 7', '3')])
    outputs = oracle.check(suite)
    assert [output['expected_output'] for output in outputs] == ['6', '0', '3']
    assert oracle.code == COUNTED_SUM_REFERENCE.strip()
    assert oracle.calls == len(client.requests) == 2
    assert client.requests[-1]['schema'] == line_edit_schema(code)
    assert response['source_sha256'] in client.requests[-1]['prompt']
    assert 'TypeError' in client.requests[-1]['prompt']


@pytest.mark.parametrize('body', [
    'return input()',
    'import sys\nreturn sys.stdin.read()',
    'import sys as stream_module\nreturn stream_module.stdin.readline()',
    'from sys import stdin as stream\nreturn stream.read()',
    'import sys\nreader = sys.stdin.read\nreturn reader()',
    'import sys\nfirst = sys.stdin\nsecond = first\nreturn second.read()',
    'from builtins import input as reader\nreturn reader()',
    'import builtins as b\nreader = b.input\nreturn reader()',
    'return open(0).read()',
    'import os\nreturn os.read(0, 100).decode()',
    'import os as system\nreturn system.fdopen(0).read()',
    'return open("/dev/stdin").read()',
    'return input_text.read()',
    'return input_text()',
    'return getattr(__import__("sys"), "stdin").read()',
    'return "ignored input"',
])
def test_invalid_input_contract_is_rejected_before_execution(body, monkeypatch):
    import textwrap
    import src.agents.test_oracle as oracle_module
    from src.agents.test_generator import build_test_suite
    def forbidden_execution(*args, **kwargs):
        pytest.fail('An invalid reference must not execute.')
    monkeypatch.setattr(oracle_module, 'run_code_many', forbidden_execution)
    code = 'def solve_input(input_text: str) -> str:\n' + textwrap.indent(body, '    ')
    oracle = ExecutableTestOracle(SequenceClient([reference(code)]), 'Interpret the supplied input.', max_attempts=1)
    with pytest.raises(OracleBudgetExhausted, match='input contract'):
        oracle.check(build_test_suite([case()]))
    assert oracle.code is None and not oracle.source_valid


def test_input_contract_violation_is_edited_without_running_the_invalid_reference(monkeypatch):
    import src.agents.test_oracle as oracle_module
    execute = oracle_module.run_code_many
    executed = []
    def record(code, *args, **kwargs):
        executed.append(code)
        return execute(code, *args, **kwargs)
    monkeypatch.setattr(oracle_module, 'run_code_many', record)
    bad = 'def solve_input(input_text: str) -> str:\n    import sys\n    data = sys.stdin.read()\n    return data'
    client = SequenceClient([[case()], [unsure()], reference(bad), repair(bad)])
    assert generate_tests(client, spec()).cases[0].expected_output == '6'
    assert len(executed) == 1
    assert client.requests[-1]['schema'] == line_edit_schema(bad)
    assert 'host has already consumed stdin' in client.requests[-1]['prompt']


def test_invalid_io_repair_leaves_source_unaccepted_and_uses_a_fresh_function(monkeypatch):
    import src.agents.test_oracle as oracle_module
    from src.agents.test_generator import build_test_suite
    execute = oracle_module.run_code_many
    executed = []
    def record(code, *args, **kwargs):
        executed.append(code)
        return execute(code, *args, **kwargs)
    monkeypatch.setattr(oracle_module, 'run_code_many', record)
    bad = 'def solve_input(input_text: str) -> str:\n    import sys\n    return sys.stdin.read()'
    invalid_edit = repair(bad, bad + '\n# Does not fix IO')
    client = SequenceClient([reference(bad), invalid_edit, reference()])
    accepted = []
    oracle = ExecutableTestOracle(client, spec().problem_statement,
        on_progress=lambda name, data: accepted.append(oracle.code) if name == 'test_oracle_rejected' else None)
    assert oracle.check(build_test_suite([case()]))[0]['expected_output'] == '6'
    assert accepted == [None, None]
    assert len(executed) == 1
    assert client.requests[-1]['schema'] == TEST_ORACLE_RESPONSE_SCHEMA


def test_syntax_invalid_edit_never_replaces_the_reference_and_uses_the_reserved_fresh_function(monkeypatch):
    import src.agents.test_oracle as oracle_module
    from src.agents.test_generator import build_test_suite
    execute = oracle_module.run_code_many
    executed = []
    def record(code, *args, **kwargs):
        executed.append(code)
        return execute(code, *args, **kwargs)
    monkeypatch.setattr(oracle_module, 'run_code_many', record)
    code = 'def solve_input(input_text: str) -> str:\n    input_text.split()\n    return []'
    response = repair(code)
    response['edits'] = [dict(start_line=3, end_line=3, replacement='if (')]
    client = SequenceClient([reference(code), response, reference()])
    preserved = []
    def progress(name, data):
        if name == 'test_oracle_rejected' and data['oracle_attempt'] == 2:
            preserved.append(oracle.code)
    oracle = ExecutableTestOracle(client, spec().problem_statement, on_progress=progress)
    assert oracle.check(build_test_suite([case()]))[0]['expected_output'] == '6'
    assert preserved == [code]
    assert len(executed) == 2 and all('if (' not in source for source in executed)
    assert len(client.requests) == 3
    assert client.requests[-1]['schema'] == TEST_ORACLE_RESPONSE_SCHEMA


def test_reference_for_a_no_input_task_can_validate_empty_input_and_return_a_constant():
    from src.agents.test_generator import build_test_suite
    code = 'def solve_input(input_text: str) -> str:\n    if input_text.strip():\n        raise InvalidInput("No input is allowed")\n    return "READY"'
    client = SequenceClient([reference(code)])
    oracle = ExecutableTestOracle(client, 'Read no input and print READY.')
    suite = build_test_suite([case('', 'READY')], allow_empty_input=True)
    assert oracle.check(suite)[0]['expected_output'] == 'READY'
    with pytest.raises(ValueError, match='input rejected'):
        oracle.check(build_test_suite([case('unexpected', 'READY')]))
    assert len(client.requests) == 1
