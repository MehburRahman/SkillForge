import json

import pytest

from src.agents.source_edits import (
    SOURCE_EDIT_RESPONSE_SCHEMA, apply_source_edits, apply_line_edits, line_edit_schema,
)
from src.agents.test_generator import TestCase as Case, TestSuite as Suite
from src.history import ExperimentStore
from src.reinforcement_loop import run_candidate_refinement_loop
from src.verifier.outcome_verifier import VerificationResult, TestCaseVerification as CaseResult
from src.verifier.runners.base import ExecutionResult, Status
from tests.test_experiment_configs import RecordingClient


def edits(*values):
    return dict(diagnosis='Replace the undefined expression.', edits=list(values))


def test_edits_use_original_offsets_even_when_replacements_shift_or_create_anchors():
    assert apply_source_edits('abc xyz', edits(
        dict(old='abc', new='xyz extended'), dict(old='xyz', new='end'),
    )) == 'xyz extended end'


@pytest.mark.parametrize('response', [
    {}, edits(), edits(dict(old='', new='new')),
    edits(dict(old='abc', new='abc')), edits(dict(old='absent', new='new')),
    edits(dict(old='a', new='new')),  # ambiguous anchor
    edits(dict(old='abc', new='')),  # removes the entire program
    edits(dict(old='abc', new='x'), dict(old='bc', new='y')),
    edits(dict(old='abc', new=None)),
    dict(diagnosis='', edits=[dict(old='abc', new='x')]),
])
def test_invalid_edits_never_partially_change_source(response):
    source = 'abc' if response != edits(dict(old='a', new='new')) else 'abc abc'
    with pytest.raises(ValueError):
        apply_source_edits(source, response)


def line_edits(source, *values):
    return dict(diagnosis='Correct the failing expression.',
                source_sha256=line_edit_schema(source)['properties']['source_sha256']['const'],
                edits=list(values))


def test_line_edits_use_original_ranges_and_keep_following_lines_separate():
    source = 'first\nsecond\nthird\nlast'
    response = line_edits(source,
        dict(start_line=4, end_line=4, replacement='final'),
        dict(start_line=1, end_line=2, replacement='new first\nnew second\nextra'),
    )
    assert apply_line_edits(source, response) == 'new first\nnew second\nextra\nthird\nfinal'


def test_line_patch_keeps_useful_edits_alongside_unchanged_ranges():
    source = 'unchanged\nbroken\n'
    response = line_edits(source,
        dict(start_line=1, end_line=1, replacement='unchanged'),
        dict(start_line=2, end_line=2, replacement='fixed\n'))
    assert apply_line_edits(source, response) == 'unchanged\nfixed\n'
    with pytest.raises(ValueError, match='changed, non-empty'):
        apply_line_edits(source, line_edits(source,
            dict(start_line=1, end_line=1, replacement='unchanged')))


def test_invalid_native_patch_switches_to_full_source_before_repeat_stop(monkeypatch):
    class Native:
        model = 'fixture'
        supports_source_edit_repairs = True
        source_edit_style = 'lines'
        def __init__(self):
            self.requests = []
        def generate(self, prompt, system='', **kwargs):
            self.requests.append(prompt)
            if 'problem analyzer' in system:
                return json.dumps(dict(selected_max_attempts=3, attempt_budget_reason='Three candidates.'))
            if 'Previous solution:' in prompt:
                assert 'Generate the complete source afresh' in prompt
                return 'fixed'
            return 'broken'
        def generate_json(self, prompt, schema, **kwargs):
            if 'source_sha256' not in schema.get('properties', {}):
                return self.generate(prompt, **kwargs)
            self.requests.append(prompt)
            return json.dumps(dict(diagnosis='Bad transport patch.', source_sha256='wrong',
                edits=[dict(start_line=1, end_line=1, replacement='fixed')]))
    def verifier(code, *args, **kwargs):
        if code == 'fixed':
            return VerificationResult(1, 1, 1, case_results=[CaseResult('', 'ok', 'ok', 'normal', True,
                ExecutionResult(Status.PASS, 'ok', '', 0))])
        failure = dict(status='compile_error', stderr='Not found: broken', input='', expected='ok', actual='')
        return VerificationResult(0, 1, 0, failure, [failure], [CaseResult('', 'ok', '', 'normal', False,
            ExecutionResult(Status.COMPILE_ERROR, '', 'Not found: broken', 1))])
    monkeypatch.setattr('src.reinforcement_loop.preflight_execution', lambda *args: None)
    monkeypatch.setattr('src.reinforcement_loop.verify', verifier)
    client = Native()
    result = run_candidate_refinement_loop(client, 'Print ok.', language='scala', max_attempts=3,
        trusted_test_suite=Suite([Case('', 'ok', 'normal')]))
    assert result.success and result.best_code == 'fixed'
    assert len(result.attempts) == 2 and len(client.requests) == 4


def test_line_edits_support_adjacent_ranges_and_deleting_whole_lines():
    source = 'first\nsecond\nthird\n'
    response = line_edits(source,
        dict(start_line=1, end_line=1, replacement=''),
        dict(start_line=2, end_line=2, replacement='changed\n'),
    )
    assert apply_line_edits(source, response) == 'changed\nthird\n'


def test_line_edits_preserve_meaningful_indentation_changes():
    source = 'def last(values):\n    for value in values:\n        pass\n        return value'
    response = line_edits(source,
        dict(start_line=4, end_line=4, replacement='    return value'),
    )
    namespace = {}
    exec(apply_line_edits(source, response), namespace)
    assert namespace['last']([2, 4, 9]) == 9


@pytest.mark.parametrize('replacement', ['return sum(values)', '    return sum(values)'])
def test_relative_line_edits_keep_the_original_python_block_indentation(replacement):
    source = 'def total(values):\n    return values[0]'
    response = line_edits(source, dict(start_line=2, end_line=2, replacement=replacement))
    namespace = {}
    fixed = apply_line_edits(source, response, relative_indentation=True)
    exec(fixed, namespace)
    assert namespace['total']([2, 5, 9]) == 16


def test_relative_line_edits_preserve_nested_blocks():
    source = 'def nonnegative(value):\n    return value'
    response = line_edits(source, dict(start_line=2, end_line=2,
        replacement='if value < 0:\n    return 0\nreturn value'))
    namespace = {}
    exec(apply_line_edits(source, response, relative_indentation=True), namespace)
    assert namespace['nonnegative'](-7) == 0
    assert namespace['nonnegative'](8) == 8


def test_line_schema_and_repair_are_bound_to_exact_source_including_newlines():
    source = 'first\nsecond\n'
    schema = line_edit_schema(source)
    assert schema['properties']['edits']['items']['properties']['end_line']['maximum'] == 2
    response = line_edits(source, dict(start_line=2, end_line=2, replacement='changed'))
    with pytest.raises(ValueError, match='different source'):
        apply_line_edits(source.rstrip(), response)


@pytest.mark.parametrize('changes', [
    dict(source_sha256='stale-hash'), dict(diagnosis=''), dict(edits=[]),
    dict(edits=[dict(start_line=0, end_line=1, replacement='changed')]),
    dict(edits=[dict(start_line=1, end_line=3, replacement='changed')]),
    dict(edits=[dict(start_line=2, end_line=1, replacement='changed')]),
    dict(edits=[dict(start_line=True, end_line=1, replacement='changed')]),
    dict(edits=[dict(start_line=1, end_line=1.0, replacement='changed')]),
    dict(edits=[dict(start_line=1, end_line=1, replacement=None)]),
    dict(edits=[dict(start_line=1, end_line=1, replacement='first')]),
    dict(edits=[dict(start_line=1, end_line=2, replacement='')]),
    dict(edits=[dict(start_line=1, end_line=2, replacement='changed'),
                dict(start_line=2, end_line=2, replacement='overlapping')]),
    dict(edits=[dict(old='first', new='changed')]),
])
def test_invalid_line_edits_are_rejected_before_applying_any_ranges(changes):
    source = 'first\nsecond'
    response = line_edits(source, dict(start_line=2, end_line=2, replacement='changed'))
    response.update(changes)
    with pytest.raises(ValueError):
        apply_line_edits(source, response)


@pytest.mark.parametrize('response', [{}, None, [], dict(diagnosis='failure', edits=[])])
def test_line_edits_reject_responses_missing_required_fields(response):
    with pytest.raises(ValueError, match='must contain'):
        apply_line_edits('first\nsecond', response)


@pytest.mark.parametrize(('language', 'broken', 'fixed'), [
    ('python', 'print(undefined_value)', 'print(7)'),
    ('javascript', 'console.log(undefinedValue)', 'console.log(7)'),
    ('scala', 'object Main { val value = undefinedValue }', 'object Main { val value = 7 }'),
])
@pytest.mark.parametrize('duplicate', [True, False])
def test_repeated_compiler_failure_switches_to_edits_and_recovers_without_extra_calls(
    tmp_path, monkeypatch, language, broken, fixed, duplicate,
):
    diagnostic = 'Candidate:1: undefined expression'
    compile_calls = []

    def verify(code, *args, **kwargs):
        compile_calls.append(code)
        if code == fixed:
            return VerificationResult(1, 1, 1, case_results=[
                CaseResult('', '7', '7', 'normal', True,
                           ExecutionResult(Status.PASS, '7', '', 0))
            ])
        failure = dict(status='compile_error', stderr=diagnostic, input='', expected='7', actual='')
        return VerificationResult(0, 1, 0, failure, [failure], [
            CaseResult('', '7', '', 'normal', False,
                       ExecutionResult(Status.COMPILE_ERROR, '', diagnostic, 1))
        ])

    monkeypatch.setattr('src.reinforcement_loop.preflight_execution', lambda *args: None)
    monkeypatch.setattr('src.reinforcement_loop.verify', verify)
    second = broken if duplicate else broken.replace('undefined', 'another')
    model = RecordingClient([broken, second, json.dumps(edits(dict(old=second, new=fixed)))])
    with ExperimentStore(tmp_path / 'history.db') as store:
        result = run_candidate_refinement_loop(model, 'Print seven.', language=language,
            max_attempts=3, history_store=store, trusted_test_suite=Suite([Case('', '7', 'normal')]))
        rows = store._connection.execute('SELECT * FROM model_requests ORDER BY call_id').fetchall()
    assert result.success and result.best_code == fixed
    assert compile_calls == ([broken, fixed] if duplicate else [broken, second, fixed])
    assert len(result.attempts) == (2 if duplicate else 3)
    assert result.model_calls == 4
    assert 'Source edit response contract' in model.requests[-1]['prompt']
    assert diagnostic in model.requests[-1]['prompt']
    assert json.loads(rows[-1]['request_json'])['method'] == 'generate'


def test_structured_client_receives_edit_schema_and_preserves_budget(monkeypatch):
    from types import SimpleNamespace
    from src.agents.code_generator import CodeCandidate
    from src.agents.refinement_agent import refine_code
    from src.agents.task_spec import TaskSpec

    class Client:
        def generate_json(self, prompt, **kwargs):
            assert kwargs['schema'] == SOURCE_EDIT_RESPONSE_SCHEMA
            assert 'repeats attempt 1' in prompt
            return json.dumps(edits(dict(old='bad', new='good')))

    result = refine_code(Client(), TaskSpec('Print a value.', 'math', ''),
        CodeCandidate('bad', 'math'),
        SimpleNamespace(failures=[dict(status='compile_error', stderr='undefined expression')],
                        first_failure=None), repeated_from=1)
    assert result.raw_code == 'good'
    assert result.repair_validation_error is None


@pytest.mark.parametrize('planned', [False, True])
def test_repeated_runtime_failure_uses_validated_edits_with_or_without_a_plan(planned):
    from types import SimpleNamespace
    from src.agents.code_generator import CodeCandidate
    from src.agents.refinement_agent import refine_code
    from src.agents.task_spec import TaskSpec
    from src.agents.algorithm_planner import AlgorithmPlan
    class Client:
        def generate_json(self, prompt, **kwargs):
            assert kwargs['schema'] == SOURCE_EDIT_RESPONSE_SCHEMA
            assert 'runtime or output' in kwargs['system']
            assert 'repeats attempt 1' in prompt
            return json.dumps(edits(dict(old='bad', new='good')))
    result = refine_code(Client(), TaskSpec('Read input and output it unchanged.', 'other', ''),
        CodeCandidate('bad', 'other'),
        SimpleNamespace(passed=0, total=1, pass_rate=0,
                        failures=[dict(status='error', stderr='Runtime failure')], first_failure=None),
        repeated_from=1, algorithm_plan=AlgorithmPlan('Preserve all records.') if planned else None)
    assert result.raw_code == 'good' and result.repair_validation_error is None


@pytest.mark.parametrize('native', [False, True])
def test_diagnosed_reader_failure_repairs_source_without_spending_an_algorithm_replan(monkeypatch, native):
    broken = ('object Main { def main(args: Array[String]): Unit = {\n'
              'val input = scala.io.StdIn.readLine()\nval records = input.split("\\n")\n'
              'println(records(1))\n}}')
    fixed = broken.replace('scala.io.StdIn.readLine()', 'scala.io.Source.stdin.mkString')
    diagnostic = 'java.lang.ArrayIndexOutOfBoundsException: Index 1 out of bounds for length 1'
    class Client:
        model = 'fixture-model'
        supports_source_edit_repairs = True
        source_edit_style = 'lines' if native else None
        def __init__(self):
            self.requests = []
        def generate(self, prompt, system='', **kwargs):
            self.requests.append(dict(prompt=prompt, system=system))
            if 'problem analyzer' in system:
                return json.dumps(dict(selected_max_attempts=2, attempt_budget_reason='Allow a repair.'))
            if 'You design an algorithm' in system or 'You review and correct a proposed algorithm' in system:
                return 'Read the two supplied records and output the second.'
            assert 'You diagnose a proven failure' not in system
            if native and 'Previous source:' in prompt:
                assert 'consumed only the first input line' in prompt
                assert 'Numbered ORIGINAL source for line edits' not in prompt
                return fixed
            return broken
        def generate_json(self, prompt, **kwargs):
            if kwargs['schema'] != SOURCE_EDIT_RESPONSE_SCHEMA:
                return self.generate(prompt, **kwargs)
            self.requests.append(dict(prompt=prompt, **kwargs))
            assert 'consumed only the first input line' in prompt
            return json.dumps(edits(dict(old='scala.io.StdIn.readLine()',
                                         new='scala.io.Source.stdin.mkString')))
    def verify(code, *args, **kwargs):
        if code == fixed:
            return VerificationResult(1, 1, 1, case_results=[
                CaseResult('first\nsecond', 'second', 'second', 'normal', True,
                           ExecutionResult(Status.PASS, 'second', '', 0))])
        failure = dict(status='error', stderr=diagnostic, input='first\nsecond', expected='second', actual='')
        return VerificationResult(0, 1, 0, failure, [failure], [
            CaseResult('first\nsecond', 'second', '', 'normal', False,
                       ExecutionResult(Status.ERROR, '', diagnostic, 1))])
    monkeypatch.setattr('src.reinforcement_loop.preflight_execution', lambda *args: None)
    monkeypatch.setattr('src.reinforcement_loop.verify', verify)
    client = Client()
    result = run_candidate_refinement_loop(client, 'Read two lines and print the second line.',
        language='scala', reasoning_mode='plan_critique', max_attempts=2,
        trusted_test_suite=Suite([Case('first\nsecond', 'second', 'normal')]))
    assert result.success and len(result.attempts) == 2
    assert result.model_calls == 5  # Analysis, plan, review, candidate, reader edit.


@pytest.mark.parametrize('status', ['compile_error', 'error'])
@pytest.mark.parametrize('stale', [False, True])
def test_native_line_repairs_restore_indentation_and_reject_stale_source(status, stale):
    from types import SimpleNamespace
    from src.agents.code_generator import CodeCandidate
    from src.agents.refinement_agent import refine_code
    from src.agents.task_spec import TaskSpec
    source = 'object Main {\n  def main(args: Array[String]): Unit = {\n    val text = bad\n    println(text)\n  }\n}'
    replacement = 'val text = {\n  val first = scala.io.StdIn.readLine()\n  first\n}'
    class Client:
        source_edit_style = 'lines'
        supports_source_edit_repairs = True
        def generate_json(self, prompt, **kwargs):
            assert kwargs['schema'] == line_edit_schema(source)
            assert kwargs['max_output_tokens'] == 4096
            assert '3:     val text = bad' in prompt
            response = line_edits(source, dict(start_line=3, end_line=3, replacement=replacement))
            if stale:
                response['source_sha256'] = '0' * 64
            return json.dumps(response)
    candidate = refine_code(Client(), TaskSpec('Read one line and print it.', 'other', ''),
        CodeCandidate(source, 'other'), SimpleNamespace(passed=0,total=1,pass_rate=0,
        failures=[dict(status=status, stderr='Visible failure')], first_failure=None), language='scala')
    if stale:
        assert candidate.raw_code == source and candidate.repair_validation_error
    else:
        assert '    val text = {\n      val first = scala.io.StdIn.readLine()\n      first\n    }\n    println(text)' in candidate.raw_code
        assert candidate.repair_validation_error is None
