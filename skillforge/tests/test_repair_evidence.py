"""Regression coverage for failures observed in the real Scala browser baseline."""
from types import SimpleNamespace

import pytest

from src.agents.code_generator import CodeCandidate
from src.agents.refinement_agent import refine_code
from src.agents.task_spec import TaskSpec
from src.evaluation import build_evaluation_report
from src.feedback import collect_feedback
from tests.test_feedback_collector import functional, runtime, constraint, judge
from tests.test_experiment_configs import RecordingClient


PARSING_ERROR = ('Exception in thread "main" java.lang.NumberFormatException: For input string: "3 6"\n'
                 '\tat scala.io.StdIn.readInt(StdIn.scala:122)\n\tat Main$.main(Candidate.scala:3)')


def repair(failures, *, language='scala', mode='stdio', feedback_mode='full', bundle=None, code=None):
    client = RecordingClient(('repaired',))
    spec = TaskSpec('Read n and target followed by n integers.', 'math', 'Whitespace-separated integers')
    candidate = CodeCandidate(code or 'object Main { val n = scala.io.StdIn.readInt() }', 'math')
    result = SimpleNamespace(passed=0, total=3, pass_rate=0, failures=failures, first_failure=None)
    refine_code(client, spec, candidate, result, language=language, execution_mode=mode,
                entry_point='solve' if mode == 'function' else None,
                feedback_mode=feedback_mode, feedback_bundle=bundle)
    return client.requests[0]['prompt']


def test_compiler_feedback_explains_uppercase_pattern_bindings_without_task_knowledge():
    prompt = repair([dict(status='compile_error', stderr='Not found: UpperName')],
                    code='object Main { val Array(UpperName, OtherName) = Array("a", "b") }')
    assert 'Scala pattern diagnosis:' in prompt
    assert 'lowercase binding names' in prompt


def test_type_mismatch_uses_complete_source_and_traces_actual_and_required_types():
    class Client:
        source_edit_style = 'lines'
        supports_source_edit_repairs = True
        model = 'fixture'
        def generate(self,prompt,**kwargs):
            assert 'Type mismatch repair:' in prompt
            assert 'value construction' in prompt
            assert 'Numbered ORIGINAL source' not in prompt
            return 'corrected source'
        def generate_json(self,*args,**kwargs):
            pytest.fail('Type mismatch requires complete-source repair')
    verification = SimpleNamespace(passed=0,total=1,pass_rate=0,first_failure=None,
        failures=[dict(status='compile_error',stderr='Type Mismatch Error: Found: Reader Required: InputStream')])
    assert refine_code(Client(),TaskSpec('Implement the supplied interface.','other',''),
        CodeCandidate('old source','other'),verification,language='scala',execution_mode='http').raw_code == 'corrected source'


def test_private_member_error_explains_scopes_without_application_specific_names():
    prompt = repair([dict(status='compile_error',stderr='private value cache cannot be accessed from class Worker')],
                    mode='http',code='object Store { private val cache = 1 }; class Worker')
    assert 'Scala access-control diagnosis:' in prompt
    assert 'brace nesting' in prompt and 'Preserve required synchronization' in prompt


@pytest.mark.parametrize(('status','diagnostic'), [
    ('timeout','Received response status 404, but its body did not complete.'),
    ('error','Remote end closed connection without response'),
    ('fail',"got status 204, body ''"),
])
def test_http_timeout_requests_complete_source_with_protocol_feedback(status, diagnostic):
    class Native:
        model = 'fixture'
        supports_source_edit_repairs = True
        source_edit_style = 'lines'
        def generate(self, prompt, **kwargs):
            assert 'HTTP protocol diagnosis:' in prompt
            assert '0 selects chunked streaming' in prompt
            assert 'Numbered ORIGINAL source' not in prompt
            return 'fixed source'
        def generate_json(self, *args, **kwargs):
            pytest.fail('A diagnosed HTTP protocol failure must not request a line patch')
    verification = SimpleNamespace(passed=0,total=1,pass_rate=0,first_failure=None,
        failures=[dict(status=status,stderr=diagnostic)])
    candidate = refine_code(Native(), TaskSpec('Implement the specified service.', 'other', ''),
        CodeCandidate('previous server source', 'other'), verification, language='scala',execution_mode='http')
    assert candidate.raw_code == 'fixed source'


def test_runtime_diagnostics_survive_three_functional_failures():
    cases = [{'input': str(i), 'expected': 'ok', 'actual': '', 'status': 'error'} for i in range(3)]
    diagnostics = [{'status': 'error', 'exit_code': 1, 'stderr': PARSING_ERROR} for _ in range(3)]
    report = build_evaluation_report(functional(passed=0, total=3, evidence=cases),
                                     runtime(total=3, errors=3, evidence=diagnostics), constraint())
    feedback = collect_feedback(report, test_source='trusted')
    text = feedback.candidate_feedback
    assert 'NumberFormatException' in text and 'Candidate.scala:3' in text
    assert text.index('Runtime evidence:') < text.index('Functional evidence:')
    assert text.count('Runtime evidence:') == 1  # identical stack traces are not repeated
    assert text.count('Functional evidence:') == 2


def test_http_repair_uses_executed_reference_and_not_unproven_judge_distractions():
    client = RecordingClient(('corrected source',))
    failures = [dict(status='fail', input='{"method":"POST","path":"/arbitrary"}',
                     expected='{"status":200,"body":"hello"}',
                     actual='{"status":204,"body":""}')]
    report = build_evaluation_report(functional(passed=0, evidence=failures), runtime(), constraint(),
                                    judge(feedback='UNPROVEN_JUDGE_DISTRACTION'))
    reference = 'def expected_responses(requests):\n    return []'
    refine_code(client, TaskSpec('Follow the supplied service contract.', 'other', ''),
                CodeCandidate('old source', 'other'),
                SimpleNamespace(passed=0,total=1,pass_rate=0,failures=failures,first_failure=None),
                language='scala', execution_mode='http', evaluation_report=report,
                feedback_bundle=collect_feedback(report, test_source='generated'), reference_program=reference)
    prompt = client.requests[0]['prompt']
    assert reference in prompt and 'not an independent correctness proof' in prompt
    assert '204' in prompt and 'hello' in prompt
    assert 'UNPROVEN_JUDGE_DISTRACTION' not in prompt


def test_http_text_byte_mismatch_gets_language_diagnosis_without_task_rules():
    from src.agents.refinement_agent import _http_protocol_diagnosis
    code = 'exchange.sendResponseHeaders(200, payload.length)\nos.write(payload.getBytes("UTF-8"))'
    text = _http_protocol_diagnosis([dict(status='fail')], 'scala', 'http', code)
    assert 'HTTP byte-length diagnosis:' in text
    assert 'Encode the response once' in text
    assert not _http_protocol_diagnosis([], 'scala', 'http', code)
    assert not _http_protocol_diagnosis([dict(status='fail')], 'scala', 'http',
        'val bytes = payload.getBytes("UTF-8")\nexchange.sendResponseHeaders(200, bytes.length)\nos.write(bytes)')


def test_single_evidence_slot_preserves_runtime_cause():
    report = build_evaluation_report(functional(passed=0, evidence=[{'input': 'x'}]),
        runtime(errors=1, evidence=[{'status': 'error', 'stderr': PARSING_ERROR}]), constraint())
    text = collect_feedback(report, 'trusted', max_evidence_items=1).candidate_feedback
    assert 'NumberFormatException' in text and 'Functional evidence:' not in text


def test_compiler_preamble_does_not_hide_source_location_and_required_type():
    diagnostic = ('Tool progress\n' * 200 + '-- [E007] Type Mismatch Error: /tmp/Candidate.scala:7:36\n'
                  '7 | new Array[Long](n + 2)\n  | Found: Long\n  | Required: Int\nCompilation failed')
    report = build_evaluation_report(functional(passed=0),
        runtime(errors=1, evidence=[{'status': 'compile_error', 'stderr': diagnostic}]), constraint())
    feedback = collect_feedback(report, 'trusted')
    assert 'Candidate.scala:7:36' in feedback.candidate_feedback
    assert 'Required: Int' in feedback.candidate_feedback
    assert len(feedback.candidate_feedback) < 2500


def test_repair_includes_specific_visible_parsing_diagnosis():
    prompt = repair([{'input': '3 6\n1 3 4', 'stderr': PARSING_ERROR, 'status': 'error'}])
    assert 'Input parsing diagnosis:' in prompt
    assert 'token parser' in prompt and 'order' in prompt
    assert 'dynamic programming' not in prompt


@pytest.mark.parametrize('options', [
    {'feedback_mode': 'scalar_repair'}, {'language': 'python'}, {'mode': 'function'},
])
def test_parsing_diagnosis_is_scala_program_full_feedback_only(options):
    prompt = repair([{'stderr': PARSING_ERROR}], **options)
    assert 'Input parsing diagnosis:' not in prompt
    if options.get('feedback_mode') == 'scalar_repair':
        assert 'NumberFormatException' not in prompt


def test_hidden_or_unrelated_errors_cannot_supply_parsing_diagnosis():
    assert 'Input parsing diagnosis:' not in repair([])
    assert 'Input parsing diagnosis:' not in repair([{'stderr': 'java.lang.NumberFormatException: bad date'}])


def test_one_line_read_with_multiline_split_receives_reader_diagnosis():
    code = 'val input = scala.io.StdIn.readLine()\nval records = input.split("\\n")\nprintln(records(1))'
    failure = dict(status='error', input='first\nsecond',
                   stderr='java.lang.ArrayIndexOutOfBoundsException: Index 1 out of bounds for length 1')
    prompt = repair([failure], code=code)
    assert 'consumed only the first input line' in prompt
    assert 'consume every required record' in prompt
    assert 'Do not merely guard the array access' in prompt


@pytest.mark.parametrize('code', [
    'val input = scala.io.Source.stdin.mkString\nval records = input.split("\\n")',
    'val input = scala.io.StdIn.readLine()\nval records = input.split(",")',
])
def test_array_errors_without_a_one_line_multiline_split_are_not_misdiagnosed(code):
    failure = dict(status='error', stderr='java.lang.ArrayIndexOutOfBoundsException: Index 1 out of bounds for length 1')
    assert 'Input parsing diagnosis:' not in repair([failure], code=code)


def test_large_http_test_and_runtime_output_are_bounded():
    failures = [{'input': 'input' * 20000, 'expected': 'body' * 20000, 'actual': 'wrong' * 20000,
                 'status': 'error', 'stderr': 'log\n' * 20000 + 'Caused by: IllegalStateException: handler failed'} for _ in range(10)]
    prompt = repair(failures, mode='http')
    assert len(prompt) < 13000
    assert 'handler failed' in prompt and '[truncated]' in prompt


def test_collected_runtime_diagnostic_is_not_duplicated_in_raw_failures():
    failures = [{'input': '3 6', 'stderr': PARSING_ERROR, 'status': 'error'}]
    report = build_evaluation_report(functional(passed=0, evidence=failures),
        runtime(errors=1, evidence=failures), constraint())
    prompt = repair(failures, bundle=collect_feedback(report, 'trusted'))
    assert prompt.count('java.lang.NumberFormatException') == 1


def test_missing_http_response_uses_counterexample_not_stale_algorithm_plan():
    from src.agents.refinement_agent import _http_protocol_diagnosis
    failures=[dict(status='fail',expected='{"status":200,"body":"required value"}',
                   actual='{"status":200,"body":""}')]
    assert 'HTTP response-body diagnosis' in _http_protocol_diagnosis(failures,'scala','http')
    client=RecordingClient(('corrected source',))
    refine_code(client,TaskSpec('Follow the supplied service contract.','other',''),
        CodeCandidate('current source','other'),
        SimpleNamespace(passed=0,total=1,pass_rate=0,failures=failures,first_failure=None),
        language='scala',execution_mode='http',algorithm_plan=SimpleNamespace(text='STALE_PLAN_TO_OMIT'))
    prompt=client.requests[0]['prompt']
    assert 'STALE_PLAN_TO_OMIT' not in prompt and 'required value' in prompt


def test_http_declared_length_evidence_is_explicit_and_not_a_hardcoded_answer():
    from src.agents.refinement_agent import _http_protocol_diagnosis
    failures=[dict(status='error',expected='{"body":"xy","utf8_bytes":2}',
                   actual='{"body":"xy","declared_body_bytes":3,"received_body_bytes":2}')]
    text=_http_protocol_diagnosis(failures,'scala','http')
    assert 'declared response length: 3' in text and 'expected response UTF-8 size: 2' in text
    assert 'do not copy observed counts as constants' in text


def test_unchanged_http_source_switches_to_validated_edits_without_inserting_a_solution():
    import hashlib,json
    source='old source'
    class Native:
        model='fixture';source_edit_style='lines';supports_source_edit_repairs=True
        def generate(self,*a,**k):pytest.fail('A repeated full source should switch representation')
        def generate_json(self,prompt,**kwargs):
            assert 'Numbered ORIGINAL source' in prompt
            return json.dumps(dict(diagnosis='Replace the failing statement',
                source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                edits=[dict(start_line=1,end_line=1,replacement='model corrected source')]))
    verification=SimpleNamespace(passed=0,total=1,pass_rate=0,first_failure=None,
        failures=[dict(status='error',stderr='IncompleteRead(1 bytes read, 1 more expected)')])
    candidate=refine_code(Native(),TaskSpec('Implement the supplied service.','other',''),
        CodeCandidate(source,'other'),verification,language='scala',execution_mode='http',repeated_from=1)
    assert candidate.raw_code=='model corrected source'
    assert candidate.repair_validation_error is None


def test_unchanged_compiler_repair_switches_to_validated_edits():
    import hashlib,json
    source='val broken = missing'
    class Native:
        model='fixture';source_edit_style='lines';supports_source_edit_repairs=True
        def generate(self,*a,**k):pytest.fail('Repeated complete source must switch representation')
        def generate_json(self,prompt,**kwargs):
            assert 'Not found: missing' in prompt
            return json.dumps(dict(diagnosis='Correct the undefined binding',
                source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                edits=[dict(start_line=1,end_line=1,replacement='val fixed = 1')]))
    verification=SimpleNamespace(passed=0,total=1,pass_rate=0,first_failure=None,
        failures=[dict(status='compile_error',stderr='Not found: missing at Candidate.scala:1')])
    candidate=refine_code(Native(),TaskSpec('Implement the supplied task.','other',''),
        CodeCandidate(source,'other'),verification,language='scala',repeated_from=1,force_fresh_source=True)
    assert candidate.raw_code=='val fixed = 1'
    assert candidate.repair_validation_error is None
