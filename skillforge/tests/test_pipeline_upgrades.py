"""Execution-driven checks with isolated model doubles and real subprocesses."""
import json
from types import SimpleNamespace

import httpx
import pytest

from src.agents.differential import expand_suite
from src.agents.test_generator import build_test_suite
from src.agents.test_generator import generate_tests
from src.agents.test_oracle import ExecutableTestOracle
from src.agents.llm_client import LLMClient
import src.agents.llm_client as llm


BUILDER = '''def build_inputs():
    import random
    rng = random.Random(42)
    return [dict(input=str(x), category='normal') for x in rng.sample(range(100), 32)]
'''


class SourceClient:
    def __init__(self, source=BUILDER):
        self.source = source
    def generate(self, *args, **kwargs):
        return self.source


def oracle():
    reference = ExecutableTestOracle(SourceClient(), 'Read an integer; print twice its value.')
    reference.code = 'def solve_input(input_text: str) -> str:\n    return str(2 * int(input_text))'
    return reference


def test_random_inputs_have_executed_answers_and_reproducible_provenance():
    outputs = []
    for _ in range(2):
        suite = build_test_suite([dict(input='100', expected_output='200', category='edge')])
        expand_suite(SourceClient(), 'Double an integer.', suite, oracle(), count=32, seed=42)
        assert len(suite.cases) == 33
        assert all(case.expected_output == str(2 * int(case.input)) for case in suite.cases)
        assert suite.validation['differential']['added'] == 32
        assert not suite.validation['differential']['independent_proof']
        outputs.append(([case.input for case in suite.cases], suite.validation))
    assert outputs[0] == outputs[1]


def test_program_generation_adds_random_cases_after_reference_validation():
    from tests.test_executable_test_builder import PlanningClient, program_responses, spec
    from tests.test_executable_test_oracle import COUNTED_SUM_REFERENCE
    from tests.test_generation_recovery import uncertain_reviews, anchors
    builder = '''def build_inputs():
    import random
    rng = random.Random(42)
    return [dict(input='2 ' + str(100 + x) + ' ' + str(rng.randrange(100)), category='normal')
            for x in rng.sample(range(100), 32)]
'''
    client = PlanningClient([*program_responses()[:3], uncertain_reviews(), COUNTED_SUM_REFERENCE, builder])
    suite = generate_tests(client, spec(), method='program', reference_test_suite=anchors(),
                           differential_cases=32, seed=42)
    assert suite.validation['status'] == 'reference_checked'
    assert suite.validation['differential']['added'] == 32
    assert len(suite.cases) == 35
    for case in suite.cases:
        assert int(case.expected_output) == sum(map(int, case.input.split()[1:]))


def test_builder_overshooting_the_requested_count_keeps_the_first_cases():
    suite = build_test_suite([dict(input='100', expected_output='200', category='edge')])
    expand_suite(SourceClient(), 'Double an integer.', suite, oracle(), count=20, seed=42)
    assert len(suite.cases) == 21
    assert suite.validation['differential']['generated'] == 32
    assert suite.validation['differential']['added'] == 20


def test_failed_reference_cannot_change_the_base_suite_or_repair_itself():
    reference = oracle()
    reference.code = 'def solve_input(input_text: str) -> str:\n    raise ValueError("broken reference")'
    before = reference.code
    suite = build_test_suite([dict(input='100', expected_output='200', category='edge')])
    with pytest.raises(ValueError, match='reference execution failed'):
        expand_suite(SourceClient(), 'Double.', suite, reference, count=32, seed=42)
    assert len(suite.cases) == 1 and reference.code == before and reference.calls == 0


@pytest.mark.parametrize('source', [
    'def build_inputs():\n    return [{"input": "1", "expected_output": "2", "category": "normal"}]',
    'def build_inputs():\n    while True: pass',
])
def test_input_builder_rejects_answers_and_bounds_nontermination(source):
    suite = build_test_suite([dict(input='100', expected_output='200', category='edge')])
    with pytest.raises(ValueError, match='construction failed'):
        expand_suite(SourceClient(source), 'Double.', suite, oracle(), count=32, seed=42)
    assert len(suite.cases) == 1


def test_schema_rejection_falls_back_without_spending_a_new_candidate(monkeypatch):
    requests = []
    class BadGrammar(Exception):
        status_code = 400
    def chat(**request):
        requests.append(request.copy())
        if 'format' in request:
            raise BadGrammar('failed to parse grammar')
        return {'message': {'content': '{"answer": 2}'}}
    monkeypatch.setattr(llm, 'ollama', SimpleNamespace(Client=lambda **kw: SimpleNamespace(chat=chat)))
    client = LLMClient()
    assert json.loads(client.generate_json('Return an answer', schema={'type': 'object'}, seed=42)) == {'answer': 2}
    assert [request.get('format') for request in requests] == [{'type': 'object'}, 'json', None]
    assert all(request['options']['seed'] == 42 for request in requests)
    assert client.last_usage['backend_request_count'] == 3
    assert len(client.last_usage['infrastructure_retries']) == 2


def test_transport_retries_are_bounded_and_recorded(monkeypatch):
    calls = []
    def chat(**request):
        calls.append(request)
        raise httpx.ConnectError('offline')
    monkeypatch.setattr(llm, 'ollama', SimpleNamespace(Client=lambda **kw: SimpleNamespace(chat=chat)))
    monkeypatch.setattr(llm.time, 'sleep', lambda seconds: None)
    client = LLMClient()
    with pytest.raises(RuntimeError, match='offline'):
        client.generate('code')
    assert len(calls) == 3
    assert len(client.last_usage['infrastructure_retries']) == 2


def test_line_edit_that_unbalances_brackets_is_rejected():
    import hashlib
    import pytest
    from src.agents.source_edits import apply_line_edits, delimiter_problem
    source = "object A {\n  val r = f(x, m => {\n    g(m)\n  })\n}\n"
    assert delimiter_problem(source) is None
    response = {"diagnosis": "type", "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "edits": [{"start_line": 2, "end_line": 2, "replacement": "  val r = f(x, m => g(m))"}]}
    with pytest.raises(ValueError, match="balanced brackets"):
        apply_line_edits(source, response)


def test_delimiter_problem_ignores_literals():
    from src.agents.source_edits import delimiter_problem
    assert delimiter_problem('val s = "{{x}}"; val r = """\\{\\{(\\w+)\\}\\}""".r; val c = \'(\'') is None
    assert delimiter_problem("f(a") == "'(' opened on line 1 is never closed"


def test_replace_all_in_without_quote_replacement_gets_hint():
    from src.agents.refinement_agent import _scala_parsing_diagnosis
    code = 'val out = re.replaceAllIn(t, m => values(m.group(1)))'
    failures = [{"input": '{"body": "price=%241"}', "stderr": ""}]
    hint = _scala_parsing_diagnosis(failures, "scala", "http", code)
    assert "quoteReplacement" in hint
    quoted = 'val out = re.replaceAllIn(t, m => Regex.quoteReplacement(values(m.group(1))))'
    assert _scala_parsing_diagnosis(failures, "scala", "http", quoted) == ""


def test_escaping_diagnosis_detects_missing_and_double_escaping():
    import json
    from src.agents.refinement_agent import _escaping_diagnosis
    missing = [{"expected": json.dumps({"status": 200, "body": "<p>&lt;b&gt; &amp;</p>"}),
                "actual": json.dumps({"status": 200, "body": "<p><b> &</p>"})}]
    assert "inserts values without escaping" in _escaping_diagnosis(missing)
    extra = [{"expected": "a < b", "actual": "a &lt; b"}]
    assert "escaped twice" in _escaping_diagnosis(extra)
    assert _escaping_diagnosis([{"expected": "1", "actual": "2"}]) == ""


def test_non_local_return_warning_becomes_repair_evidence(monkeypatch):
    import src.verifier.source_check as source_check
    from src.agents.refinement_agent import _scala_warning_diagnosis

    class Result:
        stderr = ("-- Warning: /tmp/x/Candidate.scala:7:40 \n"
                  "7 |      case None => respond(400); return\n"
                  "  |                                 ^^^^^^\n"
                  "  |Non local returns are no longer supported; use `boundary` instead\n")
    monkeypatch.setattr(source_check, "check_source", lambda code, language: Result())
    hint = _scala_warning_diagnosis("val f = x => { return }", "scala")
    assert "line 7" in hint and "non-local return" in hint
    assert _scala_warning_diagnosis("val f = x => x", "scala") == ""
