"""Real Scala integration, independent of Ollama and production experiment state."""
from dataclasses import replace
import os
from pathlib import Path
import subprocess

import pytest

from src.agents.code_prompts import execution_contract
from src.agents.test_generator import TestSuite as Suite, TestCase as Case, FunctionTestCase, build_test_suite
from src.benchmark.execution_mode import validate_execution
from src.evaluation.pipeline import evaluate_candidate
from src.evaluation.runtime_grader import grade_runtime_robustness
from src.evolution.scope import validate_evolution_scope
from src.verifier.outcome_verifier import verify
from src.verifier.runners import scala
from src.verifier.runners.base import Status
from src.verifier.sandbox import run_code, run_code_many

try:
    scala.preflight()
    HAVE_SCALA = True
except RuntimeError:
    HAVE_SCALA = False
requires_scala = pytest.mark.skipif(not HAVE_SCALA, reason='run tools/setup_scala.py with Scala CLI and JDK 21 installed')


@requires_scala
def test_compile_once_per_batch_unicode_and_compilation_accounting(monkeypatch):
    builds = []
    compile_sources = scala.compile_sources
    def counted(*args, **kwargs):
        builds.append(1)
        return compile_sources(*args, **kwargs)
    monkeypatch.setattr(scala, 'compile_sources', counted)
    source = 'object Main { def main(args: Array[String]): Unit = print(scala.io.Source.stdin.mkString) }'
    results = run_code_many(source, ['α\n  β\n', 'new request'], language='scala')
    assert [r.stdout for r in results] == ['α\n  β\n', 'new request']
    assert all(r.status == Status.PASS for r in results)
    assert len(builds) == 1
    assert results[0].compilation_seconds > 0 and results[1].compilation_seconds == 0
    assert results[0].execution_environment['scala'] == scala.SCALA_VERSION


@requires_scala
def test_native_assertions_failure_exception_and_early_exit():
    code = '''object Solution { def solve(x: Long): Long = {
      if (x == -1) throw new IllegalArgumentException("bad argument")
      if (x == -2) System.exit(0)
      x * 2
    }}'''
    results = run_code_many(code, ['assert(solve(3000000000L) == 6000000000L)',
                                   'assert(solve(2) == 5)', 'solve(-1)', 'solve(-2)'],
                            language='scala', execution_mode='function', entry_point='solve')
    assert [r.status for r in results] == [Status.PASS, Status.FAIL, Status.ERROR, Status.ERROR]


@requires_scala
def test_compile_failure_is_feedback_and_hidden_diagnostics_are_suppressed():
    suite = Suite([Case('7', '14', 'normal')])
    code = 'object Main { def main(args: Array[String]): Unit = { val x: Int = "wrong" } }'
    visible = verify(code, suite, language='scala')
    assert visible.passed == 0
    assert visible.execution_results[0].status == Status.COMPILE_ERROR
    runtime = grade_runtime_robustness(visible)
    assert runtime.compilation_error_count == 1 and runtime.compilation_seconds > 0
    hidden = verify(code, suite, language='scala', max_failures=0)
    assert hidden.failures == []
    assert hidden.execution_results[0].stderr == ''
    assert hidden.case_results[0].input == ''


@requires_scala
def test_runtime_timeout_and_output_limit():
    code = '''object Main { def main(args: Array[String]): Unit = {
       if (scala.io.StdIn.readLine() == "loop") { while (true) {} }
       else { while (true) print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx") }
    }}'''
    results = run_code_many(code, ['loop', 'output'], timeout=1, language='scala')
    assert results[0].status == Status.TIMEOUT
    assert results[1].status == Status.ERROR
    assert 'Output limit' in results[1].stderr
    assert len(results[1].stdout) <= 262144


@requires_scala
def test_general_function_contract_supports_four_dimensional_state():
    # A runner fixture, never injected into generation prompts or a problem catalog.
    code = '''object Solution {
      def solve(a: String, b: String, c: String, d: String): Int = {
        val dp = Array.ofDim[Int](a.length+1,b.length+1,c.length+1,d.length+1)
        for(i <- 1 to a.length; j <- 1 to b.length; k <- 1 to c.length; l <- 1 to d.length) {
          dp(i)(j)(k)(l) = if(a(i-1)==b(j-1) && a(i-1)==c(k-1) && a(i-1)==d(l-1))
            dp(i-1)(j-1)(k-1)(l-1)+1
          else List(dp(i-1)(j)(k)(l),dp(i)(j-1)(k)(l),dp(i)(j)(k-1)(l),dp(i)(j)(k)(l-1)).max
        }
        dp(a.length)(b.length)(c.length)(d.length)
      }
    }'''
    # Small cases are independently solved by exhaustive subsequence enumeration.
    import itertools
    strings = [('abc','bac','cab','ab'), ('','abc','abc','abc'), ('aaaa','aa','aaa','a')]
    checks = []
    import json
    for a,b,c,d in strings:
        def subseq(x,y):
            iterator = iter(y)
            return all(any(z == ch for z in iterator) for ch in x)
        expected = max(len(s) for mask in itertools.product([False,True], repeat=len(a))
                       if all(subseq(s := ''.join(ch for ch,keep in zip(a,mask) if keep), t) for t in (b,c,d)))
        checks.append(FunctionTestCase(f'assert(solve({",".join(json.dumps(s) for s in (a,b,c,d))}) == {expected})'))
    assert verify(code, Suite(checks), language='scala', execution_mode='function', entry_point='solve').pass_rate == 1


def test_contract_has_no_fixed_algorithm_or_http_route():
    contract = execution_contract('scala', 'http')
    assert 'SKILLFORGE_PORT_FILE' in contract
    assert '/process' not in contract and 'reverse' not in contract and 'dynamic programming' not in contract
    assert 'object Solution' in execution_contract('scala', 'function', 'compute')
    with pytest.raises(ValueError):
        validate_execution('scala','function','class')
    with pytest.raises(ValueError):
        validate_execution('python','http')


def test_invalid_build_directives_are_rejected_before_compiling(monkeypatch):
    monkeypatch.setattr(scala, 'preflight', lambda mode: {})
    monkeypatch.setattr(scala, 'toolchain', lambda: ('scala-cli','java','21'))
    result = run_code('//> using dep example:bad:1\nobject Main {}', language='scala')
    assert result.status == Status.COMPILE_ERROR


@requires_scala
@pytest.mark.parametrize('assertion', ['assert(solve(1) ==', 'val unrelated: Int = "bad"', 'assert(unknownHelper == 1)'])
def test_broken_trusted_harness_is_an_infrastructure_failure(assertion):
    with pytest.raises(RuntimeError, match='assertion harness'):
        run_code_many('object Solution { def solve(x: Int): Int = x }', [assertion],
                      language='scala', execution_mode='function', entry_point='solve')


@requires_scala
def test_missing_function_is_candidate_interface_feedback():
    result = run_code_many('object Solution { def other(x: Int): Int = x }', ['assert(solve(1) == 1)'],
                           language='scala', execution_mode='function', entry_point='solve')
    assert result[0].status == Status.COMPILE_ERROR


def test_candidate_compilation_timeout_records_batch_timing(monkeypatch):
    monkeypatch.setattr(scala, 'preflight', lambda mode: {})
    monkeypatch.setattr(scala, 'toolchain', lambda: ('scala-cli', 'java', '21'))
    monkeypatch.setattr(scala, 'compile_sources', lambda *a, **kw: scala.ExecutionResult(
        Status.TIMEOUT, '', 'Execution timed out', -9, 180))
    results = run_code_many('object Main {}', ['a', 'b'], language='scala')
    assert [r.status for r in results] == [Status.TIMEOUT, Status.TIMEOUT]
    assert sum(r.compilation_seconds for r in results) == 180
    assert sum(r.duration_seconds for r in results) == 0


def test_scala_analysis_cache_changes_with_toolchain_and_contract(monkeypatch, tmp_path):
    from src.agents.task_spec_cache import TaskSpecCache
    cache = TaskSpecCache(tmp_path)
    identity = cache.identity('A problem', 'model', 'scala')
    monkeypatch.setattr(scala, 'SCALA_VERSION', '3.changed')
    assert cache.identity('A problem', 'model', 'scala') != identity


def test_repeated_workspace_setup_preserves_evolved_prompt_state(tmp_path):
    from tools.setup_scala import initialize_workspace
    workspace = initialize_workspace('function', tmp_path)
    registry = workspace / 'prompts.json'
    registry.write_text('saved evolved state')
    assert initialize_workspace('function', tmp_path) == workspace
    assert registry.read_text() == 'saved evolved state'
    assert initialize_workspace('http', tmp_path) != workspace


@requires_scala
def test_hidden_harness_infrastructure_error_does_not_publish_assertions():
    secret = 'assert(privateHiddenHelper == 1)'
    with pytest.raises(RuntimeError) as error:
        verify('object Solution { def solve(x: Int): Int = x }', Suite([FunctionTestCase(secret)]),
               language='scala', execution_mode='function', entry_point='solve', max_failures=0)
    assert 'privateHiddenHelper' not in str(error.value)


def test_scala_workspace_cannot_consume_python_training_evidence():
    from types import SimpleNamespace
    task = SimpleNamespace(language='scala', execution_mode='stdio')
    store = SimpleNamespace(training_execution_scopes=lambda: [('python','stdio')],
                            training_reasoning_modes=lambda: ['direct'])
    with pytest.raises(ValueError, match='separate --history-db'):
        validate_evolution_scope([task], store)
    assert validate_evolution_scope([task]) == ('scala','stdio')


@requires_scala
def test_compiler_feedback_repairs_through_real_pipeline_and_persists(tmp_path):
    from src.reinforcement_loop import run_candidate_refinement_loop
    from src.history import ExperimentStore
    from tests.test_benchmark_runner import BenchmarkClient, make_registry
    class Client(BenchmarkClient):
        def generate(self, prompt, system=None, temperature=.2, *, seed=None):
            if 'problem analyzer' in (system or '').lower():
                return super().generate(prompt, system, temperature, seed=seed)
            self.requests.append({'prompt':prompt,'system':system or ''})
            if 'repair compiler errors' in (system or '').lower():
                assert 'Compiler diagnostics:' in prompt and 'unknownName' in prompt
                return 'object Main { def main(args: Array[String]): Unit = println(scala.io.StdIn.readInt()*2) }'
            return 'object Main { def main(args: Array[String]): Unit = println(unknownName) }'
    with ExperimentStore(tmp_path/'history.db') as store:
        result = run_candidate_refinement_loop(Client(), 'Read one integer and print twice its value.',
            trusted_test_suite=Suite([Case('4','8','normal')]), max_attempts=2, language='scala',
            prompt_registry=make_registry(tmp_path/'prompts.json'), history_store=store)
        assert result.success and len(result.attempts)==2
        attempts = store.get_attempts(result.history_run_id)
        import json
        grade = json.loads(attempts[0]['runtime_grade_json'])
        assert grade['compilation_error_count'] == 1
        assert grade['execution_environment']['scala'] == scala.SCALA_VERSION


@requires_scala
def test_scala_paired_validation_is_real_and_does_not_expose_hidden_inputs(tmp_path):
    from src.benchmark.models import BenchmarkDataset
    from src.evolution.validator import validate_candidate_prompt
    from src.history import ExperimentStore
    from tests.test_benchmark_runner import BenchmarkClient, make_registry, write_dataset
    class Client(BenchmarkClient):
        def generate(self, prompt, system=None, temperature=.2, *, seed=None):
            if 'problem analyzer' in (system or '').lower():
                return super().generate(prompt, system, temperature, seed=seed)
            self.requests.append({'prompt':prompt,'system':system or ''})
            if 'CANDIDATE SYSTEM PROMPT' in (system or ''):
                return 'object Main { def main(args: Array[String]): Unit = println(scala.io.StdIn.readLine().split(" ").map(_.toInt).sum) }'
            return 'object Main { def main(args: Array[String]): Unit = println(0) }'
    task = replace(write_dataset(tmp_path/'tasks.json').tasks[1], language='scala')
    registry = make_registry(tmp_path/'prompts.json')
    client = Client()
    with ExperimentStore(tmp_path/'history.db') as store:
        result = validate_candidate_prompt(client,BenchmarkDataset((task,)),registry,store,'P1',max_attempts=1)
        assert result.comparison.candidate_only_passes == 1
        assert not result.should_promote  # One success is insufficient evidence.
    assert '401 402' not in str(client.requests)
