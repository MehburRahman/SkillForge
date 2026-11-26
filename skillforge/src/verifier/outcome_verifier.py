from dataclasses import dataclass, field, replace
import json
from typing import Any, List, Optional

from src.agents.test_generator import FunctionTestCase, TestCase
from src.verifier.http_contract import HTTPTestCase
from src.benchmark.execution_mode import ExecutionMode, validate_execution
from src.verifier.sandbox import ExecutionResult, run_code_many, Status


@dataclass
class TestCaseVerification:
    input: Any
    expected: str
    actual: str
    category: str
    passed: bool
    execution: ExecutionResult


@dataclass
class VerificationResult:
    pass_rate: float
    total: int
    passed: int
    first_failure: Optional[dict] = None
    failures: List[dict] = field(default_factory=list)
    case_results: List[TestCaseVerification] = field(default_factory=list)

    @property
    def execution_results(self) -> List[ExecutionResult]:
        """Expose the exact executions shared by deterministic graders."""
        return [case_result.execution for case_result in self.case_results]


def _display_output(value: Any) -> str:
    """Return a stable, human-readable representation of a test output."""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _outputs_match(actual: str, expected: Any) -> bool:
    """Compare plain-text output, while allowing JSON arrays and objects."""
    expected_text = _display_output(expected)
    if actual == expected_text:
        return True

    try:
        return json.loads(actual) == json.loads(expected_text)
    except json.JSONDecodeError:
        return False


def verify(
    code: str,
    test_suite,
    max_failures: int = 3,
    timeout: float = 5,
    *,
    language="python",
    execution_mode="stdio",
    entry_point=None,
    on_progress=None,
) -> VerificationResult:
    """Score a candidate and retain a bounded set of repair examples."""
    if max_failures < 0:
        raise ValueError("max_failures must be zero or greater")
    language, execution_mode, entry_point = validate_execution(language, execution_mode, entry_point)
    function_mode = execution_mode == ExecutionMode.FUNCTION
    http_mode = execution_mode == ExecutionMode.HTTP
    expected_type = HTTPTestCase if http_mode else FunctionTestCase if function_mode else TestCase
    if any(not isinstance(case, expected_type) for case in test_suite.cases):
        raise ValueError(f"test suite cases do not match execution_mode={execution_mode}")

    passed = 0
    failures = []
    seen_http_failures = set()
    case_results = []
    options = {"on_progress": on_progress} if on_progress is not None else {}
    try:
        execution_results = run_code_many(
            code,
            [(dict(requests=case.requests,clean_environment=True,port=case.port) if case.clean_environment else case.requests)
             if http_mode else case.assertion if function_mode else case.input for case in test_suite.cases],
            timeout=timeout,
            language=language, execution_mode=execution_mode, entry_point=entry_point,
            **options,
        )
    except RuntimeError as exc:
        if max_failures == 0:
            raise RuntimeError('Hidden execution infrastructure failed; check toolchain setup and trusted harness.') from exc
        raise

    for case, result in zip(test_suite.cases, execution_results):
        test_input = case.assertion if function_mode else case.input
        actual = result.status.value if function_mode or http_mode else result.stdout.strip()
        expected = "pass" if function_mode or http_mode else _display_output(case.expected_output)
        case_passed = result.status == Status.PASS and (
            function_mode or http_mode or _outputs_match(actual, case.expected_output)
        )
        # Hidden verification suppresses all case content, including assertion
        # tracebacks that the runtime grader could otherwise pass to a judge.
        execution = replace(result, stdout="", stderr="", http_failure={}, http_failures=[], http_observations=[], http_startup={}) if max_failures == 0 else result
        case_results.append(
            TestCaseVerification(
                input="" if max_failures == 0 else test_input,
                expected="" if max_failures == 0 else expected,
                actual="" if max_failures == 0 else actual,
                category=case.category,
                passed=case_passed,
                execution=execution,
            )
        )

        if case_passed:
            passed += 1
        elif len(failures) < max_failures:
            failures.append({
                "input": test_input,
                "expected": expected,
                "actual": actual,
                "stderr": result.stderr,
                "status": result.status.value,
                "category": case.category,
                "exit_code": result.exit_code,
                "duration_seconds": result.duration_seconds,
            })
            if http_mode and result.http_failure:
                base=failures.pop()
                for detail in result.http_failures or [result.http_failure]:
                    identity=json.dumps([detail['request'],detail['expected'],detail['actual'].get('status'),
                        detail['actual'].get('body'),detail['actual'].get('phase')],sort_keys=True,ensure_ascii=False)
                    if identity in seen_http_failures:continue
                    seen_http_failures.add(identity)
                    if len(failures)>=max_failures:break
                    failures.append(dict(base,
                        input=json.dumps({key:detail[key] for key in ('request_number','request','prior_requests')},ensure_ascii=False),
                        expected=json.dumps(detail['expected'],ensure_ascii=False),
                        actual=json.dumps(detail['actual'],ensure_ascii=False),
                        status=detail.get('execution_status',base['status'])))

    total = len(test_suite.cases)
    return VerificationResult(
        pass_rate=passed / total if total else 0,
        total=total,
        passed=passed,
        first_failure=failures[0] if failures else None,
        failures=failures,
        case_results=case_results,
    )
