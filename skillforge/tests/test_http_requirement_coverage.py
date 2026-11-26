from types import SimpleNamespace

from src.verifier.http_contract import build_http_case
from src.verifier.http_requirements import execution_ledger, extract_http_requirements, library_violations

STATEMENT = """Requirements
- Use only Scala and JDK libraries.
- Escape HTML special characters in dynamic values.
"""


def _run(status='pass', passed=True):
    observation = dict(request_number=1, method='POST', path='/render', passed=passed, body_checked=True,
                       parallel_group=0, unicode=False, newlines=False, expected_status=200)
    return SimpleNamespace(execution=SimpleNamespace(status=SimpleNamespace(value=status),
                                                     http_observations=[observation], http_startup={}))


def _states(ledger):
    return {row['text']: row['state'] for row in ledger['requirements']}


def test_supplied_scenarios_verify_the_requirements_they_declare():
    requirements, _ = extract_http_requirements(STATEMENT)
    case = build_http_case(dict(requests=[dict(method='POST', path='/render', body='', expected_status=200,
                                               expected_body='x')],
                                requirements=['Escape HTML special characters in dynamic values.']))
    suite = SimpleNamespace(cases=[case])
    code = 'import com.sun.net.httpserver.HttpServer\nimport java.net.URLDecoder\nobject Main'
    ledger = execution_ledger(requirements, suite, SimpleNamespace(case_results=[_run()]), code, independent=True)
    states = _states(ledger)
    assert states['Escape HTML special characters in dynamic values.'] == 'verified'
    assert states['Use only Scala and JDK libraries.'] == 'verified'
    failed = execution_ledger(requirements, suite, SimpleNamespace(case_results=[_run(passed=False)]), code,
                              independent=True)
    assert _states(failed)['Escape HTML special characters in dynamic values.'] == 'failed'


def test_generated_tests_cannot_claim_requirement_coverage():
    requirements, _ = extract_http_requirements(STATEMENT)
    case = build_http_case(dict(requests=[dict(method='GET', path='/', body='', expected_status=200,
                                               expected_body='x')],
                                requirements=['Escape HTML special characters in dynamic values.']))
    ledger = execution_ledger(requirements, SimpleNamespace(cases=[case]), SimpleNamespace(case_results=[_run()]),
                              'object Main', independent=False)
    assert _states(ledger)['Escape HTML special characters in dynamic values.'] == 'unverified'


def test_library_check_rejects_external_imports_and_dependency_directives():
    assert library_violations('import scala.util.Try\nimport java.net.URI', False) == []
    assert library_violations('import io.circe.Json', False) == ['imports io.circe.Json']
    assert library_violations('//> using dep com.lihaoyi::upickle:3.1.0\nobject Main', False)
    assert library_violations('import java.net.ServerSocket', True) == [
        'does not use com.sun.net.httpserver.HttpServer']


def test_requirement_claims_survive_revalidation_of_a_built_case():
    from dataclasses import asdict
    case = build_http_case(dict(requests=[dict(method='GET', path='/', body='', expected_status=200,
                                               expected_body='x')], requirements=['Use only Scala and JDK libraries.']))
    assert build_http_case(asdict(case)).requirements == ('Use only Scala and JDK libraries.',)
