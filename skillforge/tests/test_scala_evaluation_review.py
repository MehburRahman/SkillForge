"""Protect evaluation provenance and keep hidden scoring outside model feedback."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

import tools.review_scala_evaluation as review


def record(task_id='scala_shortest_paths_train'):
    def attempt(number, code, passed):
        return {'attempt_number': number, 'candidate_code': code, 'hard_gate_passed': passed,
                'functional_grade': {'score': int(passed)},
                'runtime_grade': {'compilation_seconds': 2, 'total_duration_seconds': 0.1,
                                  'execution_environment': {'scala': '3.3.8'},
                                  'failure_evidence': []}}
    return {'task': {'task_id': task_id},
            'job': {'status': 'completed', 'result': {'code': 'fixed', 'model_calls': 4, 'wall_clock_seconds': 10}},
            'run': {'task_id': task_id, 'run_id': 'r1', 'benchmark_split': 'train',
                    'language': 'scala', 'execution_mode': 'stdio', 'model_name': 'test-model',
                    'seed': 42, 'config_name': 'full', 'test_source': 'trusted',
                    'attempt_budget': 3,
                    'best_attempt_number': 3, 'stop_reason': 'passed',
                    'attempts': [attempt(1, 'broken', False), attempt(2, 'broken', False), attempt(3, 'fixed', True)]}}


def workspace(tmp_path):
    catalog = review.PROJECT / 'data/scala_algorithms.json'
    manifest = {'language': 'scala', 'execution_mode': 'stdio', 'model': 'test-model',
                'seed': 42, 'config': 'full', 'max_attempts': 3, 'toolchain': {'scala': '3.3.8'},
                'catalog': 'scala_algorithms', 'catalog_sha256': hashlib.sha256(catalog.read_bytes()).hexdigest()}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    (tmp_path / 'runs').mkdir()
    saved = record()
    saved['manifest'] = deepcopy(manifest)
    path = tmp_path / 'runs/scala_shortest_paths_train.json'
    path.write_text(json.dumps(saved))
    return manifest, saved, path


def test_summary_separates_repair_repetition_hidden_and_infrastructure():
    repaired = record()
    repaired['hidden'] = {'score': 0, 'passed_all': False}
    failed_job = {'task': {'task_id': 'infra'}, 'job': {'status': 'failed', 'error': 'compiler missing'}}
    result = review.summarize([repaired, failed_job], [
        {'system': 'You repair incorrect solutions', 'prompt_characters': 200, 'usage': {'prompt_eval_count': 80}},
        {'system': 'Generate solution', 'prompt_characters': 1000},
    ])
    assert result['completed_tasks'] == 1 and result['infrastructure_failures'] == 1
    assert result['first_passes'] == 0 and result['final_passes'] == 1
    assert result['repaired_tasks'] == result['repair_eligible_tasks'] == 1
    assert result['unchanged_candidates'] == 1
    assert result['hidden_evaluated_tasks'] == 1 and result['hidden_passes'] == 0
    assert result['max_repair_prompt_characters'] == 200
    assert result['max_repair_prompt_tokens'] == 80


@pytest.mark.parametrize('change', ['manifest', 'mode', 'code', 'catalog', 'toolchain', 'budget', 'fixtures'])
def test_mixed_provenance_is_rejected_before_hidden_execution(tmp_path, monkeypatch, change):
    manifest, saved, path = workspace(tmp_path)
    if change == 'manifest':
        saved['manifest']['seed'] = 99
    elif change == 'mode':
        saved['run']['execution_mode'] = 'http'
    elif change == 'code':
        saved['job']['result']['code'] = 'unselected code'
    elif change == 'toolchain':
        saved['run']['attempts'][0]['runtime_grade']['execution_environment']['scala'] = 'changed'
    elif change == 'budget':
        saved['run']['attempt_budget'] = 10
    elif change == 'fixtures':
        saved['task']['hidden_tests'] = [{'input': 'PRIVATE_HIDDEN_FIXTURE'}]
    else:
        manifest['catalog_sha256'] = 'changed'
        (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    path.write_text(json.dumps(saved))
    monkeypatch.setattr(review, 'preflight', lambda *_: pytest.fail('must reject before running tooling'))
    with pytest.raises(ValueError):
        review.review_workspace(tmp_path, evaluate_hidden=True)


def test_hidden_evaluation_uses_only_frozen_code_and_does_not_archive_diagnostics(tmp_path, monkeypatch):
    manifest, _, _ = workspace(tmp_path)
    monkeypatch.setattr(review, 'preflight', lambda *_: manifest['toolchain'])
    calls = []
    def verify(code, suite, **options):
        calls.append((code, options))
        assert code == 'fixed' and options['max_failures'] == 0
        return SimpleNamespace(pass_rate=0, passed=0, total=2, secret='HIDDEN_DIAGNOSTIC')
    monkeypatch.setattr(review, 'verify', verify)
    monkeypatch.setattr(review, 'grade_runtime_robustness', lambda *_: SimpleNamespace(
        critical_failure=False, compilation_seconds=1, total_duration_seconds=0.2, execution_environment=manifest['toolchain']))
    result = review.review_workspace(tmp_path, evaluate_hidden=True)
    assert result['hidden_passes'] == 0 and result['hidden_evaluated_tasks'] == 1
    assert len(result['missing_tasks']) == result['expected_tasks'] - 1
    review.review_workspace(tmp_path, evaluate_hidden=True)
    assert len(calls) == 1  # cache reuses frozen code without compiling again
    for path in tmp_path.rglob('*'):
        if path.is_file():
            assert 'HIDDEN_DIAGNOSTIC' not in path.read_text()


def test_hidden_toolchain_change_is_rejected(tmp_path, monkeypatch):
    workspace(tmp_path)
    monkeypatch.setattr(review, 'preflight', lambda *_: {'scala': 'changed'})
    with pytest.raises(ValueError, match='Toolchain changed'):
        review.review_workspace(tmp_path, evaluate_hidden=True)


def test_stale_hidden_cache_is_not_counted(tmp_path):
    workspace(tmp_path)
    (tmp_path / 'runs/scala_shortest_paths_train.hidden').write_text(json.dumps(
        {'identity': 'different code', 'passed_all': True, 'score': 1}))
    result = review.review_workspace(tmp_path)
    assert result['hidden_evaluated_tasks'] == 0


def test_hidden_infrastructure_error_is_separate_from_solution_failure(tmp_path, monkeypatch):
    manifest, _, _ = workspace(tmp_path)
    monkeypatch.setattr(review, 'preflight', lambda *_: manifest['toolchain'])
    def broken(*args, **kwargs):
        raise RuntimeError('secret harness diagnostic')
    monkeypatch.setattr(review, 'verify', broken)
    result = review.review_workspace(tmp_path, evaluate_hidden=True)
    assert result['hidden_evaluated_tasks'] == 0 and result['hidden_infrastructure_failures'] == 1
    assert 'secret harness diagnostic' not in (tmp_path / 'summary.json').read_text()


def test_request_log_tolerates_only_an_incomplete_trailing_entry(tmp_path):
    path = tmp_path / 'requests.jsonl'
    path.write_text('{"usage": {}}\n{"partial"')
    assert review._read_requests(path) == [{'usage': {}}]
    path.write_text('{broken}\n{"usage": {}}\n')
    with pytest.raises(json.JSONDecodeError):
        review._read_requests(path)
