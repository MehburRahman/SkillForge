import json
from types import SimpleNamespace

import pytest

from tools import compare_reasoning
from tests.test_prompt_evolver import registry


@pytest.fixture
def comparison(monkeypatch, tmp_path):
    statement = tmp_path / 'problem.txt'
    statement.write_text('Read an integer and print it.')
    visible = tmp_path / 'visible.json'
    hidden = tmp_path / 'hidden.json'
    visible.write_text(json.dumps([{'input': '1', 'expected_output': '1', 'category': 'normal'}]))
    hidden.write_text(json.dumps([{'input': '937', 'expected_output': '937', 'category': 'edge'}]))
    output = tmp_path / 'comparison'
    identity = {'name': 'qwen2.5-coder:7b', 'digest': 'fixed-weights'}
    monkeypatch.setattr(compare_reasoning, '_model_identity', lambda model: identity)
    monkeypatch.setattr(compare_reasoning, 'preflight_execution', lambda language: {'version': 'fixture'})
    prompts = registry(tmp_path)
    monkeypatch.setattr(compare_reasoning, 'load_default_registry', lambda: prompts)
    arguments = ['--problem-file', str(statement), '--tests', str(visible), '--hidden-tests', str(hidden),
                 '--output-dir', str(output), '--language', 'python', '--max-attempts', '3', '--seed', '42']
    return SimpleNamespace(arguments=arguments, output=output, visible=visible, hidden=hidden, identity=identity)


def _result(mode):
    verification = SimpleNamespace(passed=1)
    best = SimpleNamespace(verification=verification,
                           evaluation_report=SimpleNamespace(status=SimpleNamespace(value='pass')))
    return SimpleNamespace(best_code=f'# {mode}\nprint(input())', history_run_id=mode,
                           attempts=[best], best_attempt=best, model_calls=1, wall_clock_seconds=2.5)


def _record_call(options):
    mode = options['reasoning_mode']
    owner = f'owner-{mode}'
    options['on_stage']('request_logging', {'request_log_id': owner})
    options['history_store'].record_model_request(owner, 'generator', 'candidate:1',
        {'prompt': 'Read an integer and print it.'}, 'print(input())', None,
        {'prompt_eval_count': 15, 'eval_count': 4}, 0.5)


@pytest.mark.parametrize('first', ['direct', 'plan_critique'])
def test_same_settings_and_only_final_source_gets_withheld_evaluation(comparison, monkeypatch, first):
    calls, hidden_calls = [], []

    def solve(client, statement, **options):
        calls.append((client, statement, options))
        _record_call(options)
        return _result(options['reasoning_mode'])

    def score(code, suite, **options):
        hidden_calls.append((code, suite, options))
        return SimpleNamespace(passed=1, total=1)

    monkeypatch.setattr(compare_reasoning, 'run_candidate_refinement_loop', solve)
    monkeypatch.setattr(compare_reasoning, 'verify', score)
    assert compare_reasoning.main([*comparison.arguments, '--first-mode', first]) == 0
    assert [options['reasoning_mode'] for _, _, options in calls][0] == first
    assert len(calls) == len(hidden_calls) == 2
    for client, statement, options in calls:
        assert statement == 'Read an integer and print it.'
        assert [case.input for case in options['trusted_test_suite'].cases] == ['1']
        assert 'hidden_tests' not in options
        assert options['seed'] == 42 and options['max_attempts'] == 3 and options['config_name'] == 'full'
        assert client.model == 'qwen2.5-coder:7b'
        assert client.generation_options == {'num_ctx': 8192, 'num_predict': 4096}
    for (code, suite, options), (_, _, solve_options) in zip(hidden_calls, calls):
        assert code == _result(solve_options['reasoning_mode']).best_code
        assert [case.input for case in suite.cases] == ['937']
        assert options['max_failures'] == 0
    results = json.loads((comparison.output / 'results.json').read_text())
    assert all(r['prompt_tokens'] == 15 and r['output_tokens'] == 4 and r['model_seconds'] == 0.5 for r in results)
    assert all(r['usage_recorded_calls'] == r['recorded_model_calls'] == r['model_calls'] == 1 for r in results)
    assert all(r['hidden_seconds'] >= 0 and r['seconds'] == 2.5 for r in results)
    manifest = json.loads((comparison.output / 'manifest.json').read_text())
    assert manifest['model_identity'] == comparison.identity
    assert manifest['mode_order'][0] == first
    assert manifest['judge_enabled'] is False


def test_overlapping_suites_are_rejected_before_model_access_or_output(comparison, monkeypatch):
    comparison.hidden.write_text(comparison.visible.read_text())
    monkeypatch.setattr(compare_reasoning, '_model_identity', lambda model: pytest.fail('must not contact model'))
    with pytest.raises(SystemExit, match='2'):
        compare_reasoning.main(comparison.arguments)
    assert not comparison.output.exists()


def test_failed_arm_is_preserved_without_preventing_other_arm(comparison, monkeypatch):
    def solve(client, statement, **options):
        _record_call(options)
        if options['reasoning_mode'] == 'direct':
            raise RuntimeError('model request failed')
        return _result(options['reasoning_mode'])

    monkeypatch.setattr(compare_reasoning, 'run_candidate_refinement_loop', solve)
    monkeypatch.setattr(compare_reasoning, 'verify', lambda *args, **kwargs: SimpleNamespace(passed=1, total=1))
    assert compare_reasoning.main(comparison.arguments) == 1
    direct, planned = json.loads((comparison.output / 'results.json').read_text())
    assert direct['status'] == 'run_error' and direct['recorded_model_calls'] == 1
    assert 'model request failed' in direct['error']
    assert planned['status'] == 'pass' and planned['hidden_passed'] == 1
    assert not (comparison.output / 'direct.source').exists()
    assert (comparison.output / 'plan_critique.source').exists()


def test_replaced_weights_cannot_silently_change_comparison_arm(comparison, monkeypatch):
    identities = iter([comparison.identity, comparison.identity, {'digest': 'replacement'}])
    monkeypatch.setattr(compare_reasoning, '_model_identity', lambda model: next(identities))
    calls = []

    def solve(client, statement, **options):
        calls.append(options['reasoning_mode'])
        _record_call(options)
        return _result(options['reasoning_mode'])

    monkeypatch.setattr(compare_reasoning, 'run_candidate_refinement_loop', solve)
    monkeypatch.setattr(compare_reasoning, 'verify', lambda *args, **kwargs: SimpleNamespace(passed=1, total=1))
    assert compare_reasoning.main(comparison.arguments) == 1
    assert calls == ['direct']
    results = json.loads((comparison.output / 'results.json').read_text())
    assert 'weights changed' in results[1]['error']


@pytest.mark.parametrize('entry', [
    {'model': 'local:latest', 'digest': 'weights'},
    SimpleNamespace(model='local:latest', digest='weights'),
])
def test_model_tag_resolution_records_digest(monkeypatch, entry):
    monkeypatch.setattr(compare_reasoning, 'ollama', SimpleNamespace(list=lambda: {'models': [entry]}))
    assert compare_reasoning._model_identity('local') == {'name': 'local:latest', 'digest': 'weights'}
