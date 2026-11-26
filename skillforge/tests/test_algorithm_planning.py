import json

import pytest

from src.agents.llm_client import CallCountingClient, LLMClient, RecordingClient
from src.agents.test_generator import TestCase as Case, TestSuite as Suite
from src.history import ExperimentStore
from src.reinforcement_loop import run_candidate_refinement_loop
from src.benchmark.runner import benchmark_manifest
from src.benchmark.models import BenchmarkDataset, BenchmarkTask
from src.evolution.prompt_registry import load_default_registry


class PlanningClient:
    model = 'test-qwen:7b'

    def __init__(self):
        self.requests = []

    def generate(self, prompt, system='', temperature=0.2, **kwargs):
        self.requests.append(dict(prompt=prompt, system=system, temperature=temperature, **kwargs))
        if 'problem analyzer' in system.lower():
            return json.dumps(dict(problem_type='math', constraints='', selected_max_attempts=3,
                                   attempt_budget_reason='Three candidates.', examples=[]))
        if 'You design an algorithm' in system:
            return 'Parse two integer tokens. An initial design still needs review.'
        if 'You review and correct a proposed algorithm' in system:
            return 'Read two integers, add them, and output the sum. O(1) time and memory.'
        if 'You diagnose a proven failure' in system:
            assert '"test_passed": false' in prompt
            assert '"execution_status": "pass"' in prompt
            return 'Replace the constant result with the sum of both parsed integers.'
        if 'You repair incorrect' in system:
            assert 'Revised algorithm plan' in prompt
            assert '"failure_kind": "wrong_output"' in prompt
            assert 'Evaluation report:' not in prompt
            return 'import sys; a,b=map(int,sys.stdin.read().split()); print(a+b)'
        assert 'Reviewed algorithm plan' in prompt
        return 'print(0)'


def test_planning_and_replanning_are_counted_audited_and_do_not_consume_candidates(tmp_path):
    model = PlanningClient()
    with ExperimentStore(tmp_path / 'history.db') as store:
        result = run_candidate_refinement_loop(model, 'Read two integer tokens and print their sum.',
                    config_name='full', max_attempts=3, seed=42, reasoning_mode='plan_critique',
                    trusted_test_suite=Suite([Case('2 3', '5', 'normal'), Case('4 6', '10', 'normal')]),
                    history_store=store)
        logs = store._connection.execute('SELECT stage,request_json FROM model_requests ORDER BY call_id').fetchall()
        run = store.get_run(result.history_run_id)
    assert result.success and len(result.attempts) == 2
    assert result.model_calls == 6
    assert result.reasoning_mode == run['reasoning_mode'] == 'plan_critique'
    assert [row['stage'] for row in logs] == [
        'analysis', 'algorithm_plan:1', 'algorithm_review:1', 'candidate:1',
        'algorithm_repair:2', 'candidate:2',
    ]
    requests = [json.loads(row['request_json']) for row in logs]
    assert len({request['seed'] for request in requests}) == 6
    assert requests[1]['max_output_tokens'] == 1500
    assert requests[3]['max_output_tokens'] == 4096
    assert result.best_attempt.candidate.algorithm_plan.startswith('Replace the constant')
    for request in requests[1:4]:
        assert '2 3' not in request['prompt'] and '4 6' not in request['prompt']


def test_planning_generation_and_repairs_use_the_same_model_selected_io_formats():
    input_format = 'Two whitespace-separated integer tokens'
    output_format = 'One decimal integer followed by a newline'
    class FormatClient(PlanningClient):
        def generate(self, prompt, system='', temperature=0.2, **kwargs):
            raw = super().generate(prompt, system=system, temperature=temperature, **kwargs)
            if 'problem analyzer' in system.lower():
                return json.dumps(dict(json.loads(raw), input_format=input_format, output_format=output_format))
            return raw
    client = FormatClient()
    result = run_candidate_refinement_loop(client, 'Sum the two supplied integers.',
        reasoning_mode='plan_critique', config_name='full', max_attempts=2,
        trusted_test_suite=Suite([Case('2 3', '5', 'normal')]))
    assert result.success
    for request in client.requests[1:]:
        assert input_format in request['prompt'] and output_format in request['prompt']


def test_invalid_reasoning_mode_rejects_before_model_calls():
    client = PlanningClient()
    with pytest.raises(ValueError, match='reasoning_mode'):
        run_candidate_refinement_loop(client, 'Read and sum two integers.', reasoning_mode='unknown')
    assert client.requests == []


def test_scalar_repair_does_not_replan_using_failed_examples():
    class ScalarClient(PlanningClient):
        def generate(self, prompt, system='', temperature=0.2, **kwargs):
            if 'using a test pass count' in system:
                self.requests.append(dict(prompt=prompt, system=system, **kwargs))
                assert 'Failures:' not in prompt and 'One proven failure' not in prompt
                assert '2 3' not in prompt
                return 'import sys; print(sum(map(int,sys.stdin.read().split())))'
            return super().generate(prompt, system, temperature, **kwargs)
    client = ScalarClient()
    result = run_candidate_refinement_loop(client, 'Read two integer tokens and print their sum.',
                 reasoning_mode='plan_critique', config_name='scalar_repair', max_attempts=2,
                 trusted_test_suite=Suite([Case('2 3', '5', 'normal')]))
    assert result.success and result.model_calls == 5
    assert not any('You diagnose a proven failure' in request['system'] for request in client.requests)


def test_planned_repair_receives_actionable_judge_review_without_inventing_a_failure():
    from tests.test_reinforcement_loop import AdvisoryJudgeClient

    class ReviewPlanningClient(PlanningClient):
        def generate(self, prompt, system='', temperature=0.2, **kwargs):
            if 'You repair incorrect' in system:
                self.requests.append(dict(prompt=prompt, system=system, **kwargs))
                assert 'judge_advisory' in prompt
                assert 'A possible boundary case requires review' in prompt
                assert 'Review edge-case handling.' in prompt
                assert '"test_passed": false' not in prompt
                return 'print(0)\n# Reviewed the boundary case.'
            return super().generate(prompt, system, temperature, **kwargs)

    client = ReviewPlanningClient()
    judge = AdvisoryJudgeClient([
        AdvisoryJudgeClient.response('EDGE_CASE', issues=True),
        AdvisoryJudgeClient.response(),
    ])
    result = run_candidate_refinement_loop(
        client, 'Read two integer tokens and print zero.', reasoning_mode='plan_critique',
        config_name='full', max_attempts=2, judge_client=judge,
        trusted_test_suite=Suite([Case('2 3', '0', 'normal')]),
    )

    assert result.success and len(result.attempts) == 2
    assert result.attempts[0].evaluation_report.hard_gate_passed
    assert result.attempts[0].evaluation_report.status.value == 'review'
    assert not any('You diagnose a proven failure' in request['system'] for request in client.requests)


def test_planned_repair_retains_constraint_evidence_when_revised_plan_omits_it():
    from tests.test_reinforcement_loop import ConstraintRepairClient

    class ConstraintPlanningClient(ConstraintRepairClient):
        def generate(self, prompt, system=None, temperature=0.2, **kwargs):
            if 'You design an algorithm' in (system or ''):
                return 'Read integer tokens and put them in order.'
            if 'You review and correct a proposed algorithm' in (system or ''):
                return 'Read integer tokens and put them in order.'
            if 'You diagnose a proven failure' in (system or ''):
                assert 'constraint_violations' in prompt
                # The repair must retain deterministic evidence independently
                # of the model's fallible summary of that evidence.
                return 'Replace the rejected operation with a valid algorithm.'
            if 'You repair incorrect' in (system or ''):
                assert 'constraint_violations' in prompt
                assert '"test_passed": false' not in prompt
            return super().generate(prompt, system=system, temperature=temperature)

    client = ConstraintPlanningClient()
    result = run_candidate_refinement_loop(
        client, 'Sort integers without sort().', reasoning_mode='plan_critique',
        config_name='full', max_attempts=2,
        trusted_test_suite=Suite([Case('3 1 2', '1 2 3', 'normal')]),
    )

    assert result.success and len(result.attempts) == 2
    assert result.attempts[0].verification.passed == 1
    assert result.attempts[0].evaluation_report.constraint_grade.violations


def test_benchmark_manifest_distinguishes_planning_without_changing_direct_default():
    suite = Suite([Case('2 3', '5', 'normal')])
    task = BenchmarkTask('sum', 'Sum', 'Read and sum two integers.', 'math', 'easy', 'train', suite, suite)
    dataset = BenchmarkDataset((task,))
    prompt = load_default_registry().active_prompt
    common = dict(client=PlanningClient(), dataset=dataset, selected_tasks=[task], selected_prompt=prompt)
    direct = benchmark_manifest(**common)
    planned = benchmark_manifest(**common, reasoning_mode='plan_critique')
    assert 'reasoning_mode' not in direct
    assert planned['reasoning_mode'] == 'plan_critique'
    assert {k:v for k,v in planned.items() if k != 'reasoning_mode'} == direct
