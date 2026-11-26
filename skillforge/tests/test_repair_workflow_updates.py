import json

import pytest

from src.agents.llm_client import RecordingClient as AuditClient, CallCountingClient
from src.agents.metaprompt_agent import ProposalValidationError, PROPOSAL_RESPONSE_SCHEMA
from src.evaluation.llm_judge import JUDGE_JSON_TEMPLATE
from src.history import ExperimentStore
from src.reinforcement_loop import run_candidate_refinement_loop
from src.verifier.outcome_verifier import VerificationResult, TestCaseVerification as CaseResult
from src.verifier.runners.base import ExecutionResult, Status
from src.agents.test_generator import TestSuite as Suite, TestCase as Case
from tests.test_experiment_configs import RecordingClient
from tests.test_llm_judge import JudgeClient, call_judge, valid_response
from tests.test_metaprompt_agent import call_agent, valid_response as proposal_response
from tests.test_evolution_workflow import workflow, SeededMetaClient
from tests.test_prompt_evolver import add_runs
from src.evolution.workflow import run_evolution_round


def test_compile_repeats_keep_budget_reuse_compile_and_record_actual_requests(tmp_path, monkeypatch):
    diagnostic = '-- [E006] Not Found Error: Candidate.scala:1:15\n1 | continue\n  | Not found: continue\n'
    calls = []
    def verify(*args, **kwargs):
        calls.append(args)
        failure = dict(status='compile_error', stderr=diagnostic, input='', expected='ok', actual='')
        return VerificationResult(0, 1, 0, failure, [failure], [CaseResult('', 'ok', '', 'normal', False,
            ExecutionResult(Status.COMPILE_ERROR, '', diagnostic, 1, compilation_seconds=2.0))])
    monkeypatch.setattr('src.reinforcement_loop.preflight_execution', lambda *args: None)
    monkeypatch.setattr('src.reinforcement_loop.verify', verify)
    class Judge:
        model = 'another-family'
        def generate(self, *args, **kwargs):
            pytest.fail('A compiler failure must not invoke the semantic judge')
    # The third response violates the source-edit contract. Its rejected response
    # still consumes a fixed-budget candidate and reuses the compile failure.
    model = RecordingClient(['object Main { continue }'] * 3)
    with ExperimentStore(tmp_path / 'history.db') as store:
        result = run_candidate_refinement_loop(model, 'Print ok.', language='scala', config_name='full',
            max_attempts=3, judge_client=Judge(), history_store=store,
            trusted_test_suite=Suite([Case('', 'ok', 'normal')]))
        assert len(result.attempts) == 3 and len(calls) == 1
        assert result.model_calls == 4  # analysis plus three generated candidates
        assert [a.duplicate_of for a in result.attempts] == [None, 1, 1]
        assert [a.evaluation_report.runtime_grade.compilation_seconds for a in result.attempts] == [2, 0, 0]
        assert 'repeats attempt 1' in model.code_requests[-1]['prompt']
        assert 'Compiler diagnostics:' in model.code_requests[-1]['prompt']
        assert 'Evaluation report:' not in model.code_requests[-1]['prompt']
        assert 'LOGIC_ERROR' not in result.attempts[0].feedback.candidate_feedback
        assert 'Registered strategy:' in model.code_requests[-1]['system']
        assert 'Source edit response contract' in model.code_requests[-1]['prompt']
        assert 'source edit validation' in result.attempts[-1].feedback.candidate_feedback.lower()
        saved = store.get_attempts(result.history_run_id)
        assert saved[-1]['duplicate_of'] == saved[-1]['reused_from'] == 1
        logs = store._connection.execute('SELECT * FROM model_requests WHERE owner_id=? ORDER BY call_id', (result.request_log_id,)).fetchall()
        assert len(logs) == 4
        assert json.loads(logs[-1]['request_json'])['prompt'] == model.code_requests[-1]['prompt']
        assert logs[-1]['response_text'] == 'object Main { continue }'


def test_failed_model_call_is_audited_before_any_run_exists(tmp_path):
    class Broken:
        model = 'offline-model'
        generation_options = {'num_ctx': 8192}
        def generate(self, *args, **kwargs):
            raise RuntimeError('offline')
    with ExperimentStore(tmp_path / 'history.db') as store:
        client = CallCountingClient(AuditClient(Broken(), store, 'early-log', 'generator'), seed=42)
        client.set_stage('analysis')
        with pytest.raises(RuntimeError, match='offline'):
            client.generate('original request', system='instructions')
        row = store._connection.execute('SELECT * FROM model_requests').fetchone()
        assert row['stage'] == 'analysis' and 'offline' in row['error']
        request = json.loads(row['request_json'])
        assert request['options']['num_ctx'] == 8192 and isinstance(request['seed'], int)
        assert store._connection.execute('SELECT COUNT(*) FROM runs').fetchone()[0] == 0


def test_placeholder_and_contradictory_judge_responses_are_retried():
    client = JudgeClient([JUDGE_JSON_TEMPLATE, valid_response(likely_failure_category='NONE'), valid_response()])
    grade = call_judge(client)
    assert grade.likely_failure_category != 'NONE'
    assert len(client.requests) == 3
    assert 'copied placeholder' in client.requests[1]['prompt']
    assert 'contradicts' in client.requests[2]['prompt']
    assert 'Concise semantic assessment.' not in client.requests[0]['prompt']


def test_meta_agent_uses_schema_and_scopes_code_contract_to_proposed_prompt():
    class Structured:
        def generate(self, *args, **kwargs):
            pytest.fail('Use structured generation')
        def generate_json(self, prompt, **kwargs):
            assert kwargs['schema'] == PROPOSAL_RESPONSE_SCHEMA
            assert 'Your response to this request must be proposal JSON, not solution code.' in prompt
            return proposal_response()
    assert call_agent(Structured()).proposed_prompt


class InvalidMeta(SeededMetaClient):
    def generate(self, *args, **kwargs):
        self.requests.append(kwargs)
        return '{"wrong_field": "rejected"}'


def test_explicit_proposal_retry_uses_frozen_evidence_then_validates(tmp_path, workflow):
    with ExperimentStore(tmp_path / 'history.db') as store:
        add_runs(store)
        with pytest.raises(ProposalValidationError):
            run_evolution_round(store=store, **dict(workflow, meta_client=InvalidMeta()))
        failed = store.get_evolution_round('round-1')
        before = store.get_evolution_check(failed['trigger_id'])
        assert before['failure_kind'] == 'proposal_validation'
        assert store.evolution_evidence('P0')['training_runs'] == 0
        assert store._connection.execute('SELECT COUNT(*) FROM proposal_rejections').fetchone()[0] == 2
        result = run_evolution_round(store=store, resume=True, retry_proposal=True, **workflow)
        assert result['status'] == 'complete' and result['decision'] == 'promoted'
        after = store.get_evolution_check(failed['trigger_id'])
        assert before['evidence'] == after['evidence'] and after['proposal_retry_count'] == 1
        assert store.get_evolution_round('round-1')['meta_model_calls'] == 3


def test_proposal_retry_limit_prevents_unbounded_resampling(tmp_path, workflow):
    with ExperimentStore(tmp_path / 'history.db') as store:
        add_runs(store)
        arguments = dict(workflow, meta_client=InvalidMeta())
        with pytest.raises(ProposalValidationError):
            run_evolution_round(store=store, **arguments)
        with pytest.raises(ProposalValidationError):
            run_evolution_round(store=store, resume=True, retry_proposal=True, **arguments)
        calls = len(arguments['meta_client'].requests)
        with pytest.raises(ValueError, match='once'):
            run_evolution_round(store=store, resume=True, retry_proposal=True, **arguments)
        assert len(arguments['meta_client'].requests) == calls
