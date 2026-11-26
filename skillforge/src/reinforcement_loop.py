import re
from dataclasses import dataclass, field, replace
import time
from uuid import uuid4
from typing import Callable, List, Optional

from src.agents.code_generator import CodeCandidate, generate_code
from src.agents.algorithm_planner import plan_algorithm, revise_algorithm
from src.agents.llm_client import (
    CallCountingClient, RecordingClient, validate_judge_pair, DEFAULT_TEST_MODEL_OUTPUT_TOKENS, LLMClient,
)
from src.agents.refinement_agent import refine_code, _scala_parsing_diagnosis, _http_protocol_diagnosis
from src.agents.task_spec import TaskSpec, parse_task, execution_problem_statement
from src.agents.test_generator import TestSuite, generate_tests, validate_generation_tests, reference_cases
from src.evaluation.models import EvaluationReport, EvaluationStatus, EvaluationReason
from src.evaluation.pipeline import evaluate_candidate
from src.experiments import preflight_execution, DEFAULT_EXPERIMENT_ATTEMPTS, validate_experiment, validate_generation_execution, validate_reasoning_mode
from src.feedback.collector import collect_feedback
from src.feedback.repair_context import compact_repair_evidence
import json
from src.feedback.models import FeedbackBundle
from src.verifier.outcome_verifier import VerificationResult, verify
from src.verifier.sandbox import Status
from src.verifier.diagnostics import diagnose_candidate
from src.verifier.source_check import check_source
from src.agents.compiler_quick_fixes import (compiler_quick_fix, scala_quote_replacement_fix,
                                             scala_regex_brace_fix)
from src.evolution.prompt_registry import load_default_registry


@dataclass
class AttemptResult:
    attempt_number: int
    candidate: CodeCandidate
    verification: VerificationResult
    reward: float
    evaluation_report: Optional[EvaluationReport] = None
    feedback: Optional[FeedbackBundle] = None
    attempt_budget: Optional[int] = None
    attempt_budget_source: str = "unknown"
    attempt_budget_reason: str = ""
    duplicate_of: Optional[int] = None
    reused_from: Optional[int] = None


@dataclass
class CandidateRefinementResult:
    """Complete result of the bounded, per-task candidate repair loop."""

    task_spec: TaskSpec
    test_suite: TestSuite
    attempts: List[AttemptResult] = field(default_factory=list)
    success: bool = False
    best_attempt_number: int = 0
    stop_reason: str = "not_started"
    test_source: str = "generated"
    history_run_id: Optional[str] = None
    attempt_budget: Optional[int] = None
    attempt_budget_source: str = "unknown"
    attempt_budget_reason: str = ""
    # What this run cost: every generator and judge call it made, and how
    # long it took end to end. Candidate execution time lives in the runtime
    # grade and is a different measurement.
    model_calls: int = 0
    wall_clock_seconds: float = 0.0
    config_name: str = "full"
    seed: Optional[int] = None
    language: str = "python"
    task_spec_cache_hit: bool = False
    analysis_model_calls: int = 0
    analysis_wall_clock_seconds: float = 0.0
    execution_mode: str = "stdio"
    entry_point: Optional[str] = None
    request_log_id: Optional[str] = None
    reasoning_mode: str = 'direct'
    test_model: Optional[str] = None
    test_model_calls: int = 0
    draft_candidate: Optional[CodeCandidate] = None
    validation_error: Optional[str] = None
    diagnostics: List[dict] = field(default_factory=list)
    diagnostic_repairs: int = 0

    @property
    def stage_metrics(self):
        reports = [a.evaluation_report for a in self.attempts if a.evaluation_report]
        compiled = sum(not r.runtime_grade.compilation_error_count for r in reports)
        passed = sum(r.functional_grade.passed_all for r in reports)
        return dict(evaluated_candidates=len(reports), compiled_candidates=compiled,
                    diagnostic_candidates=len({d['candidate_number'] for d in self.diagnostics}),
                    diagnostic_compilation_failures=len({d['candidate_number'] for d in self.diagnostics
                        if 'compile_error' in d['execution_statuses']}),
                    compile_rate=compiled / len(reports) if reports else None,
                    all_tests_passed_candidates=passed,
                    candidate_test_pass_rates=[r.functional_grade.score for r in reports],
                    judge_disagreement_candidates=sum(r.functional_grade.passed_all and bool(r.review_reasons) for r in reports),
                    repaired_after_initial_failure=bool(reports and not reports[0].functional_grade.passed_all and passed),
                    test_source=self.test_source, test_validation=self.test_suite.validation,
                    diagnostic_repairs=self.diagnostic_repairs,
                    model_calls=self.model_calls, wall_clock_seconds=self.wall_clock_seconds)

    @property
    def best_attempt(self) -> Optional[AttemptResult]:
        if self.best_attempt_number == 0:
            return None
        return self.attempts[self.best_attempt_number - 1]

    @property
    def best_code(self) -> str:
        attempt = self.best_attempt
        return attempt.candidate.raw_code if attempt else self.draft_candidate.raw_code if self.draft_candidate else ""

    @property
    def best_reward(self) -> float:
        attempt = self.best_attempt
        return attempt.reward if attempt else 0.0



def _compile_error_count(output):
    """Count compiler diagnostics (Scala '-- [Ennn] ... Error:' headers, else 'N errors found')."""
    headers = re.findall(r'^-- (?:\[E\d+\] )?[A-Za-z ]*Error', output or '', re.MULTILINE)
    if headers:
        return len(headers)
    total = re.search(r'(\d+) errors? found', output or '')
    return int(total.group(1)) if total else 1

def _candidate_fingerprint(code: str) -> str:
    """Normalize irrelevant trailing whitespace when detecting repeated code."""
    return "\n".join(line.rstrip() for line in code.strip().splitlines())


def _attempt_rank(attempt: AttemptResult):
    """Rank hard status before deterministic and advisory secondary scores."""
    report = attempt.evaluation_report
    if report is None:
        return (0, attempt.reward, 0.0, 0.0, -1.0)

    status_rank = {
        EvaluationStatus.FAIL: 0,
        EvaluationStatus.REVIEW: 1,
        EvaluationStatus.PASS: 2,
    }[report.status]
    constraint = report.constraint_grade
    if constraint.score is not None:
        constraint_score = constraint.score
    elif not constraint.unverified_constraints and not constraint.critical_violation:
        constraint_score = 1.0
    else:
        constraint_score = -1.0
    judge_score = report.secondary_score
    return (
        status_rank,
        report.functional_grade.score,
        report.runtime_grade.score,
        constraint_score,
        judge_score if judge_score is not None else -1.0,
    )


def _review_has_actionable_feedback(report: EvaluationReport) -> bool:
    """Only judge-backed review concerns are actionable without human input."""
    return any(
        reason.source == "judge" and reason.code != "JUDGE_UNAVAILABLE"
        for reason in report.review_reasons
    )


def _resolve_attempt_budget(
    task_spec: TaskSpec, max_attempts: Optional[int], config_name=None,
):
    if config_name == "single_shot":
        return (1, "experiment_config", "single_shot evaluates exactly one candidate.")
    if max_attempts is not None:
        return (
            max_attempts,
            "manual_override",
            "An explicit max_attempts value overrode the task-analysis decision.",
        )
    if config_name is not None:
        return (
            DEFAULT_EXPERIMENT_ATTEMPTS,
            "experiment_config",
            "Experiment configurations use a fixed three-candidate budget.",
        )
    return (
        task_spec.selected_max_attempts,
        "task_analysis_llm",
        task_spec.attempt_budget_reason,
    )


def run_candidate_refinement_loop(
    client,
    problem_statement: str,
    max_attempts: Optional[int] = None,
    max_feedback_failures: int = 3,
    max_consecutive_repeats: int = 2,
    on_attempt: Optional[Callable[[AttemptResult], None]] = None,
    trusted_test_suite: Optional[TestSuite] = None,
    prompt_registry=None,
    prompt_id: Optional[str] = None,
    judge_client=None,
    judge_max_attempts: int = 3,
    judge_review_threshold: float = 0.5,
    judge_only_passing: bool = False,
    history_store=None,
    task_id: Optional[str] = None,
    benchmark_split: str = "adhoc",
    config_name: Optional[str] = None,
    seed: Optional[int] = None,
    language: Optional[str] = None,
    complete_history: bool = True,
    task_spec_cache=None,
    benchmark_run_id: Optional[str] = None,
    execution_mode: str = "stdio",
    entry_point: Optional[str] = None,
    on_stage: Optional[Callable[[str, dict], None]] = None,
    reasoning_mode: str = 'direct',
    test_client=None,
    test_generation_method='json',
    reference_test_suite=None,
    differential_cases=0,
) -> CandidateRefinementResult:
    """Generate, evaluate, and repair candidates for one programming task.

    This is SkillForge's inner loop. It improves only the current code
    candidate; it does not train model weights or evolve prompts. Completed
    training runs can trigger the saved rollback monitor for future runs.
    A supplied trusted suite bypasses LLM
    test generation entirely. Explicit experiment configurations fix the
    budget and count repeated candidates; omitted configurations retain the
    original dynamic budget and repeat-stop behavior. Benchmark callers
    defer history completion until hidden evaluation has finished.
    """
    mode, language = validate_experiment(config_name, seed, language)
    validate_reasoning_mode(reasoning_mode)
    language, execution_mode, entry_point = validate_generation_execution(language, execution_mode, entry_point)
    validate_generation_tests(trusted_test_suite, execution_mode)
    if type(differential_cases) is not int or not 0 <= differential_cases <= 1000:
        raise ValueError('differential_cases must be between 0 and 1000')
    if differential_cases and (trusted_test_suite is not None or execution_mode != 'stdio'
                                or test_generation_method != 'program'):
        raise ValueError('differential testing requires generated stdio program tests')
    if reference_test_suite is not None:
        if trusted_test_suite is not None or execution_mode != 'stdio':
            raise ValueError('reference tests require generated stdio tests')
        reference_test_suite = reference_cases(None, reference_test_suite)
    if test_generation_method not in ('json', 'program'):
        raise ValueError('test_generation_method must be json or program')
    preflight_execution(language, execution_mode)
    validate_judge_pair(client, judge_client)
    if not problem_statement.strip():
        raise ValueError("problem_statement cannot be empty")
    if max_attempts is not None and (
        isinstance(max_attempts, bool)
        or not isinstance(max_attempts, int)
        or max_attempts < 1
    ):
        raise ValueError("max_attempts must be at least 1")
    if max_feedback_failures < 1:
        raise ValueError("max_feedback_failures must be at least 1")
    if max_consecutive_repeats < 1:
        raise ValueError("max_consecutive_repeats must be at least 1")
    if judge_max_attempts < 1:
        raise ValueError("judge_max_attempts must be at least 1")

    def stage(name, **data):
        if on_stage is not None:
            on_stage(name, data)

    # Wrap before the first model call so task analysis is counted too, and
    # rebind the names so retries are counted and seeded at the same boundary.
    run_started = time.perf_counter()
    judge_provenance = validate_judge_pair(client, judge_client)
    request_log_id = str(uuid4()) if history_store is not None else None
    # Trusted/official tests never spend calls on an auxiliary test model.
    test_client = test_client if trusted_test_suite is None and test_client is not client else None
    if trusted_test_suite is None and test_client is None and isinstance(client, LLMClient):
        test_client = client.for_tests()
    if test_client is not None:
        test_client.generation_options = {
            'num_predict': DEFAULT_TEST_MODEL_OUTPUT_TOKENS, **getattr(test_client, 'generation_options', {}),
        }
    if history_store is not None:
        client = RecordingClient(client, history_store, request_log_id, "generator")
        if test_client is not None:
            test_client = RecordingClient(test_client, history_store, request_log_id, "test_generator")
        if judge_client is not None:
            judge_client = RecordingClient(judge_client, history_store, request_log_id, "judge")
    stage("request_logging", request_log_id=request_log_id, judge_provenance=judge_provenance)
    analysis_started = run_started
    cache_hit = False
    cached_analysis_calls = 0
    stage("analysis")
    if task_spec_cache is not None:
        task_spec, cache_hit, cached_analysis_calls = task_spec_cache.get_or_analyze(
            client, problem_statement, language,
            execution_mode=execution_mode, entry_point=entry_point,
        )
    task_key = task_id if task_id is not None else problem_statement.strip()
    client = CallCountingClient(client, seed=seed, task_key=task_key)
    if test_client is not None:
        test_client = CallCountingClient(test_client, seed=seed, task_key=task_key)
    if judge_client is not None:
        judge_client = CallCountingClient(judge_client, seed=seed, task_key=task_key)

    if task_spec_cache is None:
        client.set_stage("analysis")
        task_spec = parse_task(client, problem_statement, language=language,
                               execution_mode=execution_mode, entry_point=entry_point)
    resolved_statement = execution_problem_statement(task_spec)
    analysis_calls = cached_analysis_calls + client.call_count
    analysis_seconds = time.perf_counter() - analysis_started
    (
        attempt_budget,
        attempt_budget_source,
        attempt_budget_reason,
    ) = _resolve_attempt_budget(task_spec, max_attempts, config_name)
    test_model = None
    test_model_calls = 0
    algorithm_plan = None
    candidate = None
    generation_temperature = 0.8 if mode == "best_of_n" else 0.2
    diagnostics = []
    diagnostic_repairs = 0
    diagnostic_seen = set()
    diagnostic_repair_budget = 2

    def quick_fixed(code, compiler_output, *, require_pass=False, **where):
        """Apply deterministic compiler fixes the compiler confirms.

        A fix is kept when the source then compiles, or (unless require_pass)
        when it strictly reduces the compiler's error count, leaving fewer
        errors for the model. At most three fixes are chained.
        """
        current, output, errors, notes, compiles = code, compiler_output, _compile_error_count(compiler_output), [], False
        for _ in range(3):
            found = compiler_quick_fix(current, output, language=language)
            if found is None:
                break
            fixed, description = found
            check = check_source(fixed, language=language)
            if check.status == Status.PASS:
                current, compiles = fixed, True
                notes.append(description)
                break
            remaining = _compile_error_count(check.stderr or '')
            if check.status != Status.COMPILE_ERROR or not 0 < remaining < errors:
                break
            current, output, errors = fixed, check.stderr or '', remaining
            notes.append(description)
        if current == code or (require_pass and not compiles):
            return None
        stage('compiler_quick_fix', description=' '.join(notes), compiles=compiles, **where)
        return current

    def diagnostic_recovery(suite=None, *, trusted=False, phase='source_check'):
        nonlocal candidate, diagnostic_repairs
        best_candidate = candidate
        best_rank = None
        previous_failure = False
        while True:
            fingerprint = _candidate_fingerprint(candidate.raw_code)
            if (phase, fingerprint) in diagnostic_seen:
                candidate = best_candidate
                return
            diagnostic_seen.add((phase, fingerprint))
            number = diagnostic_repairs + 1
            stage('diagnostic_verification', candidate_number=number, phase=phase,
                  tests_verified=trusted)
            progress = lambda name, data: stage(name, candidate_number=number, phase=phase, **data)
            if trusted:
                evidence = verify(candidate.raw_code, suite, language=language, on_progress=progress)
                executions = evidence.execution_results
            else:
                evidence, executions = diagnose_candidate(candidate.raw_code, language=language,
                                                          suite=suite if execution_mode != 'http' else None,
                                                          on_progress=progress)
            statuses = [execution.status.value for execution in executions]
            compiler_failed = Status.COMPILE_ERROR.value in statuses
            rank = (not compiler_failed, -len(evidence.failures))
            if best_rank is None or rank >= best_rank:
                best_rank, best_candidate = rank, candidate
            diagnostics.append(dict(candidate_number=number, phase=phase, tests_verified=trusted,
                                    execution_statuses=statuses, failures=evidence.failures,
                                    source_valid=all(status == Status.PASS.value for status in statuses)))
            stage('diagnostic_result', **diagnostics[-1])
            if compiler_failed:
                fixed = quick_fixed(candidate.raw_code, '\n'.join(
                    failure.get('stderr') or '' for failure in evidence.failures),
                    candidate_number=number, phase=phase)
                if fixed is not None:
                    # A rename costs no model call or repair budget; re-check the fixed source.
                    candidate = replace(candidate, raw_code=fixed)
                    continue
            if not evidence.failures or diagnostic_repairs >= diagnostic_repair_budget:
                candidate = best_candidate
                return
            # Sampling remains independent: diagnostics cannot turn best_of_n
            # into sequential repair or exceed its draw budget.
            if mode == 'best_of_n':
                return
            diagnostic_repairs += 1
            stage('refinement', candidate_number=diagnostic_repairs + 1, phase=phase,
                  attempt_budget=attempt_budget)
            client.set_stage(f'candidate:{diagnostic_repairs + 1}')
            repaired = refine_code(client, task_spec, candidate, evidence,
                language=language, execution_mode=execution_mode, entry_point=entry_point,
                algorithm_plan=algorithm_plan,
                diagnostic_only=not trusted, prefer_source_edits=previous_failure,
                force_fresh_source=previous_failure or bool(candidate.repair_validation_error))
            if (phase, _candidate_fingerprint(repaired.raw_code)) in diagnostic_seen:
                stage('diagnostic_duplicate', candidate_number=diagnostic_repairs + 1,
                      reason='Unchanged repair; requesting fresh source within diagnostic budget.')
                if diagnostic_repairs >= diagnostic_repair_budget:
                    candidate = best_candidate
                    return
                diagnostic_repairs += 1
                client.set_stage(f'candidate:{diagnostic_repairs + 1}')
                repaired = refine_code(client, task_spec, candidate, evidence,
                    language=language, execution_mode=execution_mode, entry_point=entry_point,
                    algorithm_plan=algorithm_plan, diagnostic_only=not trusted,
                    force_fresh_source=True, temperature=0.4, repeated_from=number)
            candidate = repaired
            previous_failure = compiler_failed
            stage('draft_ready', code=candidate.raw_code, status='unverified',
                  prompt_id=candidate.prompt_version, model=candidate.model_name)

    def initial_candidate():
        nonlocal algorithm_plan
        if reasoning_mode == 'plan_critique':
            algorithm_plan = plan_algorithm(client, resolved_statement, language=language,
                                            execution_mode=execution_mode, entry_point=entry_point,
                                            on_progress=lambda name, data: stage(name, **data))
        stage("generation", attempt_number=1, attempt_budget=attempt_budget)
        client.set_stage("candidate:1")
        return generate_code(client, task_spec, prompt_registry=prompt_registry, prompt_id=prompt_id,
                             temperature=generation_temperature, language=language,
                             execution_mode=execution_mode, entry_point=entry_point,
                             algorithm_plan=algorithm_plan)

    def unverified_result(reason, suite=None):
        draft = CandidateRefinementResult(
            task_spec=task_spec, test_suite=suite or TestSuite(), draft_candidate=candidate,
            validation_error=str(reason), stop_reason='test_validation_failed', test_source='generated',
            attempt_budget=attempt_budget, attempt_budget_source=attempt_budget_source,
            attempt_budget_reason=attempt_budget_reason, config_name=mode, seed=seed,
            language=language, execution_mode=execution_mode, entry_point=entry_point,
            reasoning_mode=reasoning_mode, request_log_id=request_log_id, test_model=test_model,
            test_model_calls=test_model_calls, task_spec_cache_hit=cache_hit,
            analysis_model_calls=analysis_calls, analysis_wall_clock_seconds=analysis_seconds,
            model_calls=cached_analysis_calls + client.call_count
                + (test_client.call_count if test_client is not None else 0)
                + (judge_client.call_count if judge_client is not None else 0),
            wall_clock_seconds=time.perf_counter() - run_started,
            diagnostics=diagnostics, diagnostic_repairs=diagnostic_repairs,
        )
        stage('unverified', reason=draft.validation_error, request_log_id=request_log_id)
        return draft

    if trusted_test_suite is None:
        # A draft never receives a reward or enters training history. It survives
        # test failures in the API result/events and the persisted model audit.
        candidate = initial_candidate()
        stage('draft_ready', code=candidate.raw_code, status='unverified',
              prompt_id=candidate.prompt_version, model=candidate.model_name)
        anchors = reference_cases(task_spec, reference_test_suite) if execution_mode == 'stdio' else None
        diagnostic_recovery(anchors, trusted=anchors is not None,
                            phase='samples' if anchors is not None else 'source_check')
        tests_client = test_client if test_client is not None else client
        test_model = getattr(tests_client, 'model', None)
        stage("test_generation", test_model=test_model)
        tests_client.set_stage("tests")
        calls_before_tests = tests_client.call_count
        try:
            test_suite = generate_tests(tests_client, task_spec, method=test_generation_method,
                                        **(dict(execution_mode='http') if execution_mode == 'http' else {}),
                                        reference_test_suite=reference_test_suite,
                                        **(dict(differential_cases=differential_cases, seed=seed) if differential_cases else {}),
                                        on_progress=lambda name, data: stage(name, **data))
        except (ValueError, RuntimeError, OSError) as exc:
            test_model_calls = tests_client.call_count - calls_before_tests
            rejected_suite = getattr(exc, 'suite', None)
            if anchors is not None:
                test_suite = anchors
                test_suite.validation = dict(status='trusted_samples_only', generated_tests_error=str(exc))
                stage('trusted_test_fallback', test_count=len(anchors.cases), reason=str(exc))
            else:
                if rejected_suite is not None and rejected_suite.cases:
                    diagnostic_recovery(rejected_suite, phase='unverified_tests')
                return unverified_result(exc, rejected_suite)
        test_model_calls = tests_client.call_count - calls_before_tests
        if test_suite.validation.get('status') == 'needs_review':
            if anchors is not None:
                test_suite = anchors
                test_suite.validation = dict(status='trusted_samples_only',
                    generated_tests_error='Generated test answers remain unverified.')
                stage('trusted_test_fallback', test_count=len(anchors.cases), reason='Generated answers remain unverified.')
            else:
                diagnostic_recovery(test_suite, phase='unverified_tests')
                return unverified_result('Reference answers need independently verified cases before grading the draft.', test_suite)
        test_source = "trusted" if test_suite.validation.get('status') == 'trusted_samples_only' else "generated"
    else:
        test_suite = trusted_test_suite
        test_source = "trusted"
    if not test_suite.cases:
        raise ValueError("the test suite is empty")

    if execution_mode=='http' and test_suite.validation.get('requirements'):
        stage('requirement_plan',requirements=test_suite.validation['requirements'],
              endpoints=test_suite.validation.get('endpoint_inventory',[]))
    stage("tests_ready", test_count=len(test_suite.cases), test_source=test_source)
    if candidate is None:
        candidate = initial_candidate()
    result = CandidateRefinementResult(
        task_spec=task_spec,
        test_suite=test_suite,
        test_source=test_source,
        attempt_budget=attempt_budget,
        attempt_budget_source=attempt_budget_source,
        attempt_budget_reason=attempt_budget_reason,
        config_name=mode,
        seed=seed,
        language=language,
        task_spec_cache_hit=cache_hit,
        analysis_model_calls=analysis_calls,
        analysis_wall_clock_seconds=analysis_seconds,
        execution_mode=execution_mode,
        entry_point=entry_point,
        request_log_id=request_log_id,
        reasoning_mode=reasoning_mode,
        test_model=test_model,
        test_model_calls=test_model_calls,
        diagnostics=diagnostics, diagnostic_repairs=diagnostic_repairs,
    )
    if history_store is not None:
        result.history_run_id = history_store.start_run(
            problem_statement=problem_statement,
            prompt_version=candidate.prompt_version,
            model_name=candidate.model_name,
            test_source=test_source,
            judge_enabled=judge_client is not None,
            judge_model=getattr(judge_client, "model", None) if judge_client else None,
            task_id=task_id,
            benchmark_split=benchmark_split,
            config_name=mode,
            seed=seed,
            language=language,
            attempt_budget=attempt_budget,
            attempt_budget_source=attempt_budget_source,
            attempt_budget_reason=attempt_budget_reason,
            benchmark_run_id=benchmark_run_id,
            execution_mode=execution_mode,
            entry_point=entry_point,
            reasoning_mode=reasoning_mode,
            test_model=test_model,
        )
    if history_store is not None:
        history_store.link_request_log(result.history_run_id, request_log_id)
    stage("run_started", history_run_id=result.history_run_id)
    best_rank = None
    seen_candidates = {_candidate_fingerprint(candidate.raw_code): 1}
    evaluated_candidates = {}
    strategy_registry = prompt_registry or load_default_registry()
    strategy = strategy_registry.get(candidate.prompt_version).prompt_text
    repeated_count = 0
    compiler_failure_streak = 0

    remaining_attempts = attempt_budget
    gap_rounds = 0
    for attempt_number in range(1, remaining_attempts + 1):
        if language == "scala":
            # Java rejects literal '{' in a regex only at run time, often inside
            # a server thread that never reports it; escape it before testing.
            braces = scala_regex_brace_fix(candidate.raw_code)
            if braces is not None:
                candidate = replace(candidate, raw_code=braces[0])
                stage('compiler_quick_fix', attempt_number=attempt_number, compiles=None,
                      description=f'Escaped literal braces in {braces[1]} regular expression(s); '
                                  'Java reads an unescaped { as a repetition count and throws at run time.')
            quoted = scala_quote_replacement_fix(candidate.raw_code)
            if quoted is not None:
                candidate = replace(candidate, raw_code=quoted[0])
                stage('compiler_quick_fix', attempt_number=attempt_number, compiles=None,
                      description=f'Quoted the result of {quoted[1]} Regex.replaceAllIn function(s); '
                                  'replaceAllIn reads $ and \\ in inserted text as group references.')
        fingerprint = _candidate_fingerprint(candidate.raw_code)
        previous = evaluated_candidates.get(fingerprint)
        duplicate_of = previous.attempt_number if previous is not None else None
        reused_from = None
        stage("verification", attempt_number=attempt_number)
        if judge_client is not None:
            judge_client.set_stage(f"visible_judge:{attempt_number}")
        can_reuse = (previous is not None and previous.candidate.raw_code == candidate.raw_code
                     and previous.verification.case_results
                     and all(r.execution.status == Status.COMPILE_ERROR for r in previous.verification.case_results))
        if can_reuse:
            reused_from = duplicate_of
            verification = replace(previous.verification, case_results=[
                replace(case, execution=replace(case.execution, duration_seconds=0.0, compilation_seconds=0.0))
                for case in previous.verification.case_results])
            stage("repeated_candidate", attempt_number=attempt_number, duplicate_of=duplicate_of)
        else:
            verification = verify(
                candidate.raw_code,
                test_suite,
                max_failures=max_feedback_failures,
                language=language, execution_mode=execution_mode, entry_point=entry_point,
                **({"on_progress": lambda name, data: stage(name, attempt_number=attempt_number, **data)}
                   if language == "scala" and on_stage is not None else {}),
            )
        compiler_failed = any(result.status == Status.COMPILE_ERROR
                              for result in verification.execution_results)
        if compiler_failed and not can_reuse:
            fixed = quick_fixed(candidate.raw_code, '\n'.join(
                result.stderr or '' for result in verification.execution_results
                if result.status == Status.COMPILE_ERROR)[:20000], attempt_number=attempt_number)
            if fixed is not None:
                candidate = replace(candidate, raw_code=fixed)
                fingerprint = _candidate_fingerprint(candidate.raw_code)
                verification = verify(
                    candidate.raw_code, test_suite, max_failures=max_feedback_failures,
                    language=language, execution_mode=execution_mode, entry_point=entry_point,
                    **({"on_progress": lambda name, data: stage(name, attempt_number=attempt_number, **data)}
                       if language == "scala" and on_stage is not None else {}),
                )
                compiler_failed = any(result.status == Status.COMPILE_ERROR
                                      for result in verification.execution_results)
        compiler_failure_streak = compiler_failure_streak + 1 if compiler_failed else 0
        stage("evaluation", attempt_number=attempt_number)
        evaluation_report, feedback = evaluate_candidate(
            task_spec,
            candidate.raw_code,
            verification,
            test_source=test_source,
            judge_client=judge_client,
            judge_max_attempts=judge_max_attempts,
            judge_review_threshold=judge_review_threshold,
            judge_only_passing=judge_only_passing,
            language=language, execution_mode=execution_mode, entry_point=entry_point,
            **({'test_suite':test_suite,'generator_model':getattr(client,'model',None)} if execution_mode=='http' else {}),
        )
        if execution_mode=='http':
            stage('requirement_evidence',ledger=evaluation_report.constraint_grade.requirement_ledger)
        while (execution_mode=='http' and test_source=='generated'
               and evaluation_report.functional_grade.passed_all and gap_rounds<2):
            ledger=evaluation_report.constraint_grade.requirement_ledger
            port_stated=any(r['capability']=='bind_port' for r in ledger.get('requirements',[]))
            gaps=[r for r in ledger.get('requirements',[]) if r['state']=='unverified'
                  and r['capability']!='unclassified'
                  and (r['capability']!='clean_environment' or port_stated)]
            if not gaps:break
            gap_rounds+=1
            test_suite.validation['gap_rounds']=gap_rounds
            stage('requirement_gaps',round=gap_rounds,requirement_ids=[r['id'] for r in gaps])
            tests_client=test_client if test_client is not None else client
            before=tests_client.call_count
            try:
                from src.agents.http_test_generator import generate_http_tests
                extra=generate_http_tests(tests_client,resolved_statement,max_attempts=1,
                    target_requirements=gaps,atomic_requirements=task_spec.requirements,
                    reference_source=test_suite.validation.get('reference_source'),
                    on_progress=lambda name,data:stage(name,**data))
                identities={case.input for case in test_suite.cases}
                additions=[case for case in extra.cases if case.input not in identities]
                if not additions:
                    stage('requirement_gaps_exhausted',reason='Generated tests added no new inputs',
                          requirement_ids=[r['id'] for r in gaps]);break
                first_new_test=len(test_suite.cases)+1
                test_suite.cases.extend(additions)
                test_suite.validation.setdefault('gap_suites',[]).append(dict(
                    test_ids=[f'T{i}' for i in range(first_new_test,len(test_suite.cases)+1)],
                    targeted_requirement_ids=[r['id'] for r in gaps],validation=extra.validation))
                evaluated_candidates.clear()
                verification=verify(candidate.raw_code,test_suite,max_failures=max_feedback_failures,
                    language=language,execution_mode=execution_mode,entry_point=entry_point)
                evaluation_report,feedback=evaluate_candidate(task_spec,candidate.raw_code,verification,
                    test_source=test_source,judge_client=judge_client,judge_max_attempts=judge_max_attempts,
                    judge_review_threshold=judge_review_threshold,language=language,
                    judge_only_passing=judge_only_passing,
                    execution_mode=execution_mode,entry_point=entry_point,test_suite=test_suite,
                    generator_model=getattr(client,'model',None))
            except (ValueError,RuntimeError,OSError) as exc:
                from src.verifier.runners.http import HTTPInfrastructureError
                if isinstance(exc,HTTPInfrastructureError) or str(exc).startswith('Infrastructure error:'):
                    raise
                stage('requirement_gaps_exhausted',reason=str(exc),requirement_ids=[r['id'] for r in gaps]);break
            finally:
                test_model_calls+=tests_client.call_count-before
            test_suite.validation['gap_rounds']=gap_rounds
            test_suite.validation['request_count']=sum(len(case.requests) for case in test_suite.cases)
            test_suite.validation['scenario_count']=len(test_suite.cases)
        if execution_mode=='http':
            stage('requirement_evidence',ledger=evaluation_report.constraint_grade.requirement_ledger)
        if test_suite.validation.get('coverage_complete') is False and not evaluation_report.constraint_grade.requirement_ledger:
            evaluation_report.review_reasons.append(EvaluationReason(
                code='INCOMPLETE_TEST_COVERAGE', source='test_coverage',
                message=test_suite.validation['coverage_warning']))
            if evaluation_report.status == EvaluationStatus.PASS:
                evaluation_report.status = EvaluationStatus.REVIEW
            feedback = collect_feedback(evaluation_report, test_source=test_source)
        attempt = AttemptResult(
            attempt_number=attempt_number,
            candidate=candidate,
            verification=verification,
            reward=verification.pass_rate,
            evaluation_report=evaluation_report,
            feedback=feedback,
            attempt_budget=attempt_budget,
            attempt_budget_source=attempt_budget_source,
            attempt_budget_reason=attempt_budget_reason,
            duplicate_of=duplicate_of,
            reused_from=reused_from,
        )
        if duplicate_of is not None:
            attempt.feedback = collect_feedback(evaluation_report, test_source=test_source, stop_reason="repeated_candidate")
            feedback = attempt.feedback
        if candidate.repair_validation_error:
            feedback.candidate_feedback += "\nSource edit validation: " + candidate.repair_validation_error
        evaluated_candidates.setdefault(fingerprint, attempt)
        result.attempts.append(attempt)

        attempt_rank = _attempt_rank(attempt)
        if best_rank is None or attempt_rank > best_rank:
            best_rank = attempt_rank
            result.best_attempt_number = attempt_number

        if on_attempt:
            on_attempt(attempt)
        if history_store is not None:
            history_store.record_attempt(result.history_run_id, attempt)

        if mode == "best_of_n":
            # Every independent draw consumes one sample, including duplicates
            # and draws after a PASS. Hidden tests never participate in selection.
            result.success = (
                result.success or evaluation_report.status == EvaluationStatus.PASS
            )
            result.stop_reason = "sample_budget"
            if attempt_number < remaining_attempts:
                stage("generation", attempt_number=attempt_number + 1, attempt_budget=attempt_budget)
                client.set_stage(f"candidate:{attempt_number + 1}")
                candidate = generate_code(
                    client, task_spec, temperature=generation_temperature,
                    prompt_registry=prompt_registry, prompt_id=candidate.prompt_version,
                    language=language, execution_mode=execution_mode, entry_point=entry_point,
                    algorithm_plan=algorithm_plan,
                )
            continue

        if evaluation_report.status == EvaluationStatus.PASS:
            result.success = True
            result.stop_reason = "passed"
            break
        if (
            evaluation_report.status == EvaluationStatus.REVIEW
            and (not _review_has_actionable_feedback(evaluation_report)
                 # Interactive solves: every test passed and only the advisory
                 # judge objects. Report REVIEW with its concerns rather than
                 # rewriting code that the executed evidence says is correct.
                 or (judge_only_passing and evaluation_report.hard_gate_passed
                     and evaluation_report.functional_grade.passed_all))
        ):
            result.stop_reason = "review_required"
            break

        if attempt_number < remaining_attempts:
            stage("refinement", attempt_number=attempt_number + 1, attempt_budget=attempt_budget)
            parsing_failure = _scala_parsing_diagnosis(
                verification.failures, language, execution_mode, candidate.raw_code,
            ) or _http_protocol_diagnosis(verification.failures, language, execution_mode, candidate.raw_code)
            if (algorithm_plan is not None and not compiler_failed and mode != 'scalar_repair'
                    and not evaluation_report.hard_gate_passed and not parsing_failure):
                evidence = compact_repair_evidence(verification.failures,
                                                   first_failure=verification.first_failure,
                                                   runtime_failures=evaluation_report.runtime_grade.failure_evidence)
                if execution_mode == 'http' and test_suite.validation.get('reference_source'):
                    evidence['model_generated_reference'] = test_suite.validation['reference_source'][:12000]
                    evidence['reference_is_independent_proof'] = False
                if evaluation_report.constraint_grade.violations:
                    evidence['constraint_violations'] = evaluation_report.constraint_grade.violations[:3]
                algorithm_plan = revise_algorithm(client, resolved_statement, algorithm_plan,
                                                   candidate.raw_code, json.dumps(evidence, ensure_ascii=False),
                                                   attempt_number=attempt_number + 1,
                                                   on_progress=lambda name, data: stage(name, **data))
            client.set_stage(f"candidate:{attempt_number + 1}")
            while True:
                refined_candidate = refine_code(
                    client,
                    task_spec,
                    candidate,
                    verification,
                    evaluation_report=evaluation_report,
                    feedback_bundle=feedback,
                    feedback_mode=("scalar_repair" if mode == "scalar_repair" else "full"),
                    language=language, execution_mode=execution_mode, entry_point=entry_point,
                    strategy=strategy, repeated_from=duplicate_of,
                    prefer_source_edits=compiler_failure_streak >= 2,
                    algorithm_plan=algorithm_plan,
                    reference_program=test_suite.validation.get('reference_source'),
                    force_fresh_source=(bool(candidate.repair_validation_error)
                                        or (bool(parsing_failure)
                                            and getattr(client, 'source_edit_style', None) == 'lines')
                                        or (compiler_failed and attempt_number > 1
                                            and getattr(client, 'source_edit_style', None) == 'lines')),
                )
                if (config_name is None and not compiler_failed
                        and refined_candidate.raw_code != candidate.raw_code):
                    # A partial patch must not turn compiling code into a build
                    # failure and waste a graded attempt. Try the deterministic
                    # quick fix, then one complete regeneration. Interactive
                    # solves only: ablation arms keep their fixed call budget.
                    build = check_source(refined_candidate.raw_code, language=language)
                    if build.status == Status.COMPILE_ERROR:
                        fixed = quick_fixed(refined_candidate.raw_code, build.stderr or '',
                                            require_pass=True, attempt_number=attempt_number + 1)
                        if fixed is not None:
                            refined_candidate = replace(refined_candidate, raw_code=fixed)
                        else:
                            stage("repair_broke_build", attempt_number=attempt_number + 1,
                                  reason='The repair no longer compiled; requesting complete source.')
                            refined_candidate = refine_code(
                                client, task_spec, candidate, verification,
                                evaluation_report=evaluation_report, feedback_bundle=feedback,
                                feedback_mode="full", language=language,
                                execution_mode=execution_mode, entry_point=entry_point,
                                strategy=strategy, repeated_from=duplicate_of,
                                algorithm_plan=algorithm_plan,
                                reference_program=test_suite.validation.get('reference_source'),
                                force_fresh_source=True,
                            )
                if refined_candidate.repair_validation_error:
                    stage("repair_rejected", attempt_number=attempt_number + 1,
                          reason=refined_candidate.repair_validation_error)
                if config_name is not None:
                    # Ablations count every generated candidate, even repeated
                    # code. Extra retries here would exceed the sampling budget.
                    candidate = refined_candidate
                    break
                fingerprint = _candidate_fingerprint(refined_candidate.raw_code)
                if fingerprint not in seen_candidates:
                    repeated_count = 0
                    seen_candidates[fingerprint] = attempt_number + 1
                    candidate = refined_candidate
                    break

                # Give a stochastic repair model a bounded opportunity to
                # produce a new solution, but do not execute identical code
                # again or count it as another evaluated attempt.
                repeated_count += 1
                duplicate_of = seen_candidates[fingerprint]
                # Keep a rejected edit's reason available to the next bounded
                # repair call, without treating it as newly executable source.
                candidate = replace(candidate, repair_validation_error=refined_candidate.repair_validation_error)
                if repeated_count >= max_consecutive_repeats:
                    if judge_only_passing and attempt_number < remaining_attempts:
                        # Interactive solves: a model that keeps echoing its draft
                        # gets a fresh, independent sample (higher temperature)
                        # instead of ending the run with budget left. Each sample
                        # is graded as an attempt, so the budget still bounds this.
                        stage("fresh_sample", attempt_number=attempt_number + 1,
                              reason='Repairs repeated the same code; drawing a new independent solution.')
                        fresh = generate_code(
                            client, task_spec, temperature=0.8,
                            prompt_registry=prompt_registry, prompt_id=candidate.prompt_version,
                            language=language, execution_mode=execution_mode, entry_point=entry_point,
                            algorithm_plan=algorithm_plan,
                        )
                        fingerprint = _candidate_fingerprint(fresh.raw_code)
                        if fingerprint not in seen_candidates:
                            seen_candidates[fingerprint] = attempt_number + 1
                            candidate = fresh
                            repeated_count = 0
                            break
                    result.stop_reason = "repeated_candidate"
                    break

            if result.stop_reason == "repeated_candidate":
                attempt.feedback = collect_feedback(
                    attempt.evaluation_report,
                    test_source=test_source,
                    stop_reason="repeated_candidate",
                )
                if candidate.repair_validation_error:
                    attempt.feedback.candidate_feedback += "\nSource edit validation: " + candidate.repair_validation_error
                if history_store is not None:
                    history_store.record_attempt(result.history_run_id, attempt)
                break
        else:
            result.stop_reason = (
                "review_required"
                if evaluation_report.status == EvaluationStatus.REVIEW
                else "attempt_limit"
            )

    result.test_model_calls = test_model_calls
    result.model_calls = cached_analysis_calls + client.call_count + (
        judge_client.call_count if judge_client is not None else 0
    ) + (test_client.call_count if test_client is not None else 0)
    result.wall_clock_seconds = time.perf_counter() - run_started
    if history_store is not None and complete_history:
        history_store.complete_run(result.history_run_id, result)
        from src.evolution.promotion import monitor_training_completion
        monitor_training_completion(history_store, prompt_registry, benchmark_split)
    stage("finished", history_run_id=result.history_run_id, stop_reason=result.stop_reason,
          metrics=result.stage_metrics)
    return result


# Backward-compatible names for callers from the original milestone. New code
# should use the candidate-refinement terminology to distinguish this per-task
# loop from the future cross-task prompt-evolution loop.
ReinforcementResult = CandidateRefinementResult


def run_reinforcement_loop(
    client,
    problem_statement: str,
    max_attempts: Optional[int] = None,
    max_feedback_failures: int = 3,
    max_consecutive_repeats: int = 2,
    on_attempt: Optional[Callable[[AttemptResult], None]] = None,
    trusted_test_suite: Optional[TestSuite] = None,
    prompt_registry=None,
    prompt_id: Optional[str] = None,
    judge_client=None,
    judge_max_attempts: int = 3,
    judge_review_threshold: float = 0.5,
    judge_only_passing: bool = False,
    history_store=None,
    task_id: Optional[str] = None,
    benchmark_split: str = "adhoc",
    config_name: Optional[str] = None,
    seed: Optional[int] = None,
    language: Optional[str] = None,
    task_spec_cache=None,
    reasoning_mode='direct',
    test_client=None,
) -> CandidateRefinementResult:
    """Compatibility wrapper for :func:`run_candidate_refinement_loop`."""
    return run_candidate_refinement_loop(
        client=client,
        problem_statement=problem_statement,
        max_attempts=max_attempts,
        max_feedback_failures=max_feedback_failures,
        max_consecutive_repeats=max_consecutive_repeats,
        on_attempt=on_attempt,
        trusted_test_suite=trusted_test_suite,
        prompt_registry=prompt_registry,
        prompt_id=prompt_id,
        judge_client=judge_client,
        judge_max_attempts=judge_max_attempts,
        judge_review_threshold=judge_review_threshold,
        judge_only_passing=judge_only_passing,
        history_store=history_store,
        task_id=task_id,
        benchmark_split=benchmark_split,
        config_name=config_name,
        seed=seed,
        language=language,
        task_spec_cache=task_spec_cache,
        reasoning_mode=reasoning_mode,
        test_client=test_client,
    )
