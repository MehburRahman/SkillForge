import hashlib
from pathlib import Path
from typing import Tuple

from src.benchmark.execution_mode import validate_execution
from src.evaluation.aggregator import build_evaluation_report
from src.evaluation.constraint_grader import grade_constraint_adherence
from src.evaluation.functional_grader import grade_functional_correctness
from src.evaluation.llm_judge import judge_code
from src.evaluation.models import EvaluationReport
from src.evaluation.runtime_grader import grade_runtime_robustness
from src.feedback.collector import collect_feedback
from src.feedback.models import FeedbackBundle


MAX_JUDGE_ERROR_CHARACTERS = 500
EVALUATION_POLICY_VERSION = "language-aware-v1"


def evaluation_policy_identity():
    """Keep benchmark resumes on the same deterministic and advisory policy."""
    directory = Path(__file__).parent
    sources = [
        (directory / name).read_bytes()
        for name in (
            "pipeline.py", "aggregator.py", "constraint_grader.py",
            "functional_grader.py", "runtime_grader.py", "llm_judge.py",
        )
    ]
    sources.extend((directory.parent / path).read_bytes() for path in
                   ('evaluation/requirement_ledger.py','verifier/http_requirements.py'))
    return {
        "version": EVALUATION_POLICY_VERSION,
        "sha256": hashlib.sha256(b"\0".join(sources)).hexdigest(),
    }


def evaluate_candidate(
    task_spec,
    code: str,
    verification_result,
    test_source: str,
    judge_client=None,
    judge_max_attempts: int = 3,
    judge_review_threshold: float = 0.5,
    *,
    language="python",
    execution_mode="stdio",
    entry_point=None,
    judge_enforce_consistency=True,
    test_suite=None,
    generator_model=None,
    judge_only_passing=False,
) -> Tuple[EvaluationReport, FeedbackBundle]:
    """Build all grades from one existing candidate execution trace."""
    language, execution_mode, entry_point = validate_execution(language, execution_mode, entry_point)
    functional_grade = grade_functional_correctness(verification_result)
    runtime_grade = grade_runtime_robustness(verification_result)
    constraint_grade = grade_constraint_adherence(code, task_spec, language=language)
    ledger = None
    if execution_mode == 'http' and test_suite is not None:
        from src.verifier.http_requirements import extract_http_requirements, execution_ledger
        requirements = test_suite.validation.get('requirements')
        if requirements is None:
            requirements, _ = extract_http_requirements(task_spec.problem_statement,
                                                       getattr(task_spec,'requirements',()))
        ledger = execution_ledger(requirements,test_suite,verification_result,code,
                                  independent=test_source=='trusted')
        constraint_grade.requirement_ledger = ledger
        constraint_grade.coverage = ledger['coverage']
        constraint_grade.score = ledger['coverage']
        constraint_grade.checked_constraints = [r['text'] for r in ledger['requirements'] if r['state']!='unverified']
        constraint_grade.passed_constraints = [r['text'] for r in ledger['requirements'] if r['state']=='verified']
        constraint_grade.unverified_constraints = [r['text'] for r in ledger['requirements'] if r['state']=='unverified']
        constraint_grade.violations = [dict(constraint=r['text'],message=r['text'],requirement_id=r['id'])
                                       for r in ledger['requirements'] if r['state']=='failed']
        constraint_grade.critical_violation = bool(constraint_grade.violations)
    judge_grade = None
    judge_error = None
    same_model = ledger is not None and generator_model is not None and getattr(judge_client,'model',None)==generator_model
    if same_model:
        judge_error='Same-model judging disabled: requirement execution evidence determines the verdict.'
    # The judge can never override a failed test, so interactive solves may skip
    # it on failing attempts; experiments keep judging every candidate (RQ3).
    skip_failed = judge_only_passing and not functional_grade.passed_all
    if judge_client is not None and not runtime_grade.compilation_error_count and not same_model and not skip_failed:
        try:
            judge_grade = judge_code(
                judge_client,
                task_spec,
                code,
                functional_grade,
                runtime_grade,
                constraint_grade,
                max_attempts=judge_max_attempts,
                language=language, execution_mode=execution_mode, entry_point=entry_point,
                enforce_consistency=judge_enforce_consistency,
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            judge_error = str(exc)[:MAX_JUDGE_ERROR_CHARACTERS]

    report = build_evaluation_report(
        functional_grade,
        runtime_grade,
        constraint_grade,
        None if ledger is not None else judge_grade,
        judge_review_threshold=judge_review_threshold,
        judge_error=judge_error,
    )
    if ledger is not None:
        report.judge_grade = judge_grade
        report.evidence_confidence = min(ledger['coverage'], functional_grade.score)
        report.secondary_score = min(judge_grade.overall_score,report.evidence_confidence) if judge_grade else report.evidence_confidence
        # Advisory judgment is retained verbatim for audit. It cannot promote
        # missing evidence or override an executed failure.
        test_suite.validation.update(requirement_ledger=ledger,
            coverage_complete=ledger['coverage_complete'],all_requirements_verified=ledger['all_verified'])
    feedback = collect_feedback(report, test_source=test_source)
    return report, feedback
