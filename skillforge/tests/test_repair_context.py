"""Regression checks for compact evidence passed to a semantic repair planner."""

import copy
import json

import pytest

from src.evaluation.models import EvaluationStatus
from src.feedback.repair_context import compact_repair_evidence


def wrong_output(stdin="2 3", expected="5", actual="6"):
    return dict(input=stdin, expected=expected, actual=actual, status="pass", exit_code=0)


def test_successful_process_is_explicitly_a_failed_output_comparison():
    evidence = compact_repair_evidence([wrong_output()])["counterexample"]
    assert evidence["execution_status"] == "pass"
    assert evidence["test_passed"] is False
    assert evidence["failure_kind"] == "wrong_output"
    assert evidence["expected"] == "5" and evidence["actual"] == "6"
    assert "status" not in evidence


def test_enum_statuses_are_rendered_as_execution_values():
    failure = wrong_output()
    failure["status"] = EvaluationStatus.PASS
    evidence = compact_repair_evidence([failure])["counterexample"]
    assert evidence["execution_status"] == "pass"
    assert evidence["failure_kind"] == "wrong_output"


def test_smallest_useful_counterexample_keeps_its_exact_whitespace_and_outputs():
    large = wrong_output("1000\n" + "1 " * 1000, "1000", "999")
    small = wrong_output("2\t3\n", "5\n", "6\n")
    selected = compact_repair_evidence([large, small])["counterexample"]
    assert selected["input"] == "2\t3\n"
    assert selected["expected"] == "5\n" and selected["actual"] == "6\n"


def test_complete_short_case_is_preferred_to_an_abbreviated_single_large_token():
    selected = compact_repair_evidence([
        wrong_output("x" * 1000), wrong_output("2 3 4"),
    ])["counterexample"]
    assert selected["input"] == "2 3 4"
    assert "input_truncated" not in selected


def test_incomplete_diagnostic_record_does_not_hide_an_actionable_case():
    selected = compact_repair_evidence([
        dict(status="error", stderr="RuntimeError: no input record"), wrong_output(),
    ])["counterexample"]
    assert selected["input"] == "2 3"
    assert selected["failure_kind"] == "wrong_output"


def test_a_legitimately_empty_input_and_expected_output_remain_present():
    selected = compact_repair_evidence([wrong_output("", "", "debug output")])["counterexample"]
    assert selected["input"] == "" and selected["expected"] == ""
    assert selected["actual"] == "debug output"


def test_expected_output_alias_also_preserves_empty_text():
    selected = compact_repair_evidence([
        dict(input="1", expected_output="", actual="debug", status="pass"),
    ])["counterexample"]
    assert selected["expected"] == "" and selected["actual"] == "debug"


@pytest.mark.parametrize("expected", [0, False, None])
def test_falsy_expected_values_are_not_dropped_or_replaced(expected):
    selected = compact_repair_evidence([wrong_output(expected=expected)])["counterexample"]
    assert "expected" in selected and selected["expected"] is expected


def test_a_crash_after_printing_the_expected_output_still_fails():
    selected = compact_repair_evidence([
        dict(input="1", expected="", actual="", status="error", exit_code=1,
             stderr="RuntimeError: crashed after output"),
    ])["counterexample"]
    assert selected["failure_kind"] == "runtime_error"
    assert selected["execution_status"] == "error" and selected["test_passed"] is False
    assert selected["expected"] == selected["actual"] == ""


def test_repeated_stack_frames_are_compacted_without_losing_the_exception_cause():
    trace = ("Exception in thread \"main\" java.lang.StackOverflowError\n"
             + "\tat Candidate.visit(Candidate.scala:8)\n" * 100
             + "Caused by: IllegalStateException: cycle\n")
    selected = compact_repair_evidence([
        dict(input="1", expected="ok", actual="", status="error", stderr=trace),
    ])["counterexample"]
    diagnostic = selected["diagnostic"]
    assert "StackOverflowError" in diagnostic and "IllegalStateException: cycle" in diagnostic
    assert diagnostic.count("Candidate.visit(Candidate.scala:8)") == 1
    assert len(diagnostic) <= 750


def test_compiler_boilerplate_does_not_hide_the_actual_error():
    diagnostic = ("Scala CLI progress\n" * 100
                  + "-- [E007] Type Mismatch Error: /tmp/Candidate.scala:2:9\n"
                  + "2 | val x: Int = text\n  | Found: String\n  | Required: Int\n")
    selected = compact_repair_evidence([
        dict(input="1", expected="2", actual="", status="compile_error", stderr=diagnostic),
    ])["counterexample"]
    assert selected["failure_kind"] == "compilation_error"
    assert "Required: Int" in selected["diagnostic"] and "Scala CLI progress" not in selected["diagnostic"]
    assert "input" not in selected and "expected" not in selected and "actual" not in selected


def test_large_evidence_is_marked_as_incomplete_instead_of_silently_shortened():
    selected = compact_repair_evidence([
        wrong_output("x" * 1000, "y" * 1000, "z" * 1000),
    ])["counterexample"]
    assert selected["input_truncated"] is True
    assert selected["expected_truncated"] is True and selected["actual_truncated"] is True
    assert "[truncated]" in selected["input"]


def test_first_failure_is_used_when_the_visible_failure_list_is_empty():
    failure = wrong_output()
    assert compact_repair_evidence([], first_failure=failure) == compact_repair_evidence([failure])


def test_complementary_runtime_cause_is_included_once_with_a_wrong_answer():
    trace = "RuntimeError: implementation failed\n"
    evidence = compact_repair_evidence([wrong_output()], runtime_failures=[
        dict(status="error", stderr=trace), dict(status="error", stderr=trace),
    ])
    assert evidence["counterexample"]["failure_kind"] == "wrong_output"
    assert evidence["runtime_diagnostic"] == trace.strip()
    assert json.dumps(evidence).count("implementation failed") == 1


def test_successful_execution_warning_cannot_displace_a_proven_runtime_error():
    evidence = compact_repair_evidence([
        wrong_output("2 3"), dict(wrong_output("4 5"), stderr="Warning: optional notice"),
    ], runtime_failures=[dict(status="error", stderr="RuntimeError: broken")])
    assert evidence["runtime_diagnostic"] == "RuntimeError: broken"


def test_runtime_only_evidence_does_not_manufacture_an_input_or_answer():
    evidence = compact_repair_evidence([], runtime_failures=[dict(status="error", stderr="RuntimeError: broken")])
    assert "counterexample" not in evidence
    assert evidence["runtime_diagnostic"] == "RuntimeError: broken"


def test_input_records_are_not_mutated_and_empty_evidence_remains_empty():
    failure = wrong_output()
    original = copy.deepcopy(failure)
    compact_repair_evidence([failure])
    assert failure == original
    assert compact_repair_evidence([]) == {}
