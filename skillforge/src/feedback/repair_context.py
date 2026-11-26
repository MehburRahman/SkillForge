"""Compact, explicit visible failure evidence for semantic repair planning."""

from collections.abc import Mapping
import re

from src.feedback.evidence import bounded_diagnostic, bounded_text


_VALUE_LIMIT = 500
_DIAGNOSTIC_LIMIT = 700
_FRAME = re.compile(r"^\s*(?:at\s+|File\s+\")")


def _kind(status):
    return {
        "pass": "wrong_output",
        "fail": "assertion_failure",
        "error": "runtime_error",
        "timeout": "timeout",
        "compile_error": "compilation_error",
    }.get(status, "unclassified_failure")


def _status(failure):
    value = failure.get("status", "unknown")
    return str(getattr(value, "value", value))


def _compact_diagnostic(value):
    """Retain the cause and a few distinct frames rather than a repeated trace."""
    lines, seen, frame_count = [], set(), 0
    # Deduplicate complete frames before truncation can split a frame into two
    # fragments and prevent it matching its repeated copy near the trace's tail.
    for line in str(value).splitlines():
        normalized = line.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        if _FRAME.match(line):
            if frame_count >= 3:
                continue
            frame_count += 1
        # A middle truncation marker adds no diagnosis after duplicate removal.
        if normalized in ("...[truncated]...", "[additional compiler output omitted]"):
            continue
        lines.append(line)
    return bounded_text(bounded_diagnostic("\n".join(lines)), _DIAGNOSTIC_LIMIT)


def _value(value):
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return bounded_text(value, _VALUE_LIMIT)


def _rank(failure, index):
    has_input = "input" in failure
    text = str(failure.get("input", ""))
    useful = has_input and (
        "expected" in failure or "expected_output" in failure
        or "actual" in failure or bool(failure.get("stderr"))
    )
    return (
        not useful,
        len(text) > _VALUE_LIMIT if has_input else True,
        len(text.split()) if has_input else float("inf"),
        len(text) if has_input else float("inf"),
        _kind(_status(failure)) != "wrong_output",
        index,
    )


def compact_repair_evidence(failures, *, first_failure=None, runtime_failures=()):
    """Select one small failed case from explicitly supplied visible evidence.

    ``status`` in verifier failures describes process execution. A process that
    exits successfully can still fail its output comparison, so the rendered
    evidence names execution status and test outcome separately. This helper does
    not recompute answers, inspect case results, or load experiment/hidden data.

    ``runtime_failures`` may supply an additional visible cause when the selected
    counterexample is a wrong answer and the functional excerpt omitted crashes.
    The result is a JSON-serializable dictionary ready for a diagnosis prompt.
    """
    available = [dict(item) for item in (failures or ()) if isinstance(item, Mapping)]
    if isinstance(first_failure, Mapping) and dict(first_failure) not in available:
        available.append(dict(first_failure))
    result = {}
    selected_diagnostic = ""
    if available:
        selected = min(enumerate(available), key=lambda pair: _rank(pair[1], pair[0]))[1]
        execution_status = _status(selected)
        counterexample = {
            "execution_status": execution_status,
            "test_passed": False,
            "failure_kind": _kind(execution_status),
        }
        # Compiler failures precede stdin execution; they provide no behavioral
        # counterexample even when the verifier attaches a test's input to them.
        if execution_status != "compile_error":
            for name, aliases in (
                ("input", ("input",)),
                ("expected", ("expected", "expected_output")),
                ("actual", ("actual",)),
            ):
                key = next((key for key in aliases if key in selected), None)
                if key is not None:
                    counterexample[name] = _value(selected[key])
                    if isinstance(selected[key], str) and len(selected[key]) > _VALUE_LIMIT:
                        counterexample[name + "_truncated"] = True
        if "exit_code" in selected:
            counterexample["exit_code"] = selected["exit_code"]
        if selected.get("stderr"):
            selected_diagnostic = _compact_diagnostic(selected["stderr"])
            counterexample["diagnostic"] = selected_diagnostic
        result["counterexample"] = counterexample

    # Keep one complementary cause, not another input/output example or a second
    # copy of the selected trace. Both sources were supplied explicitly by caller.
    for failure in (*available, *(runtime_failures or ())):
        if not isinstance(failure, Mapping) or not failure.get("stderr"):
            continue
        if _status(failure) not in ("error", "timeout", "compile_error"):
            continue
        diagnostic = _compact_diagnostic(failure["stderr"])
        if diagnostic and diagnostic != selected_diagnostic:
            result["runtime_diagnostic"] = diagnostic
            break
    return result
