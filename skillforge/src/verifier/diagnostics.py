"""Execution evidence for repairs; uncertain outputs never become pass scores."""

import re

from src.verifier.outcome_verifier import VerificationResult
from src.verifier.sandbox import run_code_many, Status
from src.verifier.source_check import check_source


# Default object representations that a runtime prints when a program passes an
# object instead of its value to print. They are never a valid answer to a
# stdio task, so they are defects regardless of the problem.
_LEAKED_REPRESENTATIONS = (
    re.compile(r'\[{1,3}[BCDFIJSZ]@[0-9a-f]{4,}'),                 # JVM primitive arrays: [J@1b6d3586
    re.compile(r'\[{1,3}L[\w.$]+;@[0-9a-f]{4,}'),                  # JVM object arrays
    re.compile(r'\b(?:[a-z_]\w*\.)*[A-Z][\w$]*@[0-9a-f]{5,}\b'),   # Object.toString: Main$Node@6d06d69c
    re.compile(r'<[\w.]+ object at 0x[0-9a-fA-F]+>'),              # Python default repr
    re.compile(r'<(?:function|bound method|built-in function|generator object|lambda)\b[^>]*>'),
)


def leaked_representation(stdout):
    """Return the leaked runtime object representation in stdout, if any."""
    for pattern in _LEAKED_REPRESENTATIONS:
        match = pattern.search(stdout or '')
        if match:
            return match.group(0)
    return None


def _token_kind(token):
    if re.fullmatch(r'[+-]?\d+', token):
        return 'integer'
    if re.fullmatch(r'[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?', token):
        return 'number'
    return 'text'


def output_shape(text):
    """Line count, tokens per line and token kinds, ignoring the values themselves."""
    return [tuple(_token_kind(token) for token in line.split())
            for line in (text or '').strip().splitlines() if line.strip()]


def _describe_shape(shape):
    if not shape:
        return 'no output'
    if len(shape) > 3:
        return f'{len(shape)} lines'
    return '; '.join(f"line {number}: {len(kinds)} token(s) {'/'.join(kinds) or 'empty'}"
                     for number, kinds in enumerate(shape, 1))


def shape_failure(actual, reference):
    """Explain a mismatch between the candidate's and an unverified reference's output layout.

    The reference answer value stays hidden: only its layout is compared, so an
    unverified model-generated value can never be copied into the candidate.
    """
    leaked = leaked_representation(actual)
    if leaked:
        return (f'stdout contains a runtime object representation ({leaked}) instead of a value; '
                'print the element or aggregate the task asks for, as the output format states.')
    if not isinstance(reference, str) or not reference.strip():
        return None
    expected_shape, actual_shape = output_shape(reference), output_shape(actual)
    if expected_shape == actual_shape:
        return None
    # Integers are also valid numbers; do not report 3 vs 3.0 style differences.
    loose = lambda shape: [tuple('number' if kind == 'integer' else kind for kind in line)
                           for line in shape]
    if loose(expected_shape) == loose(actual_shape):
        return None
    return ('output layout differs from an unverified reference program: expected '
            f'{_describe_shape(expected_shape)}, got {_describe_shape(actual_shape)}. '
            'Check the printed layout against the original output format; the reference '
            'answer value is unverified and is not shown.')


def diagnose_candidate(code, *, language='python', suite=None, on_progress=None):
    cases = list(suite.cases) if suite and suite.cases else []
    inputs = [case.input for case in cases]
    executions = (run_code_many(code, inputs, language=language, on_progress=on_progress)
                  if inputs else [check_source(code, language=language, on_progress=on_progress)])
    failures = []
    for index, execution in enumerate(executions):
        if execution.status != Status.PASS:
            failures.append(dict(input=inputs[index] if inputs else '',
                status=execution.status.value, stderr=execution.stderr,
                exit_code=execution.exit_code, actual=execution.stdout,
                evidence_scope='execution_only; input validity and expected output are unverified'))
        elif inputs:
            reference = (getattr(cases[index], 'expected_output', None)
                         if getattr(suite, 'reference_outputs', False) else None)
            reason = shape_failure(execution.stdout, reference)
            if reason:
                failures.append(dict(input=inputs[index], status='output_shape_mismatch',
                    reason=reason, exit_code=execution.exit_code, actual=execution.stdout,
                    evidence_scope='output layout only; expected values are unverified'))
    # No functional pass count exists here, even when execution succeeds.
    return VerificationResult(pass_rate=0.0, total=0, passed=0,
                              first_failure=failures[0] if failures else None,
                              failures=failures[:3]), executions
