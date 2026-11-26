import json
import re
from dataclasses import replace

from src.agents.code_generator import CodeCandidate
from src.agents.code_prompts import execution_contract, generation_system, task_context
from src.feedback.evidence import bounded_failures, bounded_text, bounded_diagnostic
from src.utils.parsing import strip_markdown_fence
from src.utils.parsing import parse_json_response
from src.agents.source_edits import (
    SOURCE_EDIT_RESPONSE_SCHEMA, apply_source_edits, line_edit_schema, apply_line_edits,
)


def _task_context(task_spec, previous_candidate, contract):
    """Keep the task information identical across both repair arms."""
    return f"""{task_context(task_spec)}

Execution contract:
{contract}

Previous solution:
{previous_candidate.raw_code}"""


def _comparable_body(value):
    """Return the response body of an HTTP record, or the plain stdio text."""
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except ValueError:
            return value
        value = decoded
    if isinstance(value, dict):
        body = value.get("body")
        return body if isinstance(body, str) else None
    return value if isinstance(value, str) else None


def _escaping_diagnosis(failures):
    """Name a missing/extra HTML-escaping step when that alone explains a wrong output."""
    import html
    for failure in failures[:3]:
        expected, actual = _comparable_body(failure.get("expected")), _comparable_body(failure.get("actual"))
        if not expected or not actual or expected == actual:
            continue
        if html.unescape(expected) == actual:
            return (
                "Escaping diagnosis: the expected output equals the actual output with the HTML special "
                "characters & < > \" ' replaced by entities (&amp; &lt; &gt; &quot; &#39;). The program "
                "inserts values without escaping them. Escape each inserted value, replacing & first, "
                "before it enters the output; do not escape the surrounding template text."
            )
        if html.unescape(actual) == expected:
            return (
                "Escaping diagnosis: the actual output contains HTML entities where the expected output "
                "has the literal characters. Text that must be emitted verbatim is being escaped, or a "
                "value is escaped twice. Escape only the inserted values, exactly once."
            )
    return ""


_NON_LOCAL_RETURN = re.compile(r'Candidate\.scala:(\d+):\d+[^\n]*\n(\d+ \|[^\n]*)\n[^\n]*\n[^\n]*Non local returns')


def _scala_warning_diagnosis(code, language):
    """Report compiler warnings that predict runtime failures (currently non-local returns)."""
    if language != "scala" or not re.search(r'(?<![\w.])return\b', code):
        return ""
    try:
        from src.verifier.source_check import check_source
        output = check_source(code, language="scala").stderr or ""
    except Exception:
        return ""
    found = _NON_LOCAL_RETURN.findall(output)
    if not found:
        return ""
    lines = "; ".join(f"line {number}: {text.split('|', 1)[1].strip()[:120]}" for number, text in found[:3])
    return ("Compiler warning diagnosis: `return` inside a lambda or callback is a non-local return ("
            f"{lines}). Scala implements it by throwing an exception, so in a handler, map or "
            "replaceAllIn function it aborts the enclosing work: a server request ends without a "
            "response. Do not return from inside the lambda; decide first (for example, check every "
            "required value before substituting) and then respond, or use scala.util.boundary/break.")


def _scala_parsing_diagnosis(failures, language, execution_mode, previous_code=''):
    if language == "scala":
        for failure in failures[:3]:
            missing = re.search(r'java\.util\.NoSuchElementException: key not found: (.{1,40})',
                                failure.get("stderr", "") or "")
            if missing:
                return (
                    f"Runtime diagnosis: Map.apply threw 'key not found: {missing.group(1).strip()}'. "
                    "map(key) fails for a key that was never inserted. Find every map(key) read "
                    "and every map(key) += / -= update on a key that may be absent, and use "
                    "map.getOrElse(key, 0), map.contains(key), or a map created with "
                    ".withDefaultValue(0). Keep the algorithm; fix only these lookups."
                )
        # Regex.replaceAllIn reads $n and \ in the returned text as group
        # references/escapes, so inserted data containing them is corrupted.
        if (re.search(r'\.replaceAllIn\s*\(', previous_code)
                and 'quoteReplacement' not in previous_code
                and any(re.search(r'[$\\]|%24|%5[Cc]', str(failure.get("input", "")))
                        for failure in failures[:3])):
            return (
                "Replacement diagnosis: a failing input contains $ or \\. Regex.replaceAllIn treats "
                "$1 and backslashes in each returned replacement as group references and escapes, so "
                "inserted data is altered or throws. Return "
                "scala.util.matching.Regex.quoteReplacement(value) from the replacement function. "
                "Keep the rest of the program."
            )
    if language != "scala" or execution_mode != "stdio":
        return ""
    for failure in failures[:3]:
        diagnostic = failure.get("stderr", "")
        if ("NumberFormatException" in diagnostic
                and re.search(r"scala\.io\.StdIn\.read(?:Int|Long)", diagnostic)
                and re.search(r'For input string: "[+-]?\d+\s+[+-]?\d+', diagnostic)):
            return (
                "Input parsing diagnosis: The visible crash occurs while StdIn.readInt/readLong "
                "tries to parse a line containing several numbers as one integer. Repair these input reads: "
                "use a whitespace token parser and consume tokens in the exact order stated by the task, "
                "including values sharing its header line. Match Int/Long to the task's ranges. "
                "Changing an unrelated algorithm edge case will leave this same parsing exception."
            )
        # Destructuring a one-element split (val Array(a, b) = line.split(...))
        # raises MatchError instead of an index error; the cause is the same.
        if (('ArrayIndexOutOfBoundsException' in diagnostic
                and re.search(r'Index \d+ out of bounds for length 1', diagnostic))
                or re.search(r'scala\.MatchError: \[Ljava\.lang\.String;', diagnostic)
                # lines(1) on a List built from one line: IndexOutOfBoundsException: 1
                or re.search(r'java\.lang\.IndexOutOfBoundsException: 1\b', diagnostic)):
            binding = re.search(
                r'\b(?:val|var)\s+(\w+)\s*=\s*(?:scala\.io\.)?StdIn\.readLine\s*\(',
                previous_code,
            )
            # Either quote style: split("\n") or split('\n').
            split_on_newline = bool(binding and re.search(
                re.escape(binding.group(1)) + r'\s*\.split\s*\(\s*["\'][^"\'\n]*\\n',
                previous_code,
            )) or bool(re.search(r'StdIn\.readLine\s*\(\s*\)\s*\.split\s*\(\s*["\'][^"\'\n]*\\n', previous_code))
            if split_on_newline:
                return (
                    "Input parsing diagnosis: StdIn.readLine consumed only the first input line and removed its newline. "
                    "Splitting that single line on a newline produces one field, causing the observed array-index crash. "
                    "Repair the input reader to consume every required record using separate reads or the complete input. "
                    "Preserve required spaces and valid empty/trailing records. Do not merely guard the array access or "
                    "return an empty answer; that would discard supplied arguments."
                )
    return ""


def _http_protocol_diagnosis(failures, language, execution_mode, previous_code=''):
    if language != 'scala' or execution_mode != 'http':
        return ''
    string_lengths = re.findall(r'\.sendResponseHeaders\s*\([^,]+,\s*([A-Za-z_]\w*)\.length(?:\.toLong)?\s*\)', previous_code)
    character_length = bool(failures) and any(re.search(
        rf'\b{re.escape(name)}\.getBytes\s*\(', previous_code) for name in string_lengths)
    missing_body=False
    framing=[]
    for failure in failures:
        try:
            expected=json.loads(failure.get('expected',''))
            actual=json.loads(failure.get('actual',''))
            if isinstance(expected,dict) and isinstance(actual,dict) and isinstance(expected.get('body'),str):
                missing_body=missing_body or bool(expected['body']) and actual.get('body')==''
                declared=actual.get('declared_body_bytes')
                required=expected.get('utf8_bytes')
                if type(declared) is int and type(required) is int and declared!=required:
                    message=(f'Observed declared response length: {declared} bytes; '
                             f'expected response UTF-8 size: {required} bytes; '
                             f'received: {actual.get("received_body_bytes")} bytes.')
                    if message not in framing:framing.append(message)
        except (ValueError,TypeError):pass
    if not character_length and not missing_body and not framing and not any(failure.get('status') == 'timeout'
               or re.search(r'got status (?:204|304)\b', failure.get('stderr', ''))
               or 'Remote end closed connection without response' in failure.get('stderr', '')
               or 'IncompleteRead' in failure.get('stderr','')
               or 'body did not complete' in failure.get('stderr','')
               for failure in failures):
        return ''
    return (('HTTP framing evidence: '+' '.join(framing[:3])+
             ' Derive the length from the response byte array at runtime; do not copy observed counts as constants.\n' if framing else '')
            + ('HTTP response-body diagnosis: the visible comparison requires a nonempty response body, '
             'but the handler returned no bytes. Sending headers or updating state does not send the '
             'required result. Write the required response bytes before closing the exchange.\n' if missing_body else '')
            + ('HTTP byte-length diagnosis: this source advertises a text variable .length but writes '
             'that variable .getBytes(...). String.length counts UTF-16 characters, not UTF-8 bytes. '
             'Encode the response once, pass the resulting byte array length to sendResponseHeaders, '
             'and write that exact byte array. Check every response handler for this mismatch. '
             'Decode request bytes explicitly as UTF-8 too.\n' if character_length else '')
            + 'HTTP protocol diagnosis: a handler can send headers and still leave the client waiting '
            'forever if the response is not completed. In JDK HttpServer, sendResponseHeaders length '
            '0 selects chunked streaming; it does not mean an empty response. Use -1 for bodyless '
            'responses, otherwise write exactly the advertised number of UTF-8 bytes. '
            'Call sendResponseHeaders BEFORE writing the body; writing first can make '
            'the server close the connection without a response. Close every '
            'exchange/response stream in all branches, preferably with finally. Status 204/304 '
            'cannot carry a body. Preserve the task-specified body behavior and fix response framing '
            'throughout the server rather than changing routes, expected answers, or client timeouts.')


def refine_code(
    client,
    task_spec,
    previous_candidate: CodeCandidate,
    verification_result,
    temperature: float = 0.2,
    evaluation_report=None,
    feedback_bundle=None,
    feedback_mode="full",
    *,
    language="python",
    execution_mode="stdio",
    entry_point=None,
    strategy=None,
    repeated_from=None,
    prefer_source_edits=False,
    algorithm_plan=None,
    force_fresh_source=False,
    diagnostic_only=False,
    reference_program=None,
) -> CodeCandidate:
    """Generate a repaired candidate from bounded multi-grader feedback."""
    contract = execution_contract(language, execution_mode, entry_point)
    reference_context = ''
    if isinstance(reference_program, str) and reference_program.strip():
        reference_context = ('\nExecutable reference used to derive the visible test responses '
            '(model-generated and fallible, not an independent correctness proof):\n'
            + bounded_text(reference_program,12000)
            + '\nPreserve the original specification. Check each returned response status and body '
              'against this reference and the concrete failed request; do not implement only the '
              'state update while omitting required output. None means an unspecified body, not literal text.\n')
    if reference_context and execution_mode == 'stdio':
        reference_context = ('\nSmall-input reference used to derive visible expected outputs '
            '(model-generated, fallible, not independently trusted):\n'
            + bounded_text(reference_program, 12000)
            + '\nCompare its input parsing, state transitions, initial state and output aggregation '
              'with the original specification and observed counterexamples. Correct your general '
              'algorithm, then implement it in the target language within the stated resource bounds. '
              'Do not copy test values or assume this small-input implementation is efficient.\n')
    context = _task_context(task_spec, previous_candidate, contract)
    if algorithm_plan is not None:
        context += f"""

Reviewed algorithm plan (model-generated and fallible):
{algorithm_plan.text}

Implement the revised invariants and state updates only where they satisfy the
original problem. Replace an invalid design rather than making cosmetic edits."""
    repetition = (f"This candidate repeats attempt {repeated_from}. The previous repair made no code change. "
                  "Identify the failing statements and replace them; returning the same source cannot repair this failure."
                  if repeated_from is not None else "")
    if feedback_mode == "scalar_repair":
        # This path never reads failures, reports or collected feedback. Only
        # the task, previous code and aggregate pass count reach the model.
        system = f"""You repair incorrect competitive-programming solutions using a test pass count.
Execution contract (authoritative):
{contract}"""
        prompt = f"""{context}

Passed: {verification_result.passed}/{verification_result.total}

Return an improved solution that follows the execution contract and task requirements.
The solution must be different from the previous one. Try a different approach if needed."""
        if strategy is not None:
            system = generation_system(strategy, contract) + "\n\n" + system
        return _generate_repair(client, previous_candidate, prompt, system, temperature,
                                algorithm_plan=algorithm_plan)
    if feedback_mode != "full":
        raise ValueError("feedback_mode must be full or scalar_repair")

    failures = verification_result.failures
    if not failures and verification_result.first_failure:
        failures = [verification_result.first_failure]
    http_diagnosis = _http_protocol_diagnosis(failures, language, execution_mode, previous_candidate.raw_code)
    warning_diagnosis = (_scala_warning_diagnosis(previous_candidate.raw_code, language)
                         if failures and not any(f.get("status") == "compile_error" for f in failures) else "")
    force_fresh_source = force_fresh_source or bool(http_diagnosis)

    diagnostics = []
    for failure in failures:
        if failure.get("status") == "compile_error" and failure.get("stderr"):
            diagnostic = bounded_diagnostic(failure["stderr"])
            if diagnostic not in diagnostics:
                diagnostics.append(diagnostic)
    if diagnostics:
        import_hints=''
        if language=='scala':
            from src.agents.sdk_hints import compiler_import_hints
            import_hints=compiler_import_hints(diagnostics,previous_candidate.raw_code)
        type_mismatch = any('Type Mismatch' in diagnostic or 'Required:' in diagnostic and 'Found:' in diagnostic
                            for diagnostic in diagnostics)
        visibility_error = language == 'scala' and any(
            'cannot be accessed' in diagnostic and 'private' in diagnostic for diagnostic in diagnostics)
        force_fresh_source = force_fresh_source or type_mismatch or visibility_error or bool(import_hints)
        system = f"You repair compiler errors using the compiler's evidence.\nExecution contract:\n{contract}"
        if strategy is not None:
            system = generation_system(strategy, contract) + "\n\n" + system
        prompt = f"""{task_context(task_spec)}

Execution contract:
{contract}

Previous solution:
{previous_candidate.raw_code}

Compiler diagnostics:
{chr(10).join(diagnostics[:3])}
{import_hints}
{reference_context}

{repetition}
The program did not compile, so behavioral tests have not run. Correct the reported APIs, types,
and language syntax while preserving the required interface. Check the complete source for the
same mistakes. Return the complete corrected source code."""
        if type_mismatch:
            prompt += ('\nType mismatch repair: trace the reported expression back to its declaration. '
                       'The compiler reports the actual Found type and the Required type. Change the '
                       'value construction or use an API that accepts its actual type. Renaming a '
                       'variable or repeating the same call cannot change its type. Do not cast '
                       'unrelated types to silence this error. Return a complete source with the '
                       'reported mismatch removed.')
        if visibility_error:
            prompt += ('\nScala access-control diagnosis: a top-level class cannot access private '
                       'members of another object merely by qualifying their names. Review the '
                       'brace nesting and ownership of every reported member. Keep collaborating '
                       'classes inside the owning scope, pass shared state explicitly, or provide '
                       'an appropriately scoped interface. Preserve required synchronization. '
                       'Fix the primary access errors before secondary missing-binding errors; '
                       'renaming identifiers does not repair visibility. Return the full source '
                       'with consistent declaration and use sites.')
        if language == 'scala' and any('invalid escape character' in diagnostic for diagnostic in diagnostics):
            prompt += ('\nScala string diagnosis: "\\{" and similar sequences are not valid escapes in a normal '
                       'string literal. Write regular expressions as triple-quoted raw strings, e.g. '
                       '"""\\{\\{(\\w+)\\}\\}""".r, and substitute with regex.replaceAllIn(text, m => '
                       'scala.util.matching.Regex.quoteReplacement(value)); String.replaceAll takes two Strings.')
        if language == 'scala' and any('Not found:' in diagnostic for diagnostic in diagnostics):
            if re.search(r'\b(?:val|var)\s+(?:(?:Array|List|Seq|Some)\s*)?\([^\n)]*\b[A-Z]\w*', previous_candidate.raw_code):
                prompt += ('\nScala pattern diagnosis: uppercase names inside destructuring patterns '
                           'are stable-identifier matches, not new variables. Use lowercase binding '
                           'names and update their references consistently. Also check that the '
                           'input reader actually consumes all required records; changing case '
                           'alone cannot repair a reader that reads only one line.')
        if ((repeated_from is not None or prefer_source_edits) and not previous_candidate.repair_validation_error
                and getattr(client, 'supports_source_edit_repairs', False) is True):
            return _generate_evidence_edits(client, previous_candidate, prompt, system,
                                            temperature, algorithm_plan=algorithm_plan)
        if force_fresh_source:
            prompt += '\nThe previous repair did not compile. Generate the complete source afresh; do not apply more partial edits to that invalid repair.'
            return _generate_repair(client, previous_candidate, prompt, system, temperature,
                                    algorithm_plan=algorithm_plan)
        if (repeated_from is not None or prefer_source_edits
                or getattr(client, 'source_edit_style', None) == 'lines'):
            return _generate_evidence_edits(client, previous_candidate, prompt, system, temperature,
                                            algorithm_plan=algorithm_plan)
        return _generate_repair(client, previous_candidate, prompt, system, temperature,
                                algorithm_plan=algorithm_plan)

    if http_diagnosis:
        prompt=f"""Original specification (authoritative):
{task_spec.problem_statement}

Execution contract:
{contract}

Current source:
{previous_candidate.raw_code}

Executed counterexamples:
{json.dumps(bounded_failures(failures[:3]),ensure_ascii=False)}

{http_diagnosis}
{_escaping_diagnosis(failures)}
{warning_diagnosis}
{reference_context}
{repetition}
Repair the concrete response failure throughout the program. A previous algorithm plan is
not authoritative and is omitted here. Do not substitute an acknowledgment for a required
payload. Encode each response once, advertise that byte array's length, write that array,
and close the exchange. Preserve the specification's state and interface requirements.
Return the complete corrected source, not an explanation, patch, or unchanged program."""
        if (repeated_from is not None and not previous_candidate.repair_validation_error
                and getattr(client,'supports_source_edit_repairs',False) is True):
            # Repeating a complete source is not recovery. Switch the model's
            # response representation once, while retaining all host validation
            # and the caller's existing retry/candidate limits.
            return _generate_evidence_edits(client,previous_candidate,prompt,
                'Repair only the statements established wrong by execution evidence. '+contract,
                temperature,failure_kind='HTTP response')
        return _generate_repair(client,previous_candidate,prompt,
            'Repair the program using actual execution evidence. '+contract,temperature)

    # HTTP handler repairs often span declarations, response framing and cleanup.
    # Native line patches have repeatedly left duplicate or incomplete handlers;
    # request a complete program for executed HTTP failures instead.
    force_fresh_source = force_fresh_source or execution_mode == 'http'
    system = f"""You repair incorrect competitive-programming solutions using evaluation feedback.
Execution contract (authoritative):
{contract}
Trusted functional tests, runtime evidence, and deterministic constraint checks are authoritative.
LLM-judge feedback is advisory: use it to investigate the code, but never use it to dismiss trusted
test failures, change expected outputs, or ignore a proven constraint violation."""
    if strategy is not None:
        system = generation_system(strategy, contract) + "\n\n" + system

    if diagnostic_only:
        system += ('\nThis is an execution diagnostic, not a functional grade. Expected answers '
                   'are unavailable or unverified. Fix compiler errors using their diagnostics. '
                   'For crashes, check the input against the original problem and shared delivery '
                   'contract before changing the code. Do not adopt unstated restrictions from '
                   'a generated input or infer correct output from a process exit or pass count.')

    collected_feedback = "Unavailable."
    feedback_categories = "[]"
    if feedback_bundle is not None and evaluation_report is not None and not evaluation_report.hard_gate_passed:
        from src.feedback import collect_feedback
        feedback_bundle = collect_feedback(
            replace(evaluation_report, judge_grade=None), test_source='generated',
            stop_reason='repeated_candidate' if any(
                signal.source == 'candidate_loop' for signal in feedback_bundle.signals) else None)
    if feedback_bundle is not None:
        collected_feedback = feedback_bundle.candidate_feedback[:6000]
        feedback_categories = json.dumps(
            [category.value for category in feedback_bundle.categories],
            separators=(",", ":"),
        )
    evidence = bounded_failures(failures)
    for item in evidence:
        # Keep one copy of diagnostics already supplied by the collector.
        # HTTP assertion failures still retain their response diagnostics here.
        if item.get("stderr") and repr(item["stderr"]) in collected_feedback:
            del item["stderr"]
    failures_json = json.dumps(
        evidence,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    evaluation_summary = "Unavailable (legacy verification-only call)."
    if evaluation_report is not None:
        constraint_grade = evaluation_report.constraint_grade
        judge_grade = evaluation_report.judge_grade if evaluation_report.hard_gate_passed else None
        evaluation_summary = json.dumps(
            {
                "status": evaluation_report.status.value,
                "functional": {
                    "score": evaluation_report.functional_grade.score,
                    "passed": evaluation_report.functional_grade.passed,
                    "total": evaluation_report.functional_grade.total,
                    "failed_categories": (
                        evaluation_report.functional_grade.failed_categories
                    ),
                },
                "runtime": {
                    "score": evaluation_report.runtime_grade.score,
                    "timeouts": evaluation_report.runtime_grade.timeout_count,
                    "runtime_errors": (
                        evaluation_report.runtime_grade.runtime_error_count
                    ),
                },
                "constraints": {
                    "score": constraint_grade.score,
                    "coverage": constraint_grade.coverage,
                    "violations": constraint_grade.violations[:3],
                    "unverified": constraint_grade.unverified_constraints[:5],
                },
                "judge_advisory": (
                    {
                        "overall_score": judge_grade.overall_score,
                        "failure_category": judge_grade.likely_failure_category,
                        "critical_issues": judge_grade.critical_issues[:5],
                        "feedback": judge_grade.feedback[:1000],
                    }
                    if judge_grade is not None
                    else None
                ),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    evaluation_summary = bounded_text(evaluation_summary, 4000)
    parsing_diagnosis = (_scala_parsing_diagnosis(failures, language, execution_mode,
                                                 previous_candidate.raw_code)
                         or _escaping_diagnosis(failures) or http_diagnosis)
    parsing_diagnosis = '\n'.join(part for part in (parsing_diagnosis, warning_diagnosis) if part)
    native_edits = getattr(client, 'supports_source_edit_repairs', False) is True
    edit_repair = not force_fresh_source and (
        (prefer_source_edits or repeated_from is not None or (failures and native_edits))
        and (callable(getattr(client, 'generate_json', None)) or native_edits)
    )

    if algorithm_plan is not None:
        from src.feedback.repair_context import compact_repair_evidence
        repair_evidence = compact_repair_evidence(
            failures,
            first_failure=verification_result.first_failure,
            runtime_failures=(evaluation_report.runtime_grade.failure_evidence
                             if evaluation_report is not None else ()),
        )
        if evaluation_report is not None:
            if evaluation_report.constraint_grade.violations:
                repair_evidence['constraint_violations'] = evaluation_report.constraint_grade.violations[:3]
            if evaluation_report.judge_grade is not None and evaluation_report.hard_gate_passed:
                judge = evaluation_report.judge_grade
                repair_evidence['judge_advisory'] = {
                    'failure_category': judge.likely_failure_category,
                    'critical_issues': [bounded_text(issue, 500) for issue in judge.critical_issues[:3]],
                    'feedback': bounded_text(judge.feedback, 1000),
                }
        evidence_json = json.dumps(repair_evidence, ensure_ascii=False)
        prompt = f"""{task_context(task_spec)}

Visible repair evidence (a successful process exit does not mean the output passed;
judge concerns are advisory and do not establish a failed comparison):
{evidence_json}
{reference_context}

Revised algorithm plan (fallible):
{algorithm_plan.text}

Previous source:
{previous_candidate.raw_code}

Execution contract:
{contract}

Passed: {verification_result.passed}/{verification_result.total}
{parsing_diagnosis}
{repetition}
Implement the revised algorithm and its state invariants. Address every supplied
deterministic failure or violation, investigate advisory concerns against the task,
and verify the maximum bounds. Return ONLY the complete corrected source."""
        if edit_repair:
            return _generate_evidence_edits(client, previous_candidate, prompt, system, temperature,
                                            algorithm_plan=algorithm_plan, failure_kind='runtime or output')
        return _generate_repair(client, previous_candidate, prompt, system, temperature,
                                algorithm_plan=algorithm_plan)

    prompt = f"""{context}

Reward: {verification_result.pass_rate:.4f}
Passed: {verification_result.passed}/{verification_result.total}
{parsing_diagnosis}
Failures: {failures_json}
{reference_context}

Evaluation report: {evaluation_summary}
Feedback categories: {feedback_categories}
Candidate repair feedback:
{collected_feedback}

{repetition}

Diagnose the evidence, address every deterministic failure or violation, and return a corrected
solution following the execution contract. Treat advisory feedback as a hypothesis to verify against the task.
Keep the required function signature or input/output format unchanged.
The solution must be different from the previous one. Try a different approach if needed."""

    if edit_repair:
        return _generate_evidence_edits(client, previous_candidate, prompt, system, temperature,
                                        failure_kind='runtime or output')
    return _generate_repair(client, previous_candidate, prompt, system, temperature)


def _generate_repair(client, previous_candidate, prompt, system, temperature, *, algorithm_plan=None):
    # Use slightly higher temperature to encourage different solutions
    adjusted_temperature = min(temperature + 0.1, 0.5)
    raw = client.generate(prompt, system=system, temperature=adjusted_temperature,
                          **({'max_output_tokens': 4096} if algorithm_plan is not None
                             or getattr(client, 'supports_source_edit_repairs', False) is True else {}))
    return CodeCandidate(
        raw_code=strip_markdown_fence(raw),
        problem_type=previous_candidate.problem_type,
        prompt_version=previous_candidate.prompt_version,
        model_name=getattr(client, "model", previous_candidate.model_name),
        generation_temperature=adjusted_temperature,
        algorithm_plan=algorithm_plan.text if algorithm_plan is not None else previous_candidate.algorithm_plan,
    )


def _generate_evidence_edits(client, previous_candidate, prompt, system, temperature, *,
                             algorithm_plan=None, failure_kind='compiler'):
    """A repeated or diagnosed repair spends its next call proposing explicit edits.

    Invalid edits preserve the old candidate and carry a rejection reason, so the
    loop's existing repeat limits and fixed candidate budgets remain authoritative.
    """
    if getattr(client, 'source_edit_style', None) == 'lines':
        return _generate_line_evidence_edits(client, previous_candidate, prompt, system, temperature,
                                             algorithm_plan=algorithm_plan, failure_kind=failure_kind)
    adjusted_temperature = min(temperature + 0.1, 0.5)
    error = previous_candidate.repair_validation_error
    system += f"""

For THIS request, return ONLY JSON with diagnosis and edits, not complete source code.
This response format overrides the code-only response directions above. The execution
contract still applies to the resulting program. Resolve the supplied {failure_kind}
evidence with exact replacements for the failing statements instead of repeating the program.
Each edit has old and new strings. Copy old verbatim from the previous source, including
whitespace, with enough context to match exactly once. Use non-overlapping anchors from
that original source. Every edit must change text. Never use ellipses or Markdown fences.
Diagnose the {failure_kind} evidence and change the responsible expressions, types, APIs,
input reads or state updates; cosmetic changes cannot resolve an unchanged failure."""
    prompt += f"""

Source edit response contract (authoritative for this response):
Return {{"diagnosis": "your {failure_kind} diagnosis", "edits": [{{"old": "exact source anchor", "new": "corrected replacement"}}]}}.
These are field descriptions, not values to copy. Do not return the full program.
{('Previous source edits were rejected: ' + error) if error else ''}"""
    generate_json = getattr(client, "generate_json", None)
    if callable(generate_json):
        raw = generate_json(prompt, system=system, temperature=adjusted_temperature,
                            schema=SOURCE_EDIT_RESPONSE_SCHEMA)
    else:
        raw = client.generate(prompt, system=system, temperature=adjusted_temperature)
    try:
        code = apply_source_edits(previous_candidate.raw_code, parse_json_response(raw))
        validation_error = None
    except (TypeError, ValueError) as exc:
        code = previous_candidate.raw_code
        validation_error = str(exc)
    return CodeCandidate(
        raw_code=code,
        problem_type=previous_candidate.problem_type,
        prompt_version=previous_candidate.prompt_version,
        model_name=getattr(client, "model", previous_candidate.model_name),
        generation_temperature=adjusted_temperature,
        repair_validation_error=validation_error,
        algorithm_plan=algorithm_plan.text if algorithm_plan is not None else previous_candidate.algorithm_plan,
    )


def _generate_line_evidence_edits(client, previous_candidate, prompt, system, temperature, *,
                                  algorithm_plan=None, failure_kind='compiler'):
    """Native repairs bind line ranges to the source and preserve block indentation."""
    source = previous_candidate.raw_code
    schema = line_edit_schema(source)
    adjusted_temperature = min(temperature + 0.1, 0.5)
    system += f"""
For THIS response return ONLY JSON with diagnosis, source_sha256 and edits.
This overrides earlier code-only directions. Use inclusive start_line/end_line
ranges from the numbered ORIGINAL source below and replacement source blocks.
Keep ranges non-overlapping; every edit must change text. Replacement indentation
is relative to the original block: preserve indentation between nested statements;
the host restores its base indentation. Line numbers are labels, not source code.
Resolve the supplied {failure_kind} evidence. Select the complete connected failing
block, including affected reads, declarations and dependent uses. Check for leftover
reads, duplicate bindings or missing variables outside a replacement. Preserve the
required interface and task semantics. Do not invent outputs or hardcode test cases."""
    prompt += '\n\nNumbered ORIGINAL source for line edits:\n' + '\n'.join(
        f'{index}: {line}' for index, line in enumerate(source.splitlines(), 1))
    prompt += '\nSource SHA-256: ' + schema['properties']['source_sha256']['const']
    if previous_candidate.repair_validation_error:
        prompt += '\nPrevious edit rejected: ' + previous_candidate.repair_validation_error
    generate_json = getattr(client, 'generate_json', None)
    options = dict(system=system, temperature=adjusted_temperature, max_output_tokens=4096)
    raw = (generate_json(prompt, schema=schema, **options) if callable(generate_json)
           else client.generate(prompt, **options))
    try:
        code = apply_line_edits(source, parse_json_response(raw), relative_indentation=True)
        validation_error = None
    except (TypeError, ValueError) as exc:
        code, validation_error = source, str(exc)
    return CodeCandidate(
        raw_code=code, problem_type=previous_candidate.problem_type,
        prompt_version=previous_candidate.prompt_version,
        model_name=getattr(client, 'model', previous_candidate.model_name),
        generation_temperature=adjusted_temperature, repair_validation_error=validation_error,
        algorithm_plan=algorithm_plan.text if algorithm_plan is not None else previous_candidate.algorithm_plan,
    )
