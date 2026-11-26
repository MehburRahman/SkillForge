"""Shared generation/repair contracts, independent of the registered strategy."""

import hashlib
import json
from pathlib import Path

from src.benchmark.execution_mode import ExecutionMode, validate_execution
from src.benchmark.language import Language


PROMPT_TEMPLATE_VERSION = "compile-first-grounded-review-v18"
# Adapt only the exact shipped baseline, never rewrite arbitrary evolved text.
# The registry record and its scientific identifier remain immutable.
_LEGACY_BASELINE = (
    "You are a competitive programmer. Output ONLY a complete, runnable Python solution.\n"
    "No explanation, no markdown fences, no comments about your reasoning. Just the code."
)
_BASELINE_STRATEGY = "You are a competitive programmer. Solve the task correctly and efficiently."


def execution_contract(language="python", execution_mode="stdio", entry_point=None):
    language, mode, entry_point = validate_execution(language, execution_mode, entry_point)
    if language == Language.PYTHON:
        runtime = "Python 3. Use only the Python standard library."
        stdin = "Read the actual input from standard input using input() or sys.stdin."
    elif language == Language.JAVASCRIPT:
        runtime = (
            "JavaScript running in Node.js (CommonJS). Use only Node.js built-ins; "
            "no browser APIs, TypeScript, or ES-module import/export syntax."
        )
        stdin = "Read the actual input from standard input using require('node:fs').readFileSync(0, 'utf8')."
    elif language == Language.SCALA:
        runtime = (
            "Scala 3.3.8 on JDK 21. Use only Scala/JDK standard libraries; no build directives or external dependencies. "
            "Map and Set are immutable by default: var permits rebinding, not element mutation. "
            "Use scala.collection.mutable collections for in-place updates, or reassign updated immutable values. "
            "Arrays allow indexed updates; maps use keys and require an explicit default for absent keys. "
            "Account for copying in substring/slice and collection operations inside loops; store indices "
            "and build the final result once when repeated copies would violate the required complexity. "
            "Write regular expressions as triple-quoted raw strings (\"\"\"\\d+\"\"\".r) so backslashes need no escaping. "
            "String.replaceAll takes two Strings; to compute each replacement use "
            "regex.replaceAllIn(text, m => scala.util.matching.Regex.quoteReplacement(f(m.group(1))))."
        )
        stdin = (
            "Define object Main with def main(args: Array[String]): Unit. Parse standard input exactly as specified. "
            "For whitespace-separated tokens, accept both spaces and newlines using a token parser. "
            "scala.io.StdIn.readInt/readLong consume an entire line, not one whitespace-separated token. "
            "StdIn.readLine returns only one line and removes its newline; splitting that result cannot read later lines. "
            "To read line-based input use: val lines = scala.io.Source.stdin.getLines().toList "
            "(one element per input line); call readLine once per line only when the line count is fixed. "
            "Read all required records, preserving line boundaries, spaces and valid empty records. "
            "When splitting complete input, preserve trailing empty fields if the format allows them. "
            "Use lowercase identifiers for destructuring pattern bindings; uppercase patterns refer to stable identifiers."
        )
    else:
        raise ValueError(f"{language} generation prompts are not implemented")

    if mode == ExecutionMode.STDIO:
        interface = (
            f"Execution mode: stdio. Return a complete runnable program. When the task specifies input:\n{stdin}\n"
            "If the task has no input, do not read stdin.\n"
            "Write only the required answer to standard output, following the task's input/output formats."
        )
    elif mode == ExecutionMode.HTTP:
        interface = (
            "Execution mode: http. Define object Main with def main(args: Array[String]): Unit. "
            "Implement the HTTP service specified by the problem using com.sun.net.httpserver.HttpServer. "
            "Bind to 127.0.0.1 using the port explicitly required by the task; otherwise use port 0 "
            "so the operating system chooses a free port. "
            "After server.start(), write server.getAddress.getPort.toString as UTF-8 text to the file "
            "named by sys.env.get(\"SKILLFORGE_PORT_FILE\") using java.nio.file.Files.writeString ONLY when that optional variable exists. "
            "The same program must start normally without any SKILLFORGE_* environment variables. "
            "A concurrency requirement applies to every shared mutable field, not merely a concurrent collection. "
            "Protect shared state with appropriate synchronization or atomic operations; do not assume request order. "
            "Input streams are consumable: read the request body once, retain those bytes, and decode that array. "
            "Calling readAllBytes consumes the stream; a later read cannot recover the same content. "
            "java.util maps (ConcurrentHashMap, HashMap) return null from get for an absent key, never Option; "
            "write Option(map.get(key)) before matching Some/None. "
            "Represent absent state separately from valid empty values. An empty string is not a reliable missing-state sentinel. "
            "To decode an application/x-www-form-urlencoded body: body.split(\"&\").filter(_.nonEmpty).map { pair => "
            "val i = pair.indexOf('='); val (k, v) = if (i < 0) (pair, \"\") else (pair.substring(0, i), pair.substring(i + 1)); "
            "java.net.URLDecoder.decode(k, java.nio.charset.StandardCharsets.UTF_8) -> "
            "java.net.URLDecoder.decode(v, java.nio.charset.StandardCharsets.UTF_8) }.toMap. "
            "Keep serving requests until the harness terminates the process. "
            "Implement the task's exact routes, methods, status codes, headers, and body behavior; "
            "do not invent endpoints or application requirements. The harness sends UTF-8 requests. "
            "Decode request bytes explicitly as UTF-8. Encode each response as UTF-8 bytes and use "
            "the byte count, not String.length, for sendResponseHeaders. Send response headers BEFORE "
            "writing any response-body bytes. Close each exchange. "
            "In JDK HttpServer, response length 0 means chunked streaming, not an empty body. "
            "Use length -1 for a bodyless response, and close the exchange on every branch, including errors. "
            "HTTP 204 and 304 forbid response bodies; do not choose them when a response body is required. "
            "Keep shared state in a scope accessible to all handlers, or pass it through constructors; "
            "a top-level handler cannot access an object's private fields. "
            "Handle multiple requests in the same process without unintended state leakage."
        )
    else:
        interface = (
            f"Execution mode: function. Define a top-level synchronous function named exactly {entry_point}.\n"
            "Use the parameter signature specified by the task. The harness calls this function with native "
            "arguments and checks its native return value. Return the result; do not serialize it for stdin/stdout.\n"
            "Do not read standard input, print the answer, or add top-level entry-point calls or test invocations.\n"
            "Recursive calls inside the solution are allowed unless the task forbids recursion."
        )
    if mode == ExecutionMode.FUNCTION and language == Language.SCALA:
        interface = (
            f"Execution mode: function. Define object Solution containing def {entry_point} with exactly "
            "the parameter and return types specified by the task. The harness imports this method "
            "and uses native Scala assertions. Do not add main, read stdin, or print results. "
            "Return native Scala values, not serialized text."
        )
    return (
        f"Target language: {runtime}\n{interface}\n"
        "Implement the general solution; do not hardcode example inputs or answers. "
        "Do not include demonstration calls, debug output, or an example-running driver.\n"
        "Return ONLY solution code, without explanations or Markdown fences."
    )


def generation_system(strategy, contract):
    if strategy.strip() == _LEGACY_BASELINE:
        strategy = _BASELINE_STRATEGY
    return (
        "Use the registered strategy below only where it is consistent with the execution contract. "
        "The execution contract overrides any language, interface, or output-format directions in the strategy.\n\n"
        f"Registered strategy:\n{strategy}\n\nExecution contract (authoritative):\n{contract}"
    )


def task_context(task_spec):
    """Supply identical task requirements to initial generation and both repair arms."""
    requirements = {
        "constraints": task_spec.constraints,
        "input_format": (getattr(task_spec, "input_format", "")
                         if getattr(task_spec, "input_format_origin", "unknown") != "stated" else ""),
        "output_format": (getattr(task_spec, "output_format", "")
                          if getattr(task_spec, "output_format_origin", "unknown") != "stated" else ""),
        "expected_complexity": getattr(task_spec, "expected_complexity", ""),
        "prohibited_operations": getattr(task_spec, "prohibited_operations", []),
        "required_operations": getattr(task_spec, "required_operations", []),
        "edge_cases": getattr(task_spec, "edge_cases", []),
    }
    return (
        f"Problem:\n{task_spec.problem_statement}\n\n"
        "The original problem above and the execution contract are authoritative for inputs, outputs, "
        "interfaces, and constraints. The structured analysis and generated examples below are fallible "
        "model suggestions: resolve conflicts and omissions using the original problem, and never "
        "invent requirements from an example.\n\n"
        "When the original problem leaves input/output encoding unspecified, use the "
        "formats selected by task analysis consistently. These choices specify delivery "
        "only and do not change the task semantics.\n\n"
        f"Structured requirements: {json.dumps(requirements, ensure_ascii=False, separators=(',', ':'))}\n"
        f"Examples: {json.dumps(task_spec.examples, ensure_ascii=False, separators=(',', ':'))}"
    )


def prompt_template_identity():
    """Invalidate resume when any generation/repair prompt implementation changes."""
    directory = Path(__file__).parent
    sources = [
        (directory / name).read_bytes()
        for name in ("code_prompts.py", "code_generator.py", "refinement_agent.py", "algorithm_planner.py",
                     "test_generator.py", "http_test_generator.py", "test_review.py", "test_oracle.py", "test_builder.py", "differential.py", "source_edits.py",
                     "llm_client.py", "task_spec.py", "task_spec_cache.py", "sdk_hints.py")
    ]
    sources.extend(
        (directory.parent / "feedback" / name).read_bytes()
        for name in ("collector.py", "evidence.py", "repair_context.py")
    )
    return {
        "version": PROMPT_TEMPLATE_VERSION,
        "sha256": hashlib.sha256(b"\0".join(sources)).hexdigest(),
    }
