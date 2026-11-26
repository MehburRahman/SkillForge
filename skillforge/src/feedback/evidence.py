"""Bound visible repair evidence while retaining actionable diagnostic locations."""
import re

MAX_FEEDBACK_VALUE_CHARACTERS = 500
MAX_DIAGNOSTIC_CHARACTERS = 1400
_ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
_SCALA_COMPILER = re.compile(r'^-- (?:\[E\d+\] |Error:)', re.MULTILINE)


def bounded_text(value, limit=MAX_FEEDBACK_VALUE_CHARACTERS):
    text = str(value)
    return text if len(text) <= limit else text[:limit] + '...[truncated]'


def bounded_diagnostic(value):
    text = _ANSI.sub('', str(value))
    # Scala CLI's offline notice can consume the excerpt before the actual
    # compiler error. Keep the error's source/caret/required type together.
    compiler = _SCALA_COMPILER.search(text)
    if compiler:
        text = text[compiler.start():]
        text = re.sub(r'/(?:[^\s/:]+/)+([^\s/:]+\.scala)(?=:)', r'\1', text)
        text = re.sub(r'^.*longer explanation available.*\n?', '', text, flags=re.MULTILINE)
    if len(text) <= MAX_DIAGNOSTIC_CHARACTERS:
        return text
    if compiler:
        # Keep distinct errors intact. Dropping a whole block is preferable to
        # losing the required type or API name halfway through a diagnostic.
        blocks = re.split(r'(?=^-- (?:\[E\d+\] |Error:))', text, flags=re.MULTILINE)
        kept, seen, size = [], set(), 0
        for block in blocks:
            block = re.sub(r'^.*longer explanation available.*\n?', '', block, flags=re.MULTILINE).strip()
            if not block or block in seen:
                continue
            seen.add(block)
            if size + len(block) + 1 <= MAX_DIAGNOSTIC_CHARACTERS:
                kept.append(block)
                size += len(block) + 1
        if kept:
            return '\n'.join(kept) + '\n[additional compiler output omitted]'
    # Stack traces put the exception at the start and candidate frame/cause
    # near the end. Retain both within a fixed budget.
    head = MAX_DIAGNOSTIC_CHARACTERS * 2 // 3
    tail = MAX_DIAGNOSTIC_CHARACTERS - head
    return text[:head] + '\n...[truncated]...\n' + text[-tail:]


def compiler_summary(value):
    """General API/type failure evidence, excluding source lines and task data."""
    text = _ANSI.sub('', str(value))
    messages = []
    for line in text.splitlines():
        message = re.sub(r'^\s*\|\s*', '', line).strip()
        if re.search(r'Not found:|is not a member of|constructor .*cannot be accessed|Found:|Required:', message):
            message = re.sub(r'"[^"\n]*"', '<literal>', message)
            if message not in messages:
                messages.append(message[:200])
    return '; '.join(messages[:4])


def bounded_failures(failures, *, include_diagnostics=True):
    evidence, seen_diagnostics = [], set()
    for failure in failures[:3]:
        item = {}
        for key in ('input', 'expected', 'actual', 'status', 'reason', 'category', 'exit_code'):
            if key in failure:
                value = failure[key]
                item[key] = value if isinstance(value, (int, float, bool)) or value is None else bounded_text(value)
        if include_diagnostics and failure.get('stderr'):
            diagnostic = bounded_diagnostic(failure['stderr'])
            if diagnostic not in seen_diagnostics:
                item['stderr'] = diagnostic
                seen_diagnostics.add(diagnostic)
        evidence.append(item)
    return evidence
