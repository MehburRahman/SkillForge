"""Apply model-proposed edits using exact anchors or hash-bound line ranges."""

import hashlib
import re
import textwrap

from src.agents.compiler_quick_fixes import _code_spans


SOURCE_EDIT_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["diagnosis", "edits"],
    "properties": {
        "diagnosis": {"type": "string", "minLength": 1},
        "edits": {
            "type": "array", "minItems": 1, "maxItems": 32,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["old", "new"],
                "properties": {"old": {"type": "string", "minLength": 1},
                               "new": {"type": "string"}},
            },
        },
    },
}


_CLOSERS = {')': '(', ']': '[', '}': '{'}


def delimiter_problem(source):
    """Describe the first unbalanced (), [] or {} outside literals and comments, else None."""
    stack = []
    for start, end, code in _code_spans(source):
        if not code:
            continue
        for position in range(start, end):
            char = source[position]
            if char in '([{':
                stack.append((char, position))
            elif char in _CLOSERS:
                if not stack or stack[-1][0] != _CLOSERS[char]:
                    return f"unexpected '{char}' on line {source.count(chr(10), 0, position) + 1}"
                stack.pop()
    if stack:
        char, position = stack[-1]
        return f"'{char}' opened on line {source.count(chr(10), 0, position) + 1} is never closed"
    return None


def _check_delimiters(source, result):
    """Reject a patch that breaks the bracket structure of previously balanced source."""
    if delimiter_problem(source) is None:
        problem = delimiter_problem(result)
        if problem:
            raise ValueError(
                f'the patched source no longer has balanced brackets ({problem}); a replacement '
                'must keep every (, [ and { it opens or removes matched with its partner. Extend '
                'start_line/end_line over the whole block you rewrite')


def apply_source_edits(source, response):
    """Validate every anchor against the original source before any replacement."""
    if not isinstance(response, dict) or set(response) != {"diagnosis", "edits"}:
        raise ValueError("source repair must contain diagnosis and edits")
    if not isinstance(response["diagnosis"], str) or not response["diagnosis"].strip():
        raise ValueError("source repair must explain the compiler error")
    edits = response["edits"]
    if not isinstance(edits, list) or not 1 <= len(edits) <= 32:
        raise ValueError("source repair needs between 1 and 32 edits")
    replacements = []
    for edit in edits:
        if not isinstance(edit, dict) or set(edit) != {"old", "new"}:
            raise ValueError("each source edit must contain old and new")
        old, new = edit["old"], edit["new"]
        if not isinstance(old, str) or not old or not isinstance(new, str):
            raise ValueError("source edit anchors must be non-empty text and replacements must be text")
        if old == new:
            raise ValueError("source edit makes no change")
        start = source.find(old)
        if start < 0 or source.find(old, start + 1) >= 0:
            raise ValueError("source edit anchor must occur exactly once in the previous source")
        replacements.append((start, start + len(old), new))
    replacements.sort()
    for left, right in zip(replacements, replacements[1:]):
        if left[1] > right[0]:
            raise ValueError("source edit anchors overlap")
    result = source
    for start, end, new in reversed(replacements):
        result = result[:start] + new + result[end:]
    if result.strip() == source.strip() or not result.strip():
        raise ValueError("source repair must produce a changed, non-empty candidate")
    _check_delimiters(source, result)
    return result


def line_edit_schema(source):
    """Tie numbered edits to the exact source supplied in the repair request."""
    count = len(source.splitlines())
    return {
        'type': 'object', 'additionalProperties': False,
        'required': ['diagnosis', 'source_sha256', 'edits'],
        'properties': {
            'diagnosis': {'type': 'string', 'minLength': 1},
            'source_sha256': {'const': hashlib.sha256(source.encode()).hexdigest()},
            'edits': {'type': 'array', 'minItems': 1, 'maxItems': 32, 'items': {
                'type': 'object', 'additionalProperties': False,
                'required': ['start_line', 'end_line', 'replacement'],
                'properties': {
                    'start_line': {'type': 'integer', 'minimum': 1, 'maximum': count},
                    'end_line': {'type': 'integer', 'minimum': 1, 'maximum': count},
                    'replacement': {'type': 'string'},
                },
            }},
        },
    }


_TRIVIAL_LINE = re.compile(r'^[\s{}()\[\];,]*$')
DUPLICATED_CONTEXT_LINES = 3


def _meaningful(lines):
    return [line.strip() for line in lines if not _TRIVIAL_LINE.match(line)]


def _contains_run(haystack, needle):
    width = len(needle)
    return any(haystack[index:index + width] == needle for index in range(len(haystack) - width + 1))


def _duplicated_context(lines, start, end, replacement):
    """Name kept source lines that a replacement repeats next to its own range.

    A replacement that re-includes the code just after (or before) its range
    leaves two copies, e.g. replacing only the method header with the whole
    method. Braces and blank lines are ignored so ordinary block edits pass.
    """
    replaced = _meaningful(replacement.splitlines())
    if len(replaced) < DUPLICATED_CONTEXT_LINES:
        return None
    after = _meaningful(lines[end:])[:DUPLICATED_CONTEXT_LINES]
    before = _meaningful(lines[:start])[-DUPLICATED_CONTEXT_LINES:]
    for name, context in (('after', after), ('before', before)):
        if len(context) == DUPLICATED_CONTEXT_LINES and _contains_run(replaced, context):
            return name
    return None


def restore_literal_newlines(text):
    """Re-escape raw newlines that JSON decoding placed inside one-line literals.

    Models often write ``"split('\\n')"`` in a JSON replacement meaning the two
    characters backslash and n; JSON decodes that to a real line break inside a
    character or string literal, which no supported language accepts. Comments,
    triple-quoted strings and template literals are copied unchanged.
    """
    out, index, quote, length = [], 0, None, len(text)
    while index < length:
        char = text[index]
        if quote is None:
            if text.startswith(('"""', "'''"), index):
                end = text.find(text[index:index + 3], index + 3)
                end = length if end < 0 else end + 3
            elif text.startswith('//', index) or char == '#':
                end = text.find('\n', index)
                end = length if end < 0 else end
            elif text.startswith('/*', index):
                end = text.find('*/', index + 2)
                end = length if end < 0 else end + 2
            elif char == '`':
                end = text.find('`', index + 1)
                end = length if end < 0 else end + 1
            else:
                if char in '"\'':
                    quote = char
                out.append(char)
                index += 1
                continue
            out.append(text[index:end])
            index = end
            continue
        if char == '\\' and index + 1 < length:
            out.append(text[index:index + 2])
            index += 2
            continue
        if char == quote:
            quote = None
        out.append('\\n' if char == '\n' else char)
        index += 1
    return ''.join(out)


def apply_line_edits(source, response, *, relative_indentation=False):
    if not isinstance(response, dict) or set(response) != {'diagnosis', 'source_sha256', 'edits'}:
        raise ValueError('line repair must contain diagnosis, source_sha256 and edits')
    if not isinstance(response['diagnosis'], str) or not response['diagnosis'].strip():
        raise ValueError('line repair must explain the failure')
    if response['source_sha256'] != hashlib.sha256(source.encode()).hexdigest():
        raise ValueError('line repair belongs to different source')
    edits = response['edits']
    if not isinstance(edits, list) or not 1 <= len(edits) <= 32:
        raise ValueError('line repair needs between 1 and 32 edits')
    lines = source.splitlines(keepends=True)
    replacements = []
    for edit in edits:
        if not isinstance(edit, dict) or set(edit) != {'start_line', 'end_line', 'replacement'}:
            raise ValueError('line edit fields do not match the required schema')
        start, end, replacement = edit['start_line'], edit['end_line'], edit['replacement']
        if type(start) is not int or type(end) is not int or not 1 <= start <= end <= len(lines):
            raise ValueError('line edit range is outside the supplied source')
        if not isinstance(replacement, str):
            raise ValueError('line replacement must be text')
        replacement = restore_literal_newlines(replacement)
        if replacement and relative_indentation:
            first_line = lines[start - 1]
            indentation = first_line[:len(first_line) - len(first_line.lstrip(' \t'))]
            replacement = textwrap.indent(textwrap.dedent(replacement), indentation)
        duplicated = _duplicated_context(lines, start - 1, end, replacement)
        if duplicated:
            raise ValueError(
                f'line edit {start}-{end} repeats source lines that remain {duplicated} its range, '
                'which would duplicate code; extend start_line/end_line to cover every line the '
                'replacement rewrites, or replace only the changed lines')
        if replacement and end < len(lines) and not replacement.endswith('\n'):
            replacement += '\n'
        # No-op ranges are harmless alongside real edits. Keep them in range
        # validation so overlaps remain invalid; reject a wholly unchanged
        # candidate after applying the complete patch.
        replacements.append((start - 1, end, replacement))
    replacements.sort()
    if any(left[1] > right[0] for left, right in zip(replacements, replacements[1:])):
        raise ValueError('line edit ranges overlap')
    for start, end, replacement in reversed(replacements):
        lines[start:end] = [replacement]
    result = ''.join(lines)
    if not result.strip() or result == source:
        raise ValueError('line repair must produce changed, non-empty source')
    _check_delimiters(source, result)
    return result
