"""Deterministic, compiler-confirmed fixes for language rules small models keep breaking.

A quick fix is applied only when the compiler names the exact defect, only
renames identifiers (never changes behaviour), and is kept only if the fixed
source then compiles. Callers record every applied fix as visible evidence.
"""

import re

_PATTERN_BINDING = re.compile(r'\b(?:val|var)\s+(?:[A-Za-z_][\w.]*\s*)?\(([^)=\n]*)\)\s*(?::[^=\n]+)?=')
_NOT_FOUND = re.compile(r'Not found: ([A-Z][A-Za-z0-9_]*)\b')


def _code_spans(source):
    """Yield (start, end, is_code) spans, separating literals and comments from code."""
    index, start, length = 0, 0, len(source)
    while index < length:
        if source.startswith('"""', index):
            end = source.find('"""', index + 3)
            end = length if end < 0 else end + 3
        elif source.startswith('//', index):
            end = source.find('\n', index)
            end = length if end < 0 else end
        elif source.startswith('/*', index):
            end = source.find('*/', index + 2)
            end = length if end < 0 else end + 2
        elif source[index] in '"\'':
            quote, end = source[index], index + 1
            while end < length and source[end] not in (quote, '\n'):
                end += 2 if source[end] == '\\' else 1
            end = min(end + 1, length)
        else:
            index += 1
            continue
        if start < index:
            yield start, index, True
        yield index, end, False
        start = index = end
    if start < length:
        yield start, length, True


def _rename(source, old, new):
    pattern = re.compile(r'(?<![\w$.])' + re.escape(old) + r'(?![\w$])')
    return ''.join(pattern.sub(new, source[a:b]) if code else source[a:b]
                   for a, b, code in _code_spans(source))


def _fresh_name(source, name):
    base = name[0].lower() + name[1:]
    candidate, suffix = base, 1
    while re.search(r'(?<![\w$])' + re.escape(candidate) + r'(?![\w$])', source):
        suffix += 1
        candidate = f'{base}{suffix}'
    return candidate


def scala_uppercase_pattern_fix(source, compiler_output):
    """Rename capitalised val/var pattern bindings that the compiler cannot find.

    In Scala ``val Array(S, T) = ...`` treats S and T as existing constants, so
    every use reports ``Not found``. Returns (fixed_source, renames) or None.
    """
    missing = set(_NOT_FOUND.findall(compiler_output or ''))
    if not missing:
        return None
    bound = set()
    for group in _PATTERN_BINDING.findall(source):
        bound.update(re.findall(r'(?<![\w$.])([A-Z][A-Za-z0-9_]*)(?![\w$(])', group))
    names = sorted(missing & bound)
    if not names:
        return None
    fixed, renames = source, {}
    for name in names:
        renames[name] = _fresh_name(fixed, name)
        fixed = _rename(fixed, name, renames[name])
    return (fixed, renames) if fixed != source else None


_MUTATION_ERROR = re.compile(
    r'\.scala:(\d+):\d+[^\n]*\n(?:[^\n]*\n){0,4}?[^\n]*value (?:\+=|-=|\*=|update) is not a member of '
    r'(?:Int|Long|Double|scala\.collection\.immutable\.Map|Map)\b')


def scala_immutable_map_fix(source, compiler_output):
    """Make a locally declared Map mutable when the compiler rejects in-place updates.

    ``val m = xs.groupBy(identity).view.mapValues(_.size).toMap`` followed by
    ``m(k) -= 1`` or ``m.update(k, v)`` fails because the map is immutable.
    Returns (fixed_source, names) or None.
    """
    lines = source.split('\n')
    names = []
    for line_number in sorted({int(n) for n in _MUTATION_ERROR.findall(compiler_output or '')}):
        if not 1 <= line_number <= len(lines):
            continue
        # Allow one nested call in the key, e.g. counts(s(end)) -= 1.
        match = re.search(r'\b([A-Za-z_]\w*)\s*(?:\((?:[^()]|\([^()]*\))*\)\s*(?:\+=|-=|\*=)|\.update\s*\()',
                          lines[line_number - 1])
        if match and match.group(1) not in names:
            names.append(match.group(1))
    changed = []
    for name in names:
        declaration = re.compile(r'^(\s*(?:val|var)\s+' + re.escape(name) + r'\b\s*)(:\s*Map\b)?(.*)$')
        for index, line in enumerate(lines):
            found = declaration.match(line)
            if not found:
                continue
            head, annotation, rest = found.groups()
            new_rest = rest
            if re.search(r'\.toMap\b(?!.*\.toMap\b)', rest):
                new_rest = re.sub(r'\.toMap\b(?!.*\.toMap\b)', '.to(scala.collection.mutable.Map)', rest)
            elif re.search(r'=\s*Map\s*[\[(]', rest):
                new_rest = re.sub(r'=\s*Map(\s*[\[(])', r'= scala.collection.mutable.Map\1', rest, count=1)
            if annotation:
                annotation = ': scala.collection.mutable.Map'
            replacement = head + (annotation or '') + new_rest
            if replacement != line:
                lines[index] = replacement
                changed.append(name)
            break
    return ('\n'.join(lines), changed) if changed else None


_MISMATCH = re.compile(
    r'Type Mismatch Error: [^\n]*?\.scala:(\d+):\d+[^\n]*\n'
    r'(?P<code>\s*\d+\s*\|[^\n]*)\n(?P<caret>\s*\|\s*\^+)[^\n]*\n'
    r'\s*\|\s*Found:\s*(?P<found>[^\n]*)\n\s*\|\s*Required:\s*(?P<required>[^\n]*)')


def _is_mutable_map(text):
    return bool(re.search(r'\bmutable\.(?:Hash)?Map\b', text))


def _is_immutable_map(text):
    return bool(re.search(r'(?<![\w.])(?:Map|immutable\.Map)\s*\[', text)) and not _is_mutable_map(text)


_CONVERTIBLE = r'(?:ListBuffer|ArrayBuffer|Buffer|Array|ArraySeq|Seq|IndexedSeq|Iterable|Iterator|List|Vector|LazyList|Set|HashSet)'
_CONVERSIONS = (('List', '.toList'), ('Vector', '.toVector'), ('Array', '.toArray'),
                ('Seq', '.toSeq'), ('IndexedSeq', '.toIndexedSeq'), ('Set', '.toSet'))


def _collection_conversion(found, required):
    """Return the standard conversion (e.g. ``.toList``) from Found to Required collection, else None."""
    def outer(text):
        name = re.match(r'\s*(?:scala\.)?(?:collection\.)?(?:(mutable|immutable)\.)?([A-Za-z]+)\s*\[', text)
        return (name.group(1), name.group(2)) if name else (None, None)
    found_kind, found_name = outer(found)
    required_kind, required_name = outer(required)
    if (found_name is None or required_name is None or required_kind == 'mutable'
            or not re.fullmatch(_CONVERTIBLE, found_name)
            or (found_name == required_name and found_kind == required_kind)):
        return None
    for name, suffix in _CONVERSIONS:
        if required_name == name:
            return suffix
    return None


def scala_map_mismatch_fix(source, compiler_output):
    """Convert between immutable and mutable Maps exactly where the compiler requires it.

    Uses the caret span under the reported line, so only the flagged expression
    changes: ``expr`` becomes ``expr.to(scala.collection.mutable.Map)`` or ``expr.toMap``.
    Returns (fixed_source, count) or None.
    """
    lines = source.split('\n')
    edits = []
    for match in _MISMATCH.finditer(compiler_output or ''):
        # Scala disambiguates same-named types with superscripts (Map, Map²).
        found, required = (re.sub('[\u00b9\u00b2\u00b3\u2070-\u2079]', '', match.group(name))
                           for name in ('found', 'required'))
        if _is_mutable_map(required) and not _is_mutable_map(found) and 'Map' in found:
            suffix = '.to(scala.collection.mutable.Map)'
        elif _is_immutable_map(required) and _is_mutable_map(found):
            suffix = '.toMap'
        else:
            suffix = _collection_conversion(found, required)
            if suffix is None:
                continue
        number = int(match.group(1))
        caret = match.group('caret')
        # Scala prints "NN |<source line>" and "   |<spaces>^^^" with equal prefixes.
        start, width = caret.index('^') - caret.index('|') - 1, caret.count('^')
        if not 1 <= number <= len(lines) or start < 0:
            continue
        expression = lines[number - 1][start:start + width]
        if not re.fullmatch(r'\S(?:.*\S)?', expression):
            continue
        wrapped = expression + suffix if re.fullmatch(r'[A-Za-z_][\w.]*', expression) else f'({expression}){suffix}'
        edits.append((number - 1, start, width, wrapped))
    if not edits:
        return None
    for index, start, width, wrapped in sorted(edits, reverse=True):
        line = lines[index]
        lines[index] = line[:start] + wrapped + line[start + width:]
    return '\n'.join(lines), len(edits)


_MISMATCH_BLOCK = re.compile(r'Type Mismatch Error: [^\n]*?\.scala:(\d+):\d+[^\n]*(?P<body>(?:\n(?!--)[^\n]*)*)')
_NULLABLE_SCRUTINEE = re.compile(r'(?<![\w$.])(?P<expr>[A-Za-z_][\w$.]*\.get\((?:[^()\n]|\([^()\n]*\))*\))(?P<tail>\s+match\b)')


def _enclosing_brace_line(source, offset):
    """Return the line index of the unmatched '{' that encloses ``offset``, ignoring literals."""
    depth = 0
    spans = [(a, b) for a, b, code in _code_spans(source[:offset]) if code]
    for a, b in reversed(spans):
        for position in range(b - 1, a - 1, -1):
            if source[position] == '}':
                depth += 1
            elif source[position] == '{':
                if depth == 0:
                    return source.count('\n', 0, position)
                depth -= 1
    return None


def scala_nullable_match_fix(source, compiler_output):
    """Wrap a Java ``map.get(key)`` matched against Some/None in ``Option(...)``.

    java.util maps return null for an absent key, so Scala reports the Some/None
    case pattern as incompatible with the plain element type. The scrutinee is the
    nearest ``<receiver>.get(...) match`` above each reported pattern line.
    Returns (fixed_source, count) or None.
    """
    lines = source.split('\n')
    targets = set()
    for match in _MISMATCH_BLOCK.finditer(compiler_output or ''):
        found = re.search(r'Found:\s*([^\n]*)', match.group('body'))
        required = re.search(r'Required:\s*([^\n]*)', match.group('body'))
        if not found or not required or 'Option' in required.group(1):
            continue
        if not re.match(r'\(?(?:None\.type|Some\[)', found.group(1).strip()):
            continue
        number = int(match.group(1))
        if not 1 <= number <= len(lines):
            continue
        # The pattern belongs to the match whose brace encloses its line.
        index = _enclosing_brace_line(source, sum(len(line) + 1 for line in lines[:number - 1]))
        scrutinee = None if index is None else _NULLABLE_SCRUTINEE.search(lines[index])
        if scrutinee and re.search(r'\bmatch\s*\{\s*(?://.*)?$', lines[index][scrutinee.end('expr'):]):
            targets.add((index, scrutinee.start('expr'), scrutinee.end('expr')))
    if not targets:
        return None
    for index, start, end in sorted(targets, reverse=True):
        line = lines[index]
        lines[index] = f'{line[:start]}Option({line[start:end]}){line[end:]}'
    return '\n'.join(lines), len(targets)


_INFER_ERROR = re.compile(r'\.scala:(\d+):\d+[^\n]*\n(?:[^\n]*\n){0,3}?[^\n]*Could not infer type for parameter (\w+) of anonymous function')
_RECEIVER = re.compile(r'(?:[A-Za-z_][\w$]*(?:\(\))?)(?:\.[A-Za-z_][\w$]*(?:\(\))?)*$')


def _matching_close(source, open_index):
    """Return the index of the bracket closing source[open_index], skipping literals."""
    pairs = {'(': ')', '{': '}', '[': ']'}
    stack = []
    for start, end, code in _code_spans(source):
        if end <= open_index or not code:
            continue
        for position in range(max(start, open_index), end):
            char = source[position]
            if char in pairs:
                stack.append(pairs[char])
            elif char in ')}]':
                if not stack or stack.pop() != char:
                    return None
                if not stack:
                    return position
    return None


def _split_top_level_comma(text):
    depth = 0
    for start, end, code in _code_spans(text):
        if not code:
            continue
        for position in range(start, end):
            char = text[position]
            if char in '([{':
                depth += 1
            elif char in ')]}':
                depth -= 1
            elif char == ',' and depth == 0:
                return text[:position], text[position + 1:]
    return None


def scala_replace_all_function_fix(source, compiler_output):
    """Rewrite ``text.replaceAll(pattern, m => ...)`` as ``Regex.replaceAllIn``.

    java.lang.String.replaceAll takes a replacement String, so a function argument
    cannot be typed. The rewrite keeps pattern and function body unchanged and quotes
    the computed replacement so inserted $ and \\ stay literal. Returns (source, count) or None.
    """
    lines = source.split('\n')
    rewrites = []
    for match in _INFER_ERROR.finditer(compiler_output or ''):
        number, parameter = int(match.group(1)), match.group(2)
        if not 1 <= number <= len(lines):
            continue
        line_start = sum(len(line) + 1 for line in lines[:number - 1])
        call = lines[number - 1].find('.replaceAll(')
        if call < 0:
            continue
        receiver = _RECEIVER.search(lines[number - 1][:call])
        if receiver is None:
            continue
        open_index = line_start + call + len('.replaceAll')
        close_index = _matching_close(source, open_index)
        if close_index is None:
            continue
        arguments = _split_top_level_comma(source[open_index + 1:close_index])
        if arguments is None:
            continue
        pattern, function = arguments[0].strip(), arguments[1].strip()
        braced = function.startswith('{') and function.endswith('}')
        inner = function[1:-1].strip() if braced else function
        lambda_head = re.match(re.escape(parameter) + r'\s*=>', inner)
        if lambda_head is None or not pattern:
            continue
        body = inner[lambda_head.end():].strip()
        rewritten = (f'({pattern}).r.replaceAllIn({receiver.group(0)}, ({parameter}: scala.util.matching.Regex.Match) => '
                     f'scala.util.matching.Regex.quoteReplacement {{\n{body}\n}})')
        rewrites.append((line_start + receiver.start(), close_index + 1, rewritten))
    if not rewrites:
        return None
    for start, end, text in sorted(set(rewrites), reverse=True):
        source = source[:start] + text + source[end:]
    return source, len(set(rewrites))


_JAVA_QUANTIFIER = re.compile(r'\{\d+(?:,\d*)?\}')
_REGEX_LITERAL = re.compile(r'(?P<quote>"""|")(?P<body>(?:(?!(?P=quote))[^\n]|\n(?=[^"]*"""))*?)(?P=quote)'
                            r'(?=\s*\.r\b)|(?<=Pattern\.compile\()\s*(?P<quote2>"""|")(?P<body2>.*?)(?P=quote2)')


def _escape_java_braces(pattern, *, raw):
    """Escape '{' that cannot start a Java quantifier {n}, {n,} or {n,m}, and its partner '}'."""
    slash = '\\' if raw else '\\\\'
    out, index, escaped_open = [], 0, 0
    while index < len(pattern):
        char = pattern[index]
        if pattern.startswith(slash, index):
            out.append(pattern[index:index + len(slash) + 1])
            index += len(slash) + 1
            continue
        if char == '[':
            close = pattern.find(']', index + 2)
            if close > 0:
                out.append(pattern[index:close + 1])
                index = close + 1
                continue
        if char == '{' and not _JAVA_QUANTIFIER.match(pattern, index):
            out.append(slash + '{')
            escaped_open += 1
        elif char == '}' and escaped_open:
            out.append(slash + '}')
            escaped_open -= 1
        else:
            out.append(char)
        index += 1
    return ''.join(out)


def scala_regex_brace_fix(source):
    """Escape literal braces in Scala regex literals that Java rejects as 'Illegal repetition'.

    java.util.regex treats every unescaped '{' as a repetition count, so a pattern
    such as {{(\\w+)}} compiles in Scala but throws PatternSyntaxException at run
    time, often inside a server thread where the error is never reported.
    Returns (fixed_source, count) or None.
    """
    edits = []
    for match in _REGEX_LITERAL.finditer(source):
        name = 'body' if match.group('body') is not None else 'body2'
        quote = match.group('quote') or match.group('quote2')
        body = match.group(name)
        fixed = _escape_java_braces(body, raw=quote == '"""')
        if fixed != body:
            edits.append((match.start(name), match.end(name), fixed))
    if not edits:
        return None
    for start, end, text in reversed(edits):
        source = source[:start] + text + source[end:]
    return source, len(edits)


_REPLACE_ALL_IN = re.compile(r'\.replaceAllIn\s*\(')
_LAMBDA_HEAD = re.compile(r'\s*(?:\{\s*)?(?:\(\s*)?(?:[A-Za-z_][\w$]*|_)(?:\s*:\s*[\w$.\[\]]+)?(?:\s*\))?\s*=>')


def scala_quote_replacement_fix(source):
    """Quote the text a ``Regex.replaceAllIn`` function returns.

    replaceAllIn reads $n and \\ in the returned String as group references and
    escapes, so inserting user data such as "$1" or "C:\\temp" throws or corrupts the
    output. Wrapping the function body in Regex.quoteReplacement keeps it literal.
    Returns (fixed_source, count) or None.
    """
    edits = []
    code_spans = [(start, end) for start, end, code in _code_spans(source) if code]
    for match in _REPLACE_ALL_IN.finditer(source):
        if not any(start <= match.start() < end for start, end in code_spans):
            continue
        open_index = match.end() - 1
        close_index = _matching_close(source, open_index)
        if close_index is None:
            continue
        inside = source[open_index + 1:close_index]
        arguments = _split_top_level_comma(inside)
        if arguments is None:
            continue
        function_start = open_index + 1 + len(arguments[0]) + 1
        function = arguments[1]
        head = _LAMBDA_HEAD.match(function)
        if head is None or 'quoteReplacement' in function:
            continue
        body_start = function_start + head.end()
        body_end = close_index
        if function.lstrip().startswith('{'):
            # { m => body }: keep the closing brace of the block outside the wrap.
            body_end = function_start + function.rstrip().rfind('}')
        body = source[body_start:body_end]
        if not body.strip():
            continue
        trailing = body[len(body.rstrip()):]
        block_open = body_start + len(body) - len(body.lstrip())
        if source[block_open] == '{' and _matching_close(source, block_open) == body_start + len(body.rstrip()) - 1:
            wrapped = f' scala.util.matching.Regex.quoteReplacement {body.strip()}{trailing}'
        else:
            wrapped = f' scala.util.matching.Regex.quoteReplacement {{{body.rstrip()}\n}}{trailing}'
        edits.append((body_start, body_end, wrapped))
    kept, last_start = [], None
    for start, end, text in sorted(edits, reverse=True):
        if last_start is not None and end > last_start:
            continue  # nested call: the outer rewrite already covers it
        kept.append((start, end, text))
        last_start = start
    if not kept:
        return None
    for start, end, text in kept:
        source = source[:start] + text + source[end:]
    return source, len(kept)


_PRIVATE_ACCESS = re.compile(
    r'(?:variable|value|method|lazy value) (?P<name>[A-Za-z_][\w$]*) cannot be accessed as a member of[^\n]*'
    r'(?:\n[^\n]*){0,3}?\n[^\n]*private (?:variable|value|method|lazy value) (?P=name) can only be accessed from '
    r'(?:class|object|trait) (?P<owner>[A-Za-z_][\w$]*)')


def scala_private_access_fix(source, compiler_output):
    """Drop ``private`` from a member that another class reads, as the compiler reports.

    Widening visibility never changes behaviour; it only lets the caller compile.
    Returns (fixed_source, names) or None.
    """
    names = []
    for match in _PRIVATE_ACCESS.finditer(compiler_output or ''):
        name = match.group('name')
        pattern = re.compile(r'\bprivate(?:\[\w+\])?\s+((?:lazy\s+)?(?:val|var|def)\s+' + re.escape(name) + r'\b)')
        fixed = pattern.sub(r'\1', source)
        if fixed != source:
            source = fixed
            names.append(name)
    return (source, names) if names else None


def compiler_quick_fix(source, compiler_output, *, language):
    """Return (fixed_source, description) for a known compiler defect, else None."""
    if language != 'scala':
        return None
    result = scala_uppercase_pattern_fix(source, compiler_output)
    if result is not None:
        fixed, renames = result
        listed = ', '.join(f'{old}→{new}' for old, new in renames.items())
        return fixed, f'Renamed capitalised Scala pattern bindings ({listed}); Scala reads them as existing constants.'
    result = scala_immutable_map_fix(source, compiler_output)
    if result is not None:
        fixed, names = result
        return fixed, (f'Declared {", ".join(names)} as scala.collection.mutable.Map; '
                       'the code updates it in place, which an immutable Map does not allow.')
    result = scala_private_access_fix(source, compiler_output)
    if result is not None:
        fixed, names = result
        return fixed, (f'Made {", ".join(names)} visible to the classes that read it; the compiler '
                       'rejected access to a private member.')
    result = scala_map_mismatch_fix(source, compiler_output)
    if result is not None:
        fixed, count = result
        return fixed, (f'Converted {count} collection value(s) to the type the compiler required '
                       '(for example ListBuffer to List, or mutable to immutable Map).')
    result = scala_replace_all_function_fix(source, compiler_output)
    if result is not None:
        fixed, count = result
        return fixed, (f'Rewrote {count} String.replaceAll call(s) with a function argument as '
                       'Regex.replaceAllIn; String.replaceAll accepts only a replacement String.')
    result = scala_nullable_match_fix(source, compiler_output)
    if result is not None:
        fixed, count = result
        return fixed, (f'Wrapped {count} Java map lookup(s) in Option(...); java.util maps return null '
                       'for absent keys, so Some/None patterns need an Option.')
    return None
