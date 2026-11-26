"""Execution contracts; existing tasks remain Python stdio tasks."""

from enum import Enum
import keyword
import re

from src.benchmark.language import Language


JS_RESERVED_WORDS = frozenset("""
await break case catch class const continue debugger default delete do else enum
export extends false finally for function if implements import in instanceof
interface let new null package private protected public return static super
switch this throw true try typeof var void while with yield
""".split())


class ExecutionMode(str, Enum):
    STDIO = "stdio"
    FUNCTION = "function"
    HTTP = "http"

    def __str__(self):
        return self.value


def suggested_execution_mode(statement, language='python'):
    """Conservative interface hint; explicitly supplied modes take precedence."""
    if (language == 'scala' and isinstance(statement, str)
            and re.search(r'\bhttps?\s+(?:web\s+)?(?:server|service)\b', statement, re.I)):
        return 'http'
    return 'stdio'


def validate_execution(language="python", execution_mode="stdio", entry_point=None):
    try:
        language = Language(language)
    except (TypeError, ValueError) as exc:
        raise ValueError("language must be one of: " + ", ".join(Language)) from exc
    try:
        execution_mode = ExecutionMode(execution_mode)
    except (TypeError, ValueError) as exc:
        raise ValueError("execution_mode must be stdio, function, or http") from exc
    if execution_mode == ExecutionMode.HTTP and language != Language.SCALA:
        raise ValueError("HTTP execution currently supports Scala only")
    if execution_mode != ExecutionMode.FUNCTION:
        if entry_point is not None:
            raise ValueError("entry_point is only supported in function mode")
    else:
        if not isinstance(entry_point, str) or not entry_point:
            raise ValueError("function mode requires a non-empty entry_point")
        if language == Language.PYTHON:
            valid = entry_point.isidentifier() and not keyword.iskeyword(entry_point)
        else:
            # A bare function name, never an expression interpolated into code.
            valid = re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", entry_point) is not None
            if language == Language.JAVASCRIPT:
                valid = valid and entry_point not in JS_RESERVED_WORDS
            elif language == Language.SCALA:
                reserved = JS_RESERVED_WORDS | frozenset("abstract case catch class def do else enum export extends false final finally for forSome given if implicit import inline lazy match new null object opaque override package private protected return sealed super then this throw trait transparent true try type val var while with yield end extension infix using derives".split())
                valid = valid and entry_point not in reserved and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", entry_point) is not None
        if not valid:
            raise ValueError("entry_point must be a function identifier")
    return language, execution_mode, entry_point
