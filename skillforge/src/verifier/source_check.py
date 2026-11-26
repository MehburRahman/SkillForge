"""Check source without executing a candidate or inventing task input."""

from pathlib import Path
import sys
import tempfile

from src.benchmark.language import Language
from src.verifier.runners import get_runner
from src.verifier.runners.base import Status
from src.verifier.runners.process import execute


def check_source(code, *, language='python', on_progress=None):
    language = Language(language)
    if language == Language.SCALA:
        return get_runner(language).run_many(
            code, [''], timeout=5, execution_mode='stdio', entry_point=None,
            compile_only=True, on_progress=on_progress,
        )[0]
    with tempfile.TemporaryDirectory(prefix='skillforge-source-check-') as directory:
        source = Path(directory) / ('candidate.py' if language == Language.PYTHON else 'candidate.cjs')
        source.write_text(code, encoding='utf-8')
        command = ([sys.executable, '-m', 'py_compile', str(source)]
                   if language == Language.PYTHON else ['node', '--check', str(source)])
        result = execute(command, '', directory, 10)
        if result.status == Status.ERROR:
            result.status = Status.COMPILE_ERROR
        return result
