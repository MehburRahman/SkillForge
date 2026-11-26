"""Compiler-driven JDK symbol lookup; no task solutions or fixed import table."""
from functools import lru_cache
from pathlib import Path
import re
import zipfile


@lru_cache(maxsize=4)
def _index(home, stamps):
    result={}
    for name,_,_ in stamps:
        try:
            with zipfile.ZipFile(Path(home)/'jmods'/name) as archive:
                for path in archive.namelist():
                    if not path.startswith('classes/') or not path.endswith('.class') or '$' in path:continue
                    qualified=path[len('classes/'):-len('.class')].replace('/','.')
                    if not qualified.startswith(('java.','javax.','com.sun.net.httpserver.')):continue
                    result.setdefault(qualified.rsplit('.',1)[-1],set()).add(qualified)
        except (OSError,zipfile.BadZipFile):continue
    return result


def compiler_import_hints(diagnostics, code, java_home=None):
    missing=set(re.findall(r'Not found:\s*(?:type\s+)?([A-Z]\w*)','\n'.join(diagnostics)))
    if not missing:return ''
    try:
        if java_home is None:
            from src.verifier.runners.scala import toolchain
            java_home=Path(toolchain()[1]).resolve().parent.parent
        root=Path(java_home)
        stamps=tuple((p.name,p.stat().st_mtime_ns,p.stat().st_size) for p in sorted((root/'jmods').glob('*.jmod')))
        index=_index(str(root),stamps)
    except (OSError,RuntimeError):return ''
    packages={qualified.rsplit('.',1)[0] for name in missing for qualified in index.get(name,())}
    if not packages:return ''
    # Include used sibling types to prevent replacing one necessary import with
    # another and then oscillating between the two compiler errors.
    used=set(re.findall(r'\b[A-Z]\w*\b',code))|missing
    matches={name:sorted(q for q in index.get(name,()) if q.rsplit('.',1)[0] in packages)
             for name in sorted(used)}
    matches={name:values[:6] for name,values in matches.items() if values}
    return ('JDK symbol lookup (from the installed SDK, not inferred task behavior):\n'
            + '\n'.join(name+': '+', '.join(values) for name,values in list(matches.items())[:20])
            + '\nAdd the missing import or qualify the type. Preserve imports for ALL other referenced '
              'types; importing one class does not import its siblings. Do not alternate between '
              'imports and reintroduce an earlier compiler error.\n')
