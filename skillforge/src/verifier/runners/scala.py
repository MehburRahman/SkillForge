"""Pinned Scala compilation, native assertion harnesses, and HTTP scenarios."""
from dataclasses import replace
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from src.benchmark.execution_mode import ExecutionMode
from src.verifier.runners.base import ExecutionResult, Status, _normalize_input
from src.verifier.runners.process import execute

SCALA_VERSION = '3.3.8'
SCALA_CLI_VERSION = '1.18.0'
JDK_VERSION = '21.0.13'
JVM_VERSION = '21'
COMPILE_TIMEOUT = 180
HEAP_MB = 256
HTTP_IMAGE = 'skillforge-scala-http:1'
READY_FILE = Path(__file__).resolve().parents[3] / 'data' / 'scala_toolchain_ready.json'
HTTP_RUNNER_FILES = ('runners/scala.py', 'runners/process.py', 'runners/http.py',
                     'runners/base.py', 'http_contract.py')


def http_runner_hash():
    return hashlib.sha256(b''.join((Path(__file__).parents[1] / name).read_bytes()
                                   for name in HTTP_RUNNER_FILES)).hexdigest()


@lru_cache(maxsize=4)
def _check_container_runner(image_id, expected_hash):
    program = ('import hashlib,pathlib; root=pathlib.Path("/app/src/verifier"); '
               f'print(hashlib.sha256(b"".join((root/name).read_bytes() for name in {HTTP_RUNNER_FILES!r})).hexdigest())')
    probe = subprocess.run(['docker', 'run', '--rm', '--network', 'none', '--read-only',
                            '--user', '65534:65534', image_id, 'python3', '-c', program],
                           capture_output=True, text=True, check=True, timeout=15)
    if probe.stdout.strip() != expected_hash:
        raise RuntimeError('Infrastructure error: HTTP Docker verifier is stale. Rebuild from skillforge: '
                           'docker build -t skillforge-scala-http:1 -f tools/scala-http.Dockerfile .')


def toolchain():
    cli = shutil.which('scala-cli')
    if not cli:
        raise RuntimeError('Scala CLI is missing. Install scala-cli, then run python tools/setup_scala.py.')
    try:
        cli_version = subprocess.run([cli, 'version', '--cli-version'], capture_output=True,
                                     text=True, timeout=10, check=True).stdout.strip()
        if cli_version != SCALA_CLI_VERSION:
            raise RuntimeError(f'Install Scala CLI {SCALA_CLI_VERSION}; found {cli_version}.')
        homes = [os.environ.get('SKILLFORGE_JAVA_HOME', ''), '/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home',
                 '/usr/local/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home', os.environ.get('JAVA_HOME', '')]
        java = next((Path(h) / 'bin' / 'java' for h in homes if h and (Path(h) / 'bin' / 'java').is_file()), None)
        if java is None:
            raise RuntimeError('Install JDK 21 and set SKILLFORGE_JAVA_HOME to its installation directory.')
        probe = subprocess.run([str(java), '-version'], capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError('Scala toolchain is not ready. Run python tools/setup_scala.py first.') from exc
    if not re.search(r'version "' + re.escape(JDK_VERSION) + r'(?:[.+"-])', probe.stderr + probe.stdout):
        raise RuntimeError(f'SKILLFORGE_JAVA_HOME must select JDK {JDK_VERSION}.')
    return cli, str(java), (probe.stderr or probe.stdout).strip()


def readiness_identity():
    cli, java, version = toolchain()
    return {'scala': SCALA_VERSION, 'scala_cli': SCALA_CLI_VERSION, 'java': version,
            'cli_sha256': _binary_hash(str(Path(cli).resolve()), Path(cli).stat().st_mtime_ns, Path(cli).stat().st_size),
            'java_home': str(Path(java).resolve().parent.parent)}


@lru_cache(maxsize=4)
def _binary_hash(path, modified, size):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


@lru_cache(maxsize=4)
def _check_offline_cache(identity):
    # A readiness file alone cannot prove that caches still exist. Compile a
    # tiny trusted probe before the first model call in this process.
    with tempfile.TemporaryDirectory(prefix='skillforge-scala-preflight-') as directory:
        root = Path(directory)
        source = root / 'Main.scala'
        source.write_text('object Main { def main(args: Array[String]): Unit = () }')
        result = compile_sources([source], root / 'probe.jar')
        if result.status != Status.PASS:
            raise RuntimeError('Scala offline compiler preflight failed. Run python tools/setup_scala.py.\n'
                               + result.stderr[-3000:])


def preflight(execution_mode='stdio'):
    cli, java, version = toolchain()
    identity = readiness_identity()
    try:
        if json.loads(READY_FILE.read_text()) != identity:
            raise ValueError('toolchain changed')
    except (OSError, ValueError):
        raise RuntimeError('Scala compiler cache is not prepared. Run python tools/setup_scala.py.')
    _check_offline_cache(json.dumps(identity, sort_keys=True))
    image_id = None
    http_jvm = None
    if execution_mode == 'http' and os.environ.get('SKILLFORGE_SCALA_HTTP_LOCAL') != '1':
        try:
            image_id = subprocess.run(['docker', 'image', 'inspect', '--format', '{{.Id}}', HTTP_IMAGE], capture_output=True, text=True, check=True, timeout=15).stdout.strip()
            _check_container_runner(image_id, http_runner_hash())
            http_jvm = _container_runtime(image_id)
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError('HTTP execution requires Docker and its runner image. Start Docker and run '
                               'docker build -t skillforge-scala-http:1 -f tools/scala-http.Dockerfile . '
                               'from skillforge. Local execution is available only by explicitly setting '
                               'SKILLFORGE_SCALA_HTTP_LOCAL=1 (no container isolation).') from exc
    return {'scala': SCALA_VERSION, 'scala_cli': SCALA_CLI_VERSION, 'jvm': version,
            'cli_sha256': identity['cli_sha256'], 'java_home': identity['java_home'],
            'heap_mb': HEAP_MB, 'compile_timeout': COMPILE_TIMEOUT, 'offline': True, 'http_image_id': image_id,
            'http_jvm': http_jvm,
            'http_executor': ('local' if os.environ.get('SKILLFORGE_SCALA_HTTP_LOCAL') == '1' else HTTP_IMAGE)
                             if execution_mode == 'http' else None,
            'runner_hash': http_runner_hash()}


@lru_cache(maxsize=4)
def _container_runtime(image_id):
    probe = subprocess.run(['docker', 'run', '--rm', '--network', 'none', '--read-only',
                            '--user', '65534:65534', image_id, 'java', '-version'],
                           capture_output=True, text=True, check=True, timeout=15)
    version = (probe.stderr or probe.stdout).strip()
    if not re.search(r'version "' + re.escape(JDK_VERSION) + r'(?:[.+"-])', version):
        raise RuntimeError(f'HTTP runner must use JRE {JDK_VERSION}; rebuild its Docker image.')
    return version


def compile_sources(paths, output, *, classpath=None, main_class=None, offline=True):
    cli, _, _ = toolchain()
    command = [cli, '--power', 'package', *map(str, paths), '--scala', SCALA_VERSION,
               '--java-home', str(Path(toolchain()[1]).parent.parent), '--server=false', '--assembly', '--preamble=false',
               '--force', '-o', str(output)]
    if offline:
        command.append('--offline')
    if classpath:
        command += ['--classpath', str(classpath)]
    command += ['--main-class', main_class or 'Main']
    return execute(command, '', str(output.parent), COMPILE_TIMEOUT)


def _harness(assertions, entry_point):
    cases = []
    for i, assertion in enumerate(assertions):
        cases.append(f'case {i} => {{\n{assertion}\n}}')
    return '''object SkillForgeChecks {
  def main(args: Array[String]): Unit = {
    import Solution.'''+entry_point+'''
    val outcome = java.nio.file.Paths.get(args(1))
    try {
      args(0).toInt match {
'''+ '\n'.join(cases)+'''
        case _ => throw new IllegalArgumentException("Unknown test")
      }
      java.nio.file.Files.writeString(outcome, "pass")
    } catch {
      case e: AssertionError =>
        System.err.println(e.toString)
        java.nio.file.Files.writeString(outcome, "fail")
      case scala.util.control.NonFatal(e) =>
        System.err.println(e.toString)
        java.nio.file.Files.writeString(outcome, "error")
    }
  }
}
'''


class ScalaRunner:
    def run_many(self, code, inputs, *, timeout, execution_mode, entry_point, on_progress=None,
                 compile_only=False):
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('timeout must be a finite positive number')
        if not inputs:
            return []
        environment = preflight(execution_mode)
        _, java, _ = toolchain()
        function = execution_mode == ExecutionMode.FUNCTION
        if function and any(not isinstance(a, str) or not a.strip() for a in inputs):
            raise ValueError('function tests must be non-empty native assertion source')
        # Source cannot add dependencies, compiler plugins, files, or JVM options.
        if re.search(r'(?m)^\s*(?://>|//\s*using\b)', code):
            return [ExecutionResult(Status.COMPILE_ERROR, '', 'Build directives are not permitted; use Scala/JDK standard libraries.', 1) for _ in inputs]
        with tempfile.TemporaryDirectory(prefix='skillforge-scala-') as directory:
            root = Path(directory)
            candidate = root / 'Candidate.scala'
            candidate.write_text(code, encoding='utf-8')
            sources = [candidate]
            main = 'Main'
            if function:
                harness = root / 'SkillForgeChecks.scala'
                harness.write_text(_harness(inputs, entry_point), encoding='utf-8')
                sources.append(harness)
                main = 'SkillForgeChecks'
            jar = root / 'candidate.jar'
            if on_progress:
                on_progress('compilation', {})
            compilation = compile_sources(sources, jar, main_class=main)
            compilation = replace(compilation, stderr=re.sub(r'\x1b\[[0-9;]*m', '', compilation.stderr))
            if compilation.status == Status.TIMEOUT:
                results = [ExecutionResult(Status.TIMEOUT, '', 'Scala compilation timed out.\n' + compilation.stderr[:12000],
                                           compilation.exit_code) for _ in inputs]
            elif compilation.status != Status.PASS:
                # Classpath, compiler/cache failures are infrastructure failures, not model evidence.
                if not any(marker in compilation.stderr for marker in ('[E', '-- Error:', 'error found', 'errors found', 'main class', 'Main class')):
                    raise RuntimeError('Scala build could not complete: ' + compilation.stderr[-3000:])
                # Diagnostics in assertion source belong to the trusted suite.
                # Missing method/signature errors are candidate interface failures.
                if function and re.search(r'SkillForgeChecks\.scala:\d+', compilation.stderr):
                    harness_errors = [block for block in re.split(r'(?m)^-- ', compilation.stderr)
                                      if re.search(r'SkillForgeChecks\.scala:\d+', block)]
                    if any(not (re.search(r'\b(?:Solution|' + re.escape(entry_point) + r')\b', block)
                                and any(marker in block for marker in
                                    ('[E006]', '[E007]', '[E008]', '[E050]', '[E171]', '[E172]')))
                           for block in harness_errors):
                        raise RuntimeError('Scala assertion harness could not compile: ' + compilation.stderr[-3000:])
                results = [ExecutionResult(Status.COMPILE_ERROR, '', compilation.stderr[:12000], compilation.exit_code) for _ in inputs]
            elif compile_only:
                results = [replace(compilation, stdout='') for _ in inputs]
            elif execution_mode == ExecutionMode.HTTP:
                if on_progress:
                    on_progress('server_testing', {'test_count': len(inputs)})
                from src.verifier.runners.http import run_http
                results = run_http(jar, inputs, java, timeout, image_id=environment['http_image_id'])
            else:
                if on_progress:
                    on_progress('test_execution', {'test_count': len(inputs)})
                results = []
                for i, value in enumerate(inputs):
                    with tempfile.TemporaryDirectory(dir=root) as case_dir:
                        command = [java, f'-Xmx{HEAP_MB}m', '-XX:ActiveProcessorCount=2', '-cp', str(jar), main]
                        outcome = Path(case_dir) / 'outcome'
                        if function:
                            command += [str(i), str(outcome)]
                        result = execute(command, '' if function else _normalize_input(value), case_dir, timeout,
                                         env={'PATH': os.environ.get('PATH', ''), 'LANG': 'en_US.UTF-8'})
                        if function and result.status == Status.PASS:
                            try:
                                marker = outcome.read_text()
                                if marker not in ('pass', 'fail', 'error'):
                                    raise ValueError('invalid marker')
                                result.status = Status(marker)
                            except (OSError, ValueError):
                                result.status = Status.ERROR
                                result.stderr += '\nFunction test did not complete its assertion harness.'
                        results.append(result)
            # Compilation belongs to the batch, not each test's runtime.
            results[0] = replace(results[0], compilation_seconds=compilation.duration_seconds, execution_environment=environment)
            return results
