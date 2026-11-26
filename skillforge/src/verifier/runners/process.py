"""Bounded process capture for compiled candidates and their compiler."""
import os
import selectors
import signal
import subprocess
import tempfile
import threading
import time

from src.verifier.runners.base import ExecutionResult, Status

OUTPUT_LIMIT = 262144
_STOP_LOCK = threading.Lock()


def stop_process(proc):
    # A parent may have exited while descendants are still alive.
    # Reap an exited parent first: macOS can return EPERM when signalling a
    # group whose only member is an unreaped zombie after a deadline kill.
    with _STOP_LOCK:
        if getattr(proc, '_skillforge_stop_sent', False):
            return
        proc.poll()
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            if proc.poll() is None:
                raise
        proc._skillforge_stop_sent = True


def execute(command, stdin, cwd, timeout, *, env=None, output_limit=OUTPUT_LIMIT):
    started = time.monotonic()
    outputs = [bytearray(), bytearray()]
    status, error = None, ''
    with tempfile.TemporaryFile() as input_file:
        input_file.write(stdin.encode('utf-8'))
        input_file.seek(0)
        try:
            proc = subprocess.Popen(command, stdin=input_file, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, cwd=cwd, env=env, start_new_session=True)
        except OSError as exc:
            raise RuntimeError(f'Could not start execution tool {command[0]}: {exc}') from exc
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(proc.stdout, selectors.EVENT_READ, 0)
                selector.register(proc.stderr, selectors.EVENT_READ, 1)
                while selector.get_map():
                    if time.monotonic() - started >= timeout:
                        status, error = Status.TIMEOUT, '\nExecution timed out'
                        break
                    for key, _ in selector.select(0.05):
                        chunk = os.read(key.fileobj.fileno(), 8192)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        buffer = outputs[key.data]
                        remaining = output_limit - len(buffer)
                        buffer.extend(chunk[:remaining])
                        if len(chunk) > remaining:
                            status, error = Status.ERROR, '\nOutput limit exceeded'
                            break
                    if status:
                        break
                if status is None:
                    try:
                        proc.wait(timeout=max(0.001, timeout - (time.monotonic() - started)))
                    except subprocess.TimeoutExpired:
                        status, error = Status.TIMEOUT, '\nExecution timed out'
        finally:
            stop_process(proc)
            proc.wait()
            proc.stdout.close()
            proc.stderr.close()
    return ExecutionResult(status or (Status.PASS if proc.returncode == 0 else Status.ERROR),
                           outputs[0].decode('utf-8', errors='replace'),
                           outputs[1].decode('utf-8', errors='replace') + error,
                           proc.returncode, time.monotonic() - started)
