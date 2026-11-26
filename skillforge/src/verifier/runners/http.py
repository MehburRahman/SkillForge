"""Generic HTTP scenario runner. No application routes or expected answers live here."""
from dataclasses import asdict
import http.client
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from uuid import uuid4

from src.verifier.runners.base import ExecutionResult, Status
from src.verifier.runners.process import OUTPUT_LIMIT, execute, stop_process

STARTUP_TIMEOUT = 15
MAX_RESPONSE = 1048576


class HTTPInfrastructureError(RuntimeError):
    """The harness failed; no candidate correctness verdict is available."""


def _run_scenarios(jar, scenarios, java, timeout):
    try:
        return [_scenario(jar, requests, java, timeout) for requests in scenarios]
    except Exception as exc:
        raise HTTPInfrastructureError(
            f'Infrastructure error: HTTP verifier crashed ({type(exc).__name__}): {str(exc)[:1000]}') from exc


def _excerpt(value, limit=2000):
    return value if value is None or len(value) <= limit else value[:limit] + '...[truncated]'


def _header_excerpt(items):
    return {key[:128]:_excerpt(value,256) for key,value in list(items)[:30]}


def _http_counterexample(requests, index, actual):
    request = requests[index - 1]
    def inputs(item):
        return dict(method=item['method'], path=_excerpt(item['path'], 512),
                    body=_excerpt(item.get('body', '')),
                    headers=_header_excerpt(item.get('headers', {}).items()))
    expected=dict(status=request['expected_status'], body=_excerpt(request['expected_body']),
                  headers=_header_excerpt(request.get('expected_headers', {}).items()))
    if request['expected_body'] is not None:
        expected['utf8_bytes']=len(request['expected_body'].encode('utf-8'))
    if 'expected_body_one_of' in request:
        expected['body_one_of']=[_excerpt(v) for v in request['expected_body_one_of']]
    return dict(request_number=index, request=inputs(request),
                prior_requests=[inputs(item) for item in requests[max(0,index-3):index-1]],
                expected=expected,actual=actual)


def _send_request(proc, port, requests, index, timeout, barrier=None):
    request=requests[index-1]
    connection=http.client.HTTPConnection('127.0.0.1',port,timeout=timeout)
    expired=threading.Event()
    def expire():
        expired.set();stop_process(proc)
    timer=threading.Timer(timeout,expire)
    status,phase,body=Status.PASS,'connecting',None
    actual_status=None; actual_headers={};details=[];counterexample={};raw=None;declared_length=None
    try:
        if barrier is not None:barrier.wait(timeout=timeout)
        timer.start()
        connection.request(request['method'],request['path'],body=request.get('body','').encode('utf-8'),
            headers={'content-type':'text/plain; charset=utf-8',**request.get('headers',{})})
        phase='response headers'
        response=connection.getresponse();actual_status=response.status
        actual_headers=_header_excerpt(response.getheaders());declared_length=response.length
        phase='response body'
        raw=response.read(MAX_RESPONSE+1)
        if expired.is_set():raise TimeoutError('Request deadline exceeded')
        if len(raw)>MAX_RESPONSE:raise RuntimeError('HTTP response exceeds 1 MiB limit')
        if response.length is not None and response.length>0:
            body=raw.decode('utf-8',errors='replace')
            raise http.client.IncompleteRead(raw,response.length)
        body=raw.decode('utf-8',errors='strict');phase='complete'
        expected=request.get('expected_body')
        alternatives=request.get('expected_body_one_of')
        matches=(actual_status==request['expected_status'] and (expected is None or body==expected)
                 and (alternatives is None or body in alternatives)
                 and all(response.getheader(k)==v for k,v in request.get('expected_headers',{}).items()))
        if not matches:
            status=Status.FAIL
            preview=repr(alternatives)[:1000] if alternatives is not None else '<not checked>' if expected is None else repr(expected[:1000])
            details.append(f"Request {index} {request['method']} {request['path']}: expected status {request['expected_status']}, "
                           f"body {preview}, headers {request.get('expected_headers',{})}; got status {actual_status}, "
                           f"body {body[:1000]!r}, headers {actual_headers}")
    except (TimeoutError,OSError,ValueError,http.client.HTTPException,RuntimeError,threading.BrokenBarrierError) as exc:
        if isinstance(exc,http.client.IncompleteRead):
            raw=exc.partial;body=raw.decode('utf-8',errors='replace')
        status=Status.TIMEOUT if isinstance(exc,TimeoutError) or expired.is_set() else Status.ERROR
        details.append(str(exc))
        if phase=='response body':
            details.append(f'Received response status {actual_status}, but its body did not complete. '
                           'Check response byte length, chunked-response completion, and closing the exchange.')
    finally:
        if timer.ident is not None:
            timer.cancel();timer.join()
        connection.close()
    if status!=Status.PASS:
        counterexample=_http_counterexample(requests,index,
            dict(status=actual_status,body=_excerpt(body),headers=actual_headers,phase=phase,
                 received_body_bytes=len(raw) if raw is not None else None,declared_body_bytes=declared_length,
                 error='; '.join(details)[:1000]))
        counterexample['execution_status']=status.value
    supplied=request.get('body','')
    observation=dict(request_number=index,method=request['method'],path=request['path'],status=actual_status,
        expected_status=request['expected_status'],passed=status==Status.PASS,
        body_checked=request.get('expected_body') is not None or request.get('expected_body_one_of') is not None,
        one_of_checked=request.get('expected_body_one_of') is not None,
        empty_body=not supplied,unicode=any(ord(c)>127 for c in supplied),newlines='\n' in supplied,
        parallel_group=request.get('parallel_group',0))
    return status,details,counterexample,observation


def _scenario(jar, scenario, java, timeout):
    from concurrent.futures import ThreadPoolExecutor
    requests=scenario['requests'] if isinstance(scenario,dict) else scenario
    clean=isinstance(scenario,dict) and scenario.get('clean_environment',False)
    specified_port=scenario.get('port') if isinstance(scenario,dict) else None
    started=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='skillforge-http-') as directory:
        port_file=Path(directory)/'port'
        env={'PATH':os.environ.get('PATH',''),'LANG':'en_US.UTF-8'}
        if not clean:env['SKILLFORGE_PORT_FILE']=str(port_file)
        proc=subprocess.Popen([java,'-Xmx256m','-XX:ActiveProcessorCount=2','-jar',str(jar)],cwd=directory,
            env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
        logs,overflow=[bytearray(),bytearray()],threading.Event()
        def drain(pipe,buffer):
            while True:
                chunk=pipe.read(4096)
                if not chunk:return
                remaining=OUTPUT_LIMIT-len(buffer);buffer.extend(chunk[:remaining])
                if len(chunk)>remaining:overflow.set();stop_process(proc);return
        readers=[threading.Thread(target=drain,args=(pipe,buf),daemon=True)
                 for pipe,buf in zip((proc.stdout,proc.stderr),logs)]
        for reader in readers:reader.start()
        expired=threading.Event()
        def expire():expired.set();stop_process(proc)
        deadline=threading.Timer(STARTUP_TIMEOUT+timeout*len(requests),expire);deadline.start()
        status,details,counterexample,observations=Status.PASS,[],{},[]
        counterexamples=[]
        startup=dict(clean_environment=clean,ready=False,parallel_requests=0)
        try:
            port=None
            while time.monotonic()-started<STARTUP_TIMEOUT:
                if proc.poll() is not None:raise RuntimeError('Server exited before becoming ready')
                try:
                    proposed=specified_port if clean else int(port_file.read_text().strip())
                    if type(proposed) is not int or not 1<=proposed<=65535:raise ValueError('Invalid readiness port')
                    # A TCP connection only; no invented health endpoint or HTTP
                    # request is sent before the declared initial-state checks.
                    import socket
                    with socket.create_connection(('127.0.0.1',proposed),timeout=.1):pass
                    port=proposed;break
                except (OSError,ValueError):time.sleep(.02)
            if port is None:raise TimeoutError('Server startup timed out; readiness was not observed')
            startup.update(port=port,ready=True)
            cursor=0
            while cursor<len(requests):
                group=requests[cursor].get('parallel_group',0);end=cursor+1
                if group:
                    while end<len(requests) and requests[end].get('parallel_group')==group:end+=1
                size=end-cursor
                if group and size>1:
                    startup['parallel_requests']+=size
                    barrier=threading.Barrier(size)
                    with ThreadPoolExecutor(max_workers=size) as pool:
                        futures=[pool.submit(_send_request,proc,port,requests,i+1,timeout,barrier) for i in range(cursor,end)]
                        batch=[future.result() for future in futures]
                else:batch=[_send_request(proc,port,requests,cursor+1,timeout)]
                for outcome,messages,failure,observation in batch:
                    observations.append(observation);details.extend(messages)
                    if outcome!=Status.PASS:
                        if status==Status.PASS or outcome in (Status.ERROR,Status.TIMEOUT):status=outcome
                        if not counterexample:counterexample=failure
                        if failure and len(counterexamples)<3:counterexamples.append(failure)
                cursor=end
                if status==Status.TIMEOUT or proc.poll() is not None:break
            if proc.poll() is not None and status==Status.PASS:raise RuntimeError('Server exited during its request scenario')
        except (TimeoutError,OSError,ValueError,RuntimeError) as exc:
            status=Status.TIMEOUT if isinstance(exc,TimeoutError) else Status.ERROR;details.append(str(exc))
        finally:
            deadline.cancel();stop_process(proc);proc.wait()
            for reader in readers:reader.join(timeout=1)
            proc.stdout.close();proc.stderr.close()
        if expired.is_set():status=Status.TIMEOUT;details.append('HTTP scenario deadline exceeded')
        if overflow.is_set():status=Status.ERROR;details.append('Server output limit exceeded')
        if clean and b'Exception in thread' in logs[1]:
            status=Status.ERROR
            startup['clean_environment_error']='Candidate emitted an uncaught JVM exception without harness environment variables'
            details.append(startup['clean_environment_error'])
        return ExecutionResult(status,logs[0].decode('utf-8',errors='replace')[:4000],
            ('\n'.join(details)+'\n'+logs[1].decode('utf-8',errors='replace'))[:12000],
            0 if status in (Status.PASS,Status.FAIL) else -1,time.monotonic()-started,
            http_failure=counterexample,http_failures=counterexamples,http_observations=observations,http_startup=startup)


def run_http(jar, scenarios, java, timeout, *, image_id=None):
    if os.environ.get('SKILLFORGE_SCALA_HTTP_LOCAL') == '1':
        return _run_scenarios(jar, scenarios, java, timeout)
    from src.verifier.runners.scala import HTTP_IMAGE
    # Requests run inside the same network namespace as the candidate. No ports
    # are published and the container has no external network access.
    with tempfile.TemporaryDirectory(prefix='skillforge-http-tests-') as directory:
        path = Path(directory) / 'scenarios.json'
        path.write_text(json.dumps(scenarios), encoding='utf-8')
        name = 'skillforge-' + uuid4().hex
        command = ['docker', 'run', '--rm', '--name', name, '--network', 'none', '--read-only',
                   '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--pids-limit', '128',
                   '--memory', '512m', '--cpus', '2', '--user', '65534:65534',
                   '--tmpfs', '/tmp:rw,nosuid,size=128m',
                   '-v', f'{jar}:/work/candidate.jar:ro', '-v', f'{path}:/work/scenarios.json:ro',
                   image_id or HTTP_IMAGE, 'python3', '-m', 'src.verifier.runners.http', str(timeout)]
        try:
            limit = sum(STARTUP_TIMEOUT + timeout * len(s["requests"] if isinstance(s,dict) else s) + 3 for s in scenarios) + 30
            result = execute(command, '', directory, limit, output_limit=16 * 1048576)
            if result.status != Status.PASS:
                raise HTTPInfrastructureError('Infrastructure error: HTTP container failed: ' + result.stderr[:2000])
            rows = json.loads(result.stdout)
            if isinstance(rows, dict) and 'infrastructure_error' in rows:
                raise HTTPInfrastructureError(str(rows['infrastructure_error'])[:2000])
            if len(rows) != len(scenarios):
                raise ValueError('incomplete results')
            return [ExecutionResult(**{**row, 'status': Status(row['status'])}) for row in rows]
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPInfrastructureError('Infrastructure error: HTTP runner returned invalid results') from exc
        finally:
            try:
                cleanup = subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=15)
                if cleanup is not None and cleanup.returncode and 'No such container' not in cleanup.stderr.decode(errors='replace'):
                    raise RuntimeError('HTTP container cleanup failed; check Docker for ' + name)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise RuntimeError('HTTP container cleanup failed; check Docker for ' + name) from exc


if __name__ == '__main__':
    try:
        scenarios = json.loads(Path('/work/scenarios.json').read_text())
        results = _run_scenarios(Path('/work/candidate.jar'), scenarios, 'java', float(sys.argv[1]))
        payload = [{**asdict(r), 'status': r.status.value} for r in results]
    except Exception as exc:
        payload = {'infrastructure_error': f'Infrastructure error: HTTP verifier: {str(exc)[:1500]}'}
    print(json.dumps(payload))
