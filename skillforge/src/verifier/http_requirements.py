"""HTTP capability adapter. Routes, bodies and requirements come from task data."""
import json
import re

from src.evaluation.requirement_ledger import requirement_id, build_ledger

METHODS = r'GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS'


def extract_http_requirements(statement, atomic_requirements=()):
    requirements = []
    def add(quote, capability, selector=None, text=None):
        selector = selector or {}
        item = dict(id=requirement_id(quote,capability,selector), text=text or quote,
                    source_quote=quote, capability=capability, selector=selector)
        if item['id'] not in {r['id'] for r in requirements}:
            requirements.append(item)
    endpoints = []
    for match in re.finditer(rf'(?m)^\s*\|?\s*({METHODS})\s*\|?\s+(/[^\s|]+)\s*\|?\s+([^\n]+)', statement):
        method,path,behavior = match.groups()
        quote = match.group(0).strip()
        endpoints.append(dict(method=method,path=path,behavior=behavior.strip(' |')))
        add(quote,'endpoint',dict(method=method,path=path))
    quotes = [m.group(1).strip() for m in re.finditer(r'(?m)^\s*[-*]\s+(.+)$',statement)]
    for item in atomic_requirements:
        if isinstance(item,dict) and isinstance(item.get('source_quote'),str) and item['source_quote'] in statement:
            quotes.append(item['source_quote'])
    for quote in dict.fromkeys(quotes):
        low=quote.lower(); classified=False
        count=re.search(r'\b(?:at least|minimum(?: of)?)\s+(\d+)\s+(?:(?:sequential|consecutive|HTTP)\s+)?requests',quote,re.I)
        if count:
            add(quote,'sequential_requests',dict(minimum=int(count.group(1))));classified=True
        if re.search(r'thread.safe|concurren|synchroniz',low):
            add(quote,'concurrency');classified=True
        for word,cap in [('unicode','unicode'),('newline','newlines')]:
            if word in low:add(quote,cap);classified=True
        if re.search(r'keep.*(?:running|alive)|remain.*running',low):
            add(quote,'persistent');classified=True
        status = re.search(r'\b([1-5]\d\d)\b',quote)
        paths = re.findall(r'/[A-Za-z0-9_./-]+',quote)
        if status and paths and re.search(r'before.*first|initial',low):
            add(quote,'initial_status',dict(path=paths[0].rstrip('.'),status=int(status.group(1))));classified=True
        port=re.search(r'(?:localhost|127\.0\.0\.1):(\d+)',quote)
        if port:add(quote,'bind_port',dict(port=int(port.group(1))));classified=True
        if (re.search(r'\bonly\b.*\b(?:scala|jdk|java|standard)\b.*librar|no (?:external|third.party)',low)
                or ('httpserver' in low and re.search(r'\bjdk\b|built.in',low))):
            add(quote,'library_use',dict(httpserver='httpserver' in low));classified=True
        if not classified:add(quote,'unclassified')
    for match in re.finditer(r'\b(?:at least|minimum(?: of)?)\s+(\d+)\s+(?:(?:sequential|consecutive|HTTP)\s+)?requests',statement,re.I):
        selector=dict(minimum=int(match.group(1)))
        if not any(r['capability']=='sequential_requests' and r['selector']==selector for r in requirements):
            add(match.group(0),'sequential_requests',selector)
    # Delivery is tested independently of task behavior, never inferred from a
    # successful run with injected environment variables.
    add('Candidate must start without harness-only environment variables.', 'clean_environment')
    return requirements,endpoints


_ALLOWED_IMPORT_ROOTS = ('scala', 'java', 'javax', 'jdk')


def library_violations(code, require_httpserver):
    """Deterministic source check: Scala/JDK imports only, no dependency directives."""
    problems = []
    if re.search(r'(?m)^\s*//>\s*using\s+(?:dep|deps|lib|libs|repository)\b', code):
        problems.append('declares an external dependency')
    for name in re.findall(r'(?m)^\s*import\s+(?:_root_\.)?([A-Za-z_][\w.]*)', code):
        if name.split('.')[0] not in _ALLOWED_IMPORT_ROOTS and not name.startswith('com.sun.net.httpserver'):
            problems.append(f'imports {name}')
    if require_httpserver and 'com.sun.net.httpserver' not in code:
        problems.append('does not use com.sun.net.httpserver.HttpServer')
    return problems


def _matches(claim, requirement):
    normalize = lambda text: ' '.join(text.lower().split())
    claim = normalize(claim)
    return bool(claim) and (claim in normalize(requirement['source_quote']) or claim == normalize(requirement['text']))


def execution_ledger(requirements, suite, verification, code, independent=False):
    evidence=[]
    def record(req,test_id,result,detail,evidence_type='executed_test'):
        evidence.append(dict(requirement_id=req['id'],test_id=test_id,type=evidence_type,
                             result=result,detail=detail,independent=independent))
    for req in requirements:
        cap=req['capability'];selector=req['selector']
        for n,case in enumerate(verification.case_results,1):
            ex=case.execution; test_id=f'T{n}'
            observations=ex.http_observations
            relevant=[]
            if cap=='endpoint':
                relevant=[o for o in observations if o['method']==selector['method'] and o['path']==selector['path'] and o['body_checked']]
            elif cap=='initial_status':
                relevant=[o for o in observations if o['request_number']==1 and o['path']==selector['path'] and o['expected_status']==selector['status']]
            elif cap in ('unicode','newlines'):
                relevant=[o for o in observations if o[cap] and o['body_checked']]
            elif cap=='sequential_requests':
                if len(observations)>=selector['minimum'] and not any(o['parallel_group'] for o in observations):relevant=observations
            elif cap=='persistent':
                if len(observations)>=2:relevant=observations
            elif cap=='bind_port':
                if observations and ex.http_startup.get('port')==selector['port']:relevant=observations
            elif cap=='clean_environment':
                if ex.http_startup.get('clean_environment_error'):
                    record(req,test_id,'fail',dict(reason=ex.http_startup['clean_environment_error']))
                elif ex.http_startup.get('clean_environment') and observations:relevant=observations
                elif ex.http_startup.get('clean_environment') and not ex.http_startup.get('ready'):
                    record(req,test_id,'fail',dict(reason='Candidate did not start in a clean environment'))
            elif cap=='concurrency':
                # Bounded concurrent execution is observational evidence, not
                # proof of data-race freedom. Static indicators cannot promote it.
                concurrent=[o for o in observations if o['parallel_group'] and o['body_checked']]
                after=max((o['request_number'] for o in concurrent),default=0)
                state_checks=[o for o in observations if o['request_number']>after and o.get('one_of_checked')]
                if len(concurrent)>=2 and state_checks:relevant=concurrent+state_checks
            if relevant:
                record(req,test_id,'pass' if all(o['passed'] for o in relevant) else 'fail',
                       dict(request_numbers=[o['request_number'] for o in relevant],capability=cap))
        if cap=='concurrency':
            markers=re.findall(r'@volatile|\bsynchronized\b|\bAtomic\w+|\bvar\s+\w+',code)
            record(req,None,'observed',dict(indicators=markers[:30],
                warning='Patterns are hints, not proof of synchronization or thread safety.'),'static_check')
        if cap=='library_use':
            compiled=any(case.execution.status.value!='compile_error' for case in verification.case_results)
            problems=library_violations(code,selector.get('httpserver',False))
            if compiled:
                record(req,None,'fail' if problems else 'pass',
                       dict(check='compiled offline; imports limited to Scala/JDK packages',problems=problems),
                       'deterministic_check')
        # A person who wrote a supplied scenario may state which requirements it
        # evidences; the scenario's own executed result is the verdict.
        if independent and suite is not None:
            for n,(case,result) in enumerate(zip(suite.cases,verification.case_results),1):
                if any(_matches(claim,req) for claim in getattr(case,'requirements',()) or ()):
                    observations=result.execution.http_observations
                    if observations:
                        record(req,f'T{n}','pass' if all(o['passed'] for o in observations) else 'fail',
                               dict(declared_by='supplied scenario',request_numbers=[o['request_number'] for o in observations]))
                    elif result.execution.status.value!='pass':
                        record(req,f'T{n}','fail',dict(declared_by='supplied scenario',status=result.execution.status.value))
        if cap=='unclassified' and not any(item['requirement_id']==req['id'] for item in evidence):
            record(req,None,'unverified',dict(reason='No implemented deterministic capability for this requirement.'),'unverified')
    return build_ledger(requirements,evidence)
