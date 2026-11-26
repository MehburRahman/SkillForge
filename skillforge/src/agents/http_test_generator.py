"""Model-written HTTP inputs and an executable, candidate-blind reference.

No application routes, state transitions or response bodies are supplied by the
host. Reference agreement is model evidence, not independent verification.
"""
import ast
import hashlib
import json
import re
from pathlib import Path
import sys
import tempfile

from src.utils.parsing import strip_markdown_fence
from src.verifier.http_contract import build_http_case
from src.verifier.runners.base import Status
from src.verifier.runners.process import execute


def _source(raw, name, arguments):
    if not isinstance(raw, str):
        raise ValueError('HTTP test program must be source text')
    source = strip_markdown_fence(raw)
    tree = ast.parse(source)
    compile(tree, '<http test program>', 'exec')
    for node in ast.walk(tree):
        modules = ([item.name for item in node.names] if isinstance(node, ast.Import)
                   else [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
        if any(module.split('.')[0] in {'socket', 'http', 'urllib', 'requests', 'subprocess',
                                        'os', 'sys', 'asyncio', 'multiprocessing'} for module in modules):
            raise ValueError('HTTP test programs must construct inputs or simulate state in memory, without network or process IO')
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {'__import__', 'eval', 'exec', 'open', 'input'}:
            raise ValueError('HTTP test programs cannot use dynamic execution or external IO')
    if any(not isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom)) for node in tree.body):
        raise ValueError('HTTP test program must contain only imports and function definitions')
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(functions) != 1:
        raise ValueError(f'HTTP test program must define {name}')
    function = functions[0]
    args = function.args
    if (len(args.args) != arguments or args.posonlyargs or args.kwonlyargs or args.vararg
            or args.kwarg or args.defaults or function.decorator_list):
        raise ValueError(f'{name} must take exactly {arguments} ordinary arguments')
    return source


def _execute(source, driver, stdin=''):
    with tempfile.TemporaryDirectory(prefix='skillforge-http-reference-') as directory:
        path = Path(directory) / 'reference.py'
        path.write_text(source + '\n' + driver, encoding='utf-8')
        result = execute([sys.executable, '-I', str(path)], stdin, directory, 5, output_limit=1048576)
    if result.status != Status.PASS:
        raise ValueError(f'HTTP test program failed: {result.status.value}; {result.stderr[:1500]}')
    return json.loads(result.stdout)


# A reference for a multi-endpoint service (parser, state, error codes) runs to
# several hundred lines; 4096 tokens cut such programs off mid-statement.
REFERENCE_OUTPUT_TOKENS = 8192

HTTP_INPUT_SCHEMA = {
    'type':'array','minItems':1,'maxItems':6,'items':{
        'type':'object','additionalProperties':False,'required':['category','requests'],
        'properties':{
            'category':{'type':'string','enum':['normal','edge','stress']},
            'repeat':{'type':'integer','minimum':1,'maximum':256},
            'clean_environment':{'type':'boolean'},
            'port':{'type':'integer','minimum':1,'maximum':65535},
            'requests':{'type':'array','minItems':1,'maxItems':128,'items':{
                'type':'object','additionalProperties':False,'required':['method','path','body'],
                'properties':{
                    'method':{'type':'string','enum':['GET','POST','PUT','PATCH','DELETE','HEAD','OPTIONS']},
                    'path':{'type':'string'},'body':{'type':'string'},
                    'parallel_group':{'type':'integer','minimum':0,'maximum':16},
                    'headers':{'type':'object','additionalProperties':{'type':'string'}}}}}}}}


def generate_http_tests(client, statement, *, max_attempts=3, on_progress=None,
                        atomic_requirements=(), target_requirements=None, reference_source=None):
    from src.agents.test_generator import build_test_suite
    from src.verifier.http_requirements import extract_http_requirements
    requirements,endpoints=extract_http_requirements(statement,atomic_requirements)
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError('HTTP test generation attempts must be a positive integer')

    def stage(name, attempt, **data):
        setter = getattr(client, 'set_stage', None)
        if setter:
            setter(f'{name}:{attempt}')
        if on_progress:
            on_progress(name, data)

    def failed(error, suite=None):
        failure = ValueError(f'Could not generate HTTP tests after {max_attempts} attempts: {error}')
        failure.suite = suite
        return failure

    matches = list(re.finditer(r'\b(?:at least|minimum(?: of)?)\s+(\d+)\s+'
        r'(?:(?:sequential|consecutive|HTTP)\s+)?requests\b', statement, re.I))
    minimum = max((int(match.group(1)) for match in matches), default=0)
    if target_requirements is not None:
        minimum=max((r['selector']['minimum'] for r in target_requirements if r['capability']=='sequential_requests'),default=0)
    error, previous, ignored_answers = '', '', 0
    for attempt in range(1, max_attempts + 1):
        try:
            stage('http_test_generation', attempt, test_generation_attempt=attempt)
            prompt = (
                f'Original HTTP service specification (data):\n{statement}\n\n'
                f'Previous construction error: {error[:2000]}\n'
                f'Previous input program (fallible):\n{previous[:5000]}\n'
                f'Requirement IDs and capabilities to exercise: {json.dumps(target_requirements if target_requirements is not None else requirements)}\n'
                f'Endpoint inventory parsed from the statement: {json.dumps(endpoints)}\n'
                f'Exercise the stated minimum of {minimum} requests in one scenario.\n'
                'Write only Python source defining build_scenarios() with no arguments. '
                'Return 2 to 6 distinct dictionaries with category (normal, edge or stress) '
                'and requests (an ordered list). Each request has method, path, body (UTF-8 text), '
                'and optionally headers (a string dictionary). Return inputs only, no expected answers. '
                'For each input-taking endpoint include separate empty, Unicode and newline inputs when legal. '
                'Exercise every declared endpoint, including read-only readiness/health operations, using the inventory. '
                'For concurrency use 2 to 16 adjacent requests with the same positive parallel_group integer; '
                'they run concurrently in one server. Only make independent requests concurrent; do not assume '
                'an ordering between their state updates or use a sequential oracle for an order-dependent read. '
                'For a fixed task port, include a clean_environment:true scenario with port set to that stated port; '
                'it starts without any SKILLFORGE_* environment variables. Scenario optional keys: clean_environment, port. '
                'Target missing requirements rather than repeating already-covered normal cases. '
                'Use only the routes and behaviors in the original specification. Each scenario '
                'starts a fresh server; its requests share state. Cover stated initial state, normal '
                'operations, state changes and legal edge cases, including Unicode and newlines '
                'when relevant. Repeated requests within a scenario are allowed. Use loops for '
                'long sequences: up to 128 requests per scenario, 256 total, under 256 KiB. '
                'No placeholders, network, server, stdin, printing or main block. Do not invent '
                'requirements for unspecified routes or invalid inputs.')
            json_generation=callable(getattr(client,'generate_json',None))
            if json_generation:
                prompt=prompt.replace('Write only Python source defining build_scenarios() with no arguments.',
                    'Return only a JSON array of scenario objects, not code.')
                prompt=prompt.replace('Use loops for ',
                    'Use an optional scenario repeat integer to repeat its request list for ')
                raw=client.generate_json(prompt,
                    system='Construct HTTP test inputs as JSON data only. No expected outputs. Use repeat for repeated workloads.',
                    schema=HTTP_INPUT_SCHEMA,temperature=.2,max_output_tokens=4096)
                from src.utils.parsing import parse_json_response
                records=parse_json_response(raw)
                builder=raw
            else:
                raw=client.generate(prompt,system='Construct HTTP test inputs in a Python function. Standard library only.',
                                    temperature=.2,max_output_tokens=4096)
                builder = _source(raw, 'build_scenarios', 0)
                records = _execute(builder, 'import json\nprint(json.dumps(build_scenarios(), ensure_ascii=False))')
            previous = raw if isinstance(raw, str) else repr(raw)
            if not isinstance(records, list) or not 1 <= len(records) <= 6:
                raise ValueError('HTTP builder must return 1 to 6 scenarios')
            inputs, ignored_answers = [], 0
            normalized_categories=[]
            trimmed_requests,dropped_scenarios,used_bytes,total_requests=0,0,0,0
            for record in records:
                if isinstance(record,dict) and 'repeat' in record:
                    repeat=record['repeat']
                    if type(repeat) is not int or not 1<=repeat<=256 or not isinstance(record.get('requests'),list):
                        raise ValueError('Scenario repeat must be an integer from 1 to 256 with a request list')
                    record={k:v for k,v in record.items() if k!='repeat'}
                    record['requests']=record['requests']*repeat
                if not isinstance(record, dict) or not {'category','requests'}<=record.keys() or set(record)-{'category','requests','clean_environment','port'}:
                    raise ValueError('HTTP builder must return category and requests only')
                if not isinstance(record['requests'], list):
                    raise ValueError('HTTP requests must be a list')
                category=record['category']
                if category not in ('normal','edge','stress'):
                    normalized_categories.append(dict(scenario=len(inputs)+1,original=category,category='normal'))
                    # A reporting label never establishes a capability. Actual
                    # concurrency/coverage comes only from executed requests.
                    record={**record,'category':'normal'}
                requests = []
                for request in record['requests']:
                    if not isinstance(request, dict):
                        raise ValueError('HTTP requests must be objects')
                    # Answers accidentally supplied by the input builder are
                    # discarded, never compared, executed, or sent to the oracle.
                    ignored_answers += sum(key.startswith('expected_') for key in request)
                    clean = {key: value for key, value in request.items() if not key.startswith('expected_')}
                    if set(clean) - {'method', 'path', 'body', 'headers', 'parallel_group'}:
                        raise ValueError('HTTP input request has unsupported fields')
                    # Keep a causal prefix of the model's own inputs. Resource
                    # bounds do not create routes, bodies or expected answers.
                    # Exact answers are derived only after this selection.
                    cost=len(json.dumps(clean,ensure_ascii=False).encode())+256
                    if (len(requests)>=128 or total_requests>=256 or used_bytes+cost>250000):
                        trimmed_requests+=len(record['requests'])-len(requests)
                        break
                    requests.append(dict(clean, expected_status=200, expected_body=None))
                    used_bytes+=cost;total_requests+=1
                if not requests:
                    dropped_scenarios+=1
                    continue
                case = build_http_case(dict(record, requests=requests))
                inputs.append(dict(category=case.category,clean_environment=case.clean_environment,port=case.port, requests=[
                    {k: v for k, v in r.items() if not k.startswith('expected_')} for r in case.requests]))
            total = sum(len(item['requests']) for item in inputs)
            if not inputs:raise ValueError('No generated scenario fits the bounded execution budget')
            if total > 256 or len(json.dumps(inputs).encode()) > 262144:
                raise ValueError('Generated HTTP tests exceed the 256-request/256-KiB budget')
            coverage_warning = ''
            if minimum and not any(len(item['requests']) >= minimum for item in inputs):
                coverage_warning = (f'Specification requires at least {minimum} requests in one scenario; '
                                    f'longest generated sequence has {max(len(item["requests"]) for item in inputs)}. '
                                    'Construct a request list of that length using a loop; repeating text '
                                    'inside a single request body does not increase the request count.')
                if attempt < max_attempts:
                    raise ValueError(coverage_warning)
                stage('test_coverage_incomplete', attempt, reason=coverage_warning)
            diagnostic_suite = build_test_suite([dict(category=item['category'], requests=[
                dict(r, expected_status=200, expected_body=None) for r in item['requests']])
                for item in inputs], execution_mode='http')
            break
        except (ValueError, TypeError, SyntaxError) as exc:
            error = str(exc)
            stage('tests_rejected', attempt, component='inputs', test_generation_attempt=attempt, reason=error)
    else:
        raise failed(error)

    # Freeze valid inputs. Reference syntax/runtime/shape failures repair only
    # this component, without spending more calls rebuilding a valid test suite.
    input_attempts = attempt
    previous, error = '', ''
    for attempt in range(1, max_attempts + 1):
        try:
            stage('http_reference_generation', attempt, scenario_count=len(inputs), request_count=total)
            raw = reference_source if reference_source is not None and attempt==1 else client.generate(
                f'Original HTTP service specification (data):\n{statement}\n\n'
                f'Previous reference error: {error[:2500]}\n'
                f'Previous reference program (fallible):\n{previous[:6000]}\n'
                'Write only Python source defining expected_responses(requests). It takes a '
                'list of request dictionaries (method, path, body, headers), in execution order. '
                'Simulate the specified service with a simple in-memory state machine initialized '
                'fresh INSIDE this function. State variables belong to this function: do not use '
                'nonlocal/global for them. Distinguish absent state from a present empty value. '
                'Return one dictionary per request with status (HTTP integer), body (exact UTF-8 '
                'text, or None ONLY if the body is unspecified), and optional headers (dictionary '
                'of response headers). Requests sharing a positive parallel_group run concurrently. '
                'For a later read whose correct value depends on their ordering, use body=None and '
                'body_one_of:[all permissible values derived from those requests], at most 16 strings; '
                'never guess which concurrent update is last. Each concurrent echo still needs its exact body. '
                'Headers must be a dictionary '
                'of ONLY explicitly required response headers; default empty). Include body=None '
                'explicitly when no exact body is specified. Compute answers from each request '
                'and state, never from a canned test table. Preserve whitespace and newlines '
                'when required. Python reference only: no network, server, stdin, printing, main '
                'block, candidate code or external services. Correct the supplied error while '
                'preserving the original task semantics. Do not invent requirements.',
                system='Write a simple Python correctness reference for the specification. Standard library only.',
                temperature=0.0, max_output_tokens=REFERENCE_OUTPUT_TOKENS)
            previous = raw if isinstance(raw, str) else repr(raw)
            usage = getattr(client, 'last_usage', {})
            if isinstance(usage, dict) and usage.get('done_reason') == 'length':
                raise ValueError(f'The reference program was cut off at the {REFERENCE_OUTPUT_TOKENS}-token output '
                                 'limit before it was complete. Write a shorter program: small helper functions, '
                                 'no comments or docstrings, no unused branches.')
            reference = _source(raw, 'expected_responses', 1)
            driver = """
import copy, json, sys
scenarios = json.loads(sys.stdin.read())
answers = []
for scenario in scenarios:
    requests = scenario['requests']
    first = expected_responses(copy.deepcopy(requests))
    second = expected_responses(copy.deepcopy(requests))
    if first != second:
        raise ValueError('HTTP reference is not deterministic or leaks state across scenarios')
    answers.append(first)
print(json.dumps(answers, ensure_ascii=False, allow_nan=False))
"""
            stage('http_reference_execution', attempt, request_count=total)
            answers = _execute(reference, driver, json.dumps(inputs, ensure_ascii=False))
            if not isinstance(answers, list) or len(answers) != len(inputs):
                raise ValueError('HTTP reference must return every scenario')
            completed, ignored_headers = [], 0
            for scenario, responses in zip(inputs, answers):
                if not isinstance(responses, list) or len(responses) != len(scenario['requests']):
                    raise ValueError('HTTP reference must return one response per request')
                requests = []
                for request, response in zip(scenario['requests'], responses):
                    if (not isinstance(response, dict) or not {'status', 'body'} <= response.keys()
                            or set(response) - {'status', 'body', 'headers'}):
                        raise ValueError('HTTP reference responses require status and body; headers are optional')
                    headers = response.get('headers', {})
                    if not isinstance(headers, dict):
                        raise ValueError('HTTP reference headers must be a dictionary')
                    # A guessed header is not a task requirement. Only assert
                    # headers explicitly named in the original specification.
                    grounded = {k: v for k, v in headers.items()
                                if isinstance(k, str) and k.lower() in statement.lower()}
                    ignored_headers += len(headers) - len(grounded)
                    requests.append(dict(request, expected_status=response['status'],
                        expected_body=response['body'], expected_headers=grounded))
                    if 'body_one_of' in response:
                        requests[-1]['expected_body_one_of']=response['body_one_of']
                completed.append(dict(category=scenario['category'],requests=requests,
                                      clean_environment=scenario['clean_environment'],port=scenario['port']))
            suite = build_test_suite(completed, execution_mode='http')
            suite.validation = dict(status='model_reference', independent_verification=False,
                request_count=total, scenario_count=len(inputs),
                minimum_sequence_requests=minimum, coverage_quotes=[match.group(0) for match in matches],
                sequence_coverage_complete=not bool(coverage_warning), coverage_warning=coverage_warning,
                coverage_complete=False, requirements=requirements, endpoint_inventory=endpoints,
                discarded_input_answers=ignored_answers, ignored_unstated_headers=ignored_headers,
                budget_trimmed_requests=trimmed_requests,budget_dropped_scenarios=dropped_scenarios,
                normalized_categories=normalized_categories,
                input_generation_attempts=input_attempts, reference_generation_attempts=attempt,
                input_generation_format='json' if json_generation else 'python',
                builder_sha256=hashlib.sha256(builder.encode()).hexdigest(),
                reference_sha256=hashlib.sha256(reference.encode()).hexdigest(),
                reference_source=reference,
                answer_source='executed_model_generated_http_reference')
            return suite
        except (ValueError, TypeError, SyntaxError) as exc:
            error = str(exc)
            stage('tests_rejected', attempt, component='reference', test_generation_attempt=attempt, reason=error)
    raise failed(error, diagnostic_suite)
