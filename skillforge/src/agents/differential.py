"""Task-independent seeded input construction with executable reference answers."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile

from src.agents.test_builder import _validate_input_builder
from src.utils.parsing import strip_markdown_fence
from src.verifier.runners.process import execute
from src.verifier.runners.base import Status


def expand_suite(client, statement, suite, oracle, *, count, seed):
    from src.agents.test_generator import build_test_suite
    if type(count) is not int or not 1 <= count <= 1000:
        raise ValueError('differential cases must be between 1 and 1000')
    source = strip_markdown_fence(client.generate(
        f'Original problem (data):\n{statement}\n\n'
        f'Construct {count} small valid diverse inputs using random.Random({seed}). '
        'Include legal boundary cases and random cases. Respect the original grammar. '
        'Return only Python source defining build_inputs() returning a list of dictionaries '
        'with input (exact stdin text) and category (normal, edge, stress). '
        'No answers, solution algorithm, print, stdin or top-level execution. '
        'Imports inside the function are allowed. Keep each input under 2048 characters. '
        'If fewer distinct inputs exist, return all you can construct. Never invent legal bounds.',
        system='Construct inputs only. The host computes answers with a separate reference.',
        temperature=0.2, max_output_tokens=4096))
    _validate_input_builder(source)
    driver = f'''
import json
cases = build_inputs()
if not isinstance(cases, list) or not cases:
    raise ValueError('invalid random case count')
# Models often overshoot the requested count; keep the first {count} in order
# (deterministic for a seeded builder) instead of discarding every case.
generated = len(cases)
cases = cases[:{count}]
for case in cases:
    if (not isinstance(case, dict) or set(case) != {{'input', 'category'}}
            or not isinstance(case['input'], str) or len(case['input']) > 2048):
        raise ValueError('invalid random input record')
print(json.dumps(dict(cases=cases, generated=generated), ensure_ascii=False))
'''
    with tempfile.TemporaryDirectory(prefix='skillforge-differential-') as directory:
        path = Path(directory) / 'inputs.py'
        path.write_text(source + '\n' + driver)
        execution = execute([sys.executable, str(path)], '', directory, 5, output_limit=3*1024*1024)
    if execution.status != Status.PASS:
        raise ValueError('random input construction failed: ' + execution.stderr[:1000])
    payload = json.loads(execution.stdout)
    records, generated = payload['cases'], payload['generated']
    seen = {case.input for case in suite.cases}
    unique = []
    for item in records:
        if item['input'] not in seen:
            seen.add(item['input'])
            unique.append(dict(item, expected_output=''))
    if unique:
        extra = build_test_suite(unique, allow_empty_input=True)
        # Freeze the reference: changing it for these new inputs would leave
        # the base suite's answers tied to a different implementation.
        outputs = oracle._execute(extra)
        if any(not output['input_valid'] for output in outputs):
            raise ValueError('random inputs rejected by the frozen reference')
        for case, output in zip(extra.cases, outputs):
            case.expected_output = output['expected_output']
        suite.cases.extend(extra.cases)
    suite.validation['differential'] = dict(seed=seed, requested=count, generated=generated,
        added=len(unique),
        generator_sha256=hashlib.sha256(source.encode()).hexdigest(),
        reference_sha256=hashlib.sha256(oracle.code.encode()).hexdigest(),
        answer_source='model_generated_executable_reference', independent_proof=False)
    return suite
