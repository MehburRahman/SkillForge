"""Native analysis chooses delivery independently, without changing task semantics."""
import json

import pytest

from src.agents.llm_client import CallCountingClient
from src.agents.task_spec import (
    TaskFormatError, _parse_stdio_contracts, execution_problem_statement, parse_task,
)
from src.agents.task_spec_cache import TaskSpecCache


def selected_contracts():
    return {
        'stdin': dict(explicit_in_statement=False, quoted_rules='', serialization='JSON',
                      layout='One JSON array of integers', whitespace='JSON whitespace is insignificant',
                      empty_values='An empty array is []'),
        'stdout': dict(explicit_in_statement=False, quoted_rules='', serialization='plain text',
                       layout='One decimal integer', whitespace='End with a newline',
                       empty_values='The result is always an integer'),
    }


class NativeAnalysis:
    model = 'any-small-model'
    analysis_output_budget = 2048
    task_format_output_budget = 1024

    def __init__(self, contracts=None):
        self.contracts = list(contracts or [selected_contracts()])
        self.requests = []

    def generate_json(self, prompt, *, system, temperature, schema, max_output_tokens, seed=None):
        self.requests.append(dict(prompt=prompt, system=system, schema=schema,
                                  max_output_tokens=max_output_tokens, seed=seed))
        if 'stdin' in schema['required']:
            return json.dumps(self.contracts.pop(0))
        return json.dumps(dict(problem_type='array', constraints='', examples=[],
                               selected_max_attempts=2, attempt_budget_reason='Allow a repair.'))


def test_independent_delivery_is_complete_and_shared_with_execution():
    native = NativeAnalysis()
    client = CallCountingClient(native, seed=5)
    statement = 'Compute the sum of the supplied integers.'
    spec = parse_task(client, statement, language='scala')
    assert spec.problem_statement == statement
    assert spec.input_format_origin == spec.output_format_origin == 'selected'
    assert json.loads(spec.input_format)['layout'] == 'One JSON array of integers'
    assert json.loads(spec.output_format)['whitespace'] == 'End with a newline'
    execution = execution_problem_statement(spec)
    assert spec.input_format == json.loads(execution.split('\n')[3])['input_format']
    assert spec.required_operations == spec.examples == []
    assert client.call_count == len(native.requests) == 2
    assert 'input_format' not in native.requests[0]['schema']['required']
    assert native.requests[1]['max_output_tokens'] == 1024
    assert '"language": "scala"' in native.requests[1]['prompt']


def test_invented_stated_quote_is_rejected_and_fresh_model_choice_is_used():
    wrong = selected_contracts()
    wrong['stdin'].update(explicit_in_statement=True, quoted_rules='Input uses comma-separated integers.')
    client = NativeAnalysis([wrong, selected_contracts()])
    spec = parse_task(client, 'Compute the sum of the supplied integers.')
    assert spec.input_format_origin == 'selected'
    assert len(client.requests) == 3
    assert 'source excerpt must quote' in client.requests[-1]['prompt']
    assert 'comma-separated' not in client.requests[-1]['prompt']


def test_stated_rules_are_preserved_instead_of_model_paraphrases():
    contracts = selected_contracts()
    quote = 'The first line is a count; the next line contains that many comma-separated integers.'
    contracts['stdin'].update(explicit_in_statement=True, quoted_rules=quote)
    spec = parse_task(NativeAnalysis([contracts]), 'Compute the sum. ' + quote)
    assert spec.input_format == spec.input_format_source_excerpt == quote
    assert spec.input_format_origin == 'stated'


@pytest.mark.parametrize('mutation', [
    {'explicit_in_statement': 'false'},
    {'serialization': 'unknown'},
    {'layout': ''},
    {'whitespace': None},
    {'empty_values': 'x' * 501},
    {'quoted_rules': 'A quote claimed for a model selection'},
    {'explicit_in_statement': True, 'quoted_rules': ''},
])
def test_malformed_contracts_are_never_accepted(mutation):
    contracts = selected_contracts()
    contracts['stdin'].update(mutation)
    with pytest.raises(TaskFormatError):
        _parse_stdio_contracts(json.dumps(contracts), 'Compute a result.')


def test_incomplete_contract_retries_are_bounded():
    incomplete = selected_contracts()
    del incomplete['stdin']['layout']
    client = NativeAnalysis([incomplete] * 2)
    with pytest.raises(TaskFormatError, match='after 2 attempts'):
        parse_task(client, 'Compute a result.', max_generation_attempts=2)
    assert len(client.requests) == 3  # One task analysis, two contract attempts.


def test_selected_contracts_survive_revalidation_and_cache_hits(tmp_path):
    cache = TaskSpecCache(tmp_path)
    client = NativeAnalysis()
    first, hit, calls = cache.get_or_analyze(client, 'Compute the sum of integers.')
    assert not hit and calls == 2
    second, hit, calls = cache.get_or_analyze(client, 'Compute the sum of integers.')
    assert hit and calls == 0 and first == second


def test_function_execution_keeps_native_argument_analysis():
    client = NativeAnalysis()
    parse_task(client, 'Compute a result.', language='scala', execution_mode='function', entry_point='solve')
    assert len(client.requests) == 1
    assert 'input_format' in client.requests[0]['schema']['required']
    assert client.requests[0]['schema']['properties']['examples']['items']['properties']['input'] == {}
