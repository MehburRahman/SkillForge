import json

import pytest

from src.agents.task_spec import (
    TASK_SPEC_RESPONSE_SCHEMA, MAX_ATTEMPT_BUDGET_REASON_CHARACTERS,
    parse_task, _parse_task_response, task_spec_response_schema, TaskFormatError,
)


class TaskAnalysisClient:
    def __init__(self, response, include_budget=True):
        self.response = response
        self.include_budget = include_budget
        self.requests = []

    def generate(self, prompt, system=None, temperature=0.2):
        self.requests.append({
            "prompt": prompt,
            "system": system,
            "temperature": temperature,
        })
        if isinstance(self.response, str):
            return self.response
        response = self.response
        if self.include_budget and isinstance(response, dict):
            response = {
                "selected_max_attempts": 2,
                "attempt_budget_reason": "Test-selected attempt budget.",
                **response,
            }
        return json.dumps(response)


class SequentialTaskAnalysisClient:
    def __init__(self, responses, include_budget=True):
        self.responses = list(responses)
        self.include_budget = include_budget
        self.requests = []

    def generate(self, prompt, system=None, temperature=0.2):
        self.requests.append({
            "prompt": prompt,
            "system": system,
            "temperature": temperature,
        })
        response = self.responses.pop(0)
        if self.include_budget and isinstance(response, dict):
            response = {
                "selected_max_attempts": 2,
                "attempt_budget_reason": "Test-selected attempt budget.",
                **response,
            }
        return response if isinstance(response, str) else json.dumps(response)


def test_budget_explanation_has_the_same_bound_in_schema_and_host_validation():
    limit = MAX_ATTEMPT_BUDGET_REASON_CHARACTERS
    assert TASK_SPEC_RESPONSE_SCHEMA['properties']['attempt_budget_reason']['maxLength'] == limit
    payload = dict(selected_max_attempts=2, attempt_budget_reason='x' * limit)
    assert len(_parse_task_response(json.dumps(payload), 'Process input.').attempt_budget_reason) == limit
    payload['attempt_budget_reason'] += 'x'
    with pytest.raises(ValueError, match='attempt_budget_reason must fit'):
        _parse_task_response(json.dumps(payload), 'Process input.')


def test_schema_uses_exact_stdio_example_text_and_preserves_native_function_examples():
    wire = task_spec_response_schema('stdio')['properties']['examples']
    native = task_spec_response_schema('function')['properties']['examples']
    assert wire['maxItems'] == native['maxItems'] == 3
    assert wire['items']['properties']['input'] == {'type': 'string'}
    assert wire['items']['properties']['output'] == {'type': 'string'}
    assert native['items']['properties']['input'] == native['items']['properties']['output'] == {}
    assert wire['items']['properties']['input'] == {'type': 'string'}


def test_truncated_analysis_is_regenerated_without_copying_a_repeating_explanation():
    repeated = 'Repeated budget explanation. ' * 100
    class TruncatedClient(SequentialTaskAnalysisClient):
        def generate(self, *args, **kwargs):
            raw = super().generate(*args, **kwargs)
            self.last_usage = {'done_reason': 'length' if len(self.requests) == 1 else 'stop'}
            return raw
    client = TruncatedClient(['{"attempt_budget_reason":"' + repeated,
                              dict(problem_type='other', examples=[])])
    task = parse_task(client, 'Process input.', max_generation_attempts=2)
    assert task.selected_max_attempts == 2 and len(client.requests) == 2
    assert 'reached its output budget' in client.requests[1]['prompt']
    assert 'Repeated budget explanation' not in client.requests[1]['prompt']


def test_a_complete_valid_analysis_at_the_output_boundary_is_accepted():
    client = TaskAnalysisClient(dict(problem_type='other', examples=[]))
    client.last_usage = {'done_reason': 'length'}
    assert parse_task(client, 'Process input.').selected_max_attempts == 2
    assert len(client.requests) == 1


def test_format_errors_are_repaired_before_optional_suggestions_without_copying_the_bad_contract():
    wrong = dict(input_format='Invented delivery rule', input_format_origin='stated',
                 input_format_source_excerpt='', required_operations=['invented algorithm'])
    corrected = dict(input_format='One record per line', input_format_origin='selected',
                     input_format_source_excerpt='', output_format='One result per line',
                     output_format_origin='selected', output_format_source_excerpt='',
                     required_operations=[], examples=[])
    client = SequentialTaskAnalysisClient([wrong, corrected])
    task = parse_task(client, 'Process the supplied records.')
    assert task.input_format_origin == task.output_format_origin == 'selected'
    assert task.input_format == corrected['input_format']
    assert 'source excerpt must quote' in client.requests[1]['prompt']
    assert 'Invented delivery rule' not in client.requests[1]['prompt']
    assert 'An empty source_excerpt can NEVER have' in client.requests[0]['system']


def test_a_missing_stated_format_quote_is_not_silently_reclassified_as_selected():
    client = TaskAnalysisClient(dict(input_format='A model choice', input_format_origin='stated',
                                    input_format_source_excerpt=''))
    with pytest.raises(TaskFormatError, match='quote the original format'):
        _parse_task_response(client.generate(''), 'Process a value.')


def test_parses_complete_task_spec():
    client = TaskAnalysisClient({
        "problem_type": "sorting",
        "constraints": "1 <= n <= 100000",
        "input_format": "First line n, second line n integers",
        "output_format": "Print the sorted integers",
        "expected_complexity": "O(n log n)",
        "prohibited_operations": ["built-in sort()"],
        "required_operations": ["merge sort"],
        "edge_cases": ["n = 1", "duplicate values"],
        "examples": [{"input": "3\n3 1 1", "output": "1 1 3",
                      "source_excerpt": "Example input: 3\n3 1 1\nOutput: 1 1 3"}],
    })

    spec = parse_task(client, "Sort the values using merge sort without sort().\n"
                             "Example input: 3\n3 1 1\nOutput: 1 1 3")

    assert spec.problem_type == "sorting"
    assert spec.constraints == "1 <= n <= 100000"
    assert spec.input_format == "First line n, second line n integers"
    assert spec.output_format == "Print the sorted integers"
    assert spec.expected_complexity == "O(n log n)"
    assert spec.prohibited_operations == ["built-in sort()"]
    assert spec.required_operations == ["merge sort"]
    assert spec.edge_cases == ["n = 1", "duplicate values"]
    assert spec.selected_max_attempts == 2
    assert spec.attempt_budget_reason == "Test-selected attempt budget."
    assert spec.examples == [{"input": "3\n3 1 1", "output": "1 1 3",
                             "source_excerpt": "Example input: 3\n3 1 1\nOutput: 1 1 3"}]
    assert client.requests[0]["temperature"] == 0.0
    assert "prohibited_operations" in client.requests[0]["system"]


def test_legacy_response_gets_safe_defaults():
    client = TaskAnalysisClient({
        "problem_type": "math",
        "constraints": "two integers",
        "examples": [],
    })

    spec = parse_task(client, "Read two integers and print their sum.")

    assert spec.problem_type == "math"
    assert spec.constraints == "two integers"
    assert spec.input_format == ""
    assert spec.output_format == ""
    assert spec.expected_complexity == ""
    assert spec.prohibited_operations == []
    assert spec.required_operations == []
    assert spec.edge_cases == []


def test_invented_examples_are_retried_then_removed_by_the_analyzer():
    client = SequentialTaskAnalysisClient([
        {'examples': [{'input': {'count': 2}, 'output': {}}]},
        {'examples': []},
    ])
    spec = parse_task(client, 'Read a count and then that many values. Print their sum.')
    assert spec.examples == []
    assert 'source_excerpt must quote the original example' in client.requests[1]['prompt']


def test_example_answer_cannot_be_changed_while_reusing_a_real_source_quote():
    excerpt = 'Example input: 8 3. Output: 5.'
    client = TaskAnalysisClient({'examples': [
        dict(input='8 3', output='999', source_excerpt=excerpt),
    ]})
    problem = 'Subtract two integers. ' + excerpt
    with pytest.raises(ValueError, match='output is not grounded'):
        _parse_task_response(client.generate(''), problem)
    spec = parse_task(client, problem)
    assert spec.examples == []
    assert 'output is not grounded' in spec.discarded_inferred_examples[0]['reason']


def test_example_quote_cannot_be_invented_even_if_values_appear_in_constraints():
    client = TaskAnalysisClient({'examples': [
        dict(input='1', output='10', source_excerpt='Example input: 1. Output: 10.'),
    ]})
    spec = parse_task(client, 'Read an integer between 1 and 10.')
    assert spec.examples == []
    assert 'source_excerpt must quote' in spec.discarded_inferred_examples[0]['reason']


def test_final_repair_discards_both_inferred_operations_and_examples():
    example = dict(input='3 1 2 1', output='1 1 2',
                   source_excerpt='Example input: 3 1 2 1. Output: 1 1 2.')
    response = dict(problem_type='sorting', required_operations=['radix sort'],
                    examples=[example], input_format='Count then whitespace-separated values',
                    selected_max_attempts=6, attempt_budget_reason='A model-selected budget.')
    client = SequentialTaskAnalysisClient([response, dict(response, required_operations=[]), response])
    spec = parse_task(client, 'Sort the supplied values.')
    assert len(client.requests) == 3
    assert spec.examples == [] and spec.required_operations == []
    assert spec.selected_max_attempts == 6
    assert spec.input_format == response['input_format']
    assert spec.discarded_inferred_operations[0]['value'] == 'radix sort'
    assert spec.discarded_inferred_examples[0]['example_index'] == 1
    assert '3 1 2 1' not in json.dumps(spec.discarded_inferred_examples)


def test_final_repair_keeps_grounded_examples_and_discards_only_invented_ones():
    excerpt = 'Example input: 8 3. Output: 5.'
    grounded = dict(input='8 3', output='5', source_excerpt=excerpt)
    invented = dict(input='1 1', output='0', source_excerpt='Example input: 1 1. Output: 0.')
    spec = parse_task(TaskAnalysisClient({'examples': [grounded, invented]}),
                      'Subtract two integers. ' + excerpt)
    assert spec.examples == [grounded]
    assert [item['example_index'] for item in spec.discarded_inferred_examples] == [2]


def test_example_recovery_does_not_accept_invalid_required_analysis_fields():
    response = dict(examples=[dict(input='1', output='1', source_excerpt='Invented example')],
                    selected_max_attempts=0, attempt_budget_reason='Invalid budget.')
    with pytest.raises(ValueError, match='selected_max_attempts.*positive integer'):
        parse_task(TaskAnalysisClient(response), 'Return a value.')


@pytest.mark.parametrize('input_format,output_format', [
    ('JSON array of numbers', 'One decimal integer'),
    ('Comma-separated numbers', 'JSON number'),
])
def test_missing_io_formats_are_chosen_by_the_model_without_an_algorithm_default(input_format, output_format):
    response = dict(input_format=input_format, output_format=output_format)
    client = TaskAnalysisClient(response)
    spec = parse_task(client, 'Calculate the sum of supplied numbers.')
    assert spec.input_format == response['input_format']
    assert spec.output_format == response['output_format']


def test_format_selection_receives_the_actual_language_and_execution_mode():
    client = TaskAnalysisClient({})
    parse_task(client, 'Process native arguments.', language='scala', execution_mode='function', entry_point='solve')
    prompt = client.requests[0]['prompt']
    assert '"language": "scala"' in prompt and '"execution_mode": "function"' in prompt
    assert '"entry_point": "solve"' in prompt


def test_response_with_all_optional_fields_missing_is_supported():
    spec = parse_task(TaskAnalysisClient({}), "Print hello.")

    assert spec.problem_type == "other"
    assert spec.constraints == ""
    assert spec.examples == []


def test_accepts_different_llm_selected_attempt_budgets_by_problem():
    easy = parse_task(TaskAnalysisClient({
        "selected_max_attempts": 1,
        "attempt_budget_reason": "One direct output statement.",
    }), "Print hello.")
    complex_task = parse_task(TaskAnalysisClient({
        "selected_max_attempts": 7,
        "attempt_budget_reason": "Several states and edge cases may need repairs.",
    }), "Solve a state-transition problem.")

    assert easy.selected_max_attempts == 1
    assert complex_task.selected_max_attempts == 7
    assert easy.attempt_budget_reason != complex_task.attempt_budget_reason


@pytest.mark.parametrize(
    "invalid_budget",
    [None, 0, -1, 1.5, "3", True],
)
def test_rejects_invalid_llm_selected_attempt_budget(invalid_budget):
    client = TaskAnalysisClient({
        "selected_max_attempts": invalid_budget,
        "attempt_budget_reason": "Invalid test budget.",
    })

    with pytest.raises(
        ValueError,
        match="selected_max_attempts.*positive integer",
    ):
        parse_task(client, "Any programming problem")

    assert len(client.requests) == 3


def test_requires_attempt_budget_reason():
    client = TaskAnalysisClient({
        "selected_max_attempts": 2,
        "attempt_budget_reason": "",
    })

    with pytest.raises(
        ValueError,
        match="attempt_budget_reason.*non-empty string",
    ):
        parse_task(client, "Any programming problem")


def test_retries_malformed_edge_cases_and_accepts_corrected_task_spec():
    client = SequentialTaskAnalysisClient([
        {
            "problem_type": "string",
            "constraints": "input is non-empty",
            "edge_cases": [""],
        },
        {
            "problem_type": "string",
            "constraints": "input is non-empty",
            "edge_cases": ["single bracket", "incorrect closing order"],
        },
    ])

    spec = parse_task(client, "Determine whether brackets are balanced.")

    assert spec.edge_cases == ["single bracket", "incorrect closing order"]
    assert len(client.requests) == 2
    retry_prompt = client.requests[1]["prompt"]
    assert "previous TaskSpec response was rejected" in retry_prompt
    assert "field 'edge_cases' item 1 must be a non-empty string" in retry_prompt
    assert "Return the entire corrected JSON object" in retry_prompt


def test_structured_analysis_retries_invalid_description_and_keeps_empty_example():
    class StructuredClient(SequentialTaskAnalysisClient):
        def __init__(self, responses):
            super().__init__(responses)
            self.schemas = []

        def generate_json(self, prompt, system=None, temperature=0.2, schema=None):
            self.schemas.append(schema)
            return super().generate(prompt, system=system, temperature=temperature)

    client = StructuredClient([
        {"problem_type": "string", "edge_cases": [""]},
        {
            "problem_type": "string",
            "edge_cases": ["empty input line", "incorrect closing order"],
            "examples": [{"input": "", "output": "YES",
                          "source_excerpt": 'Example: input "" gives YES.'}],
        },
    ])

    spec = parse_task(client, 'Check bracket nesting. An empty line is valid. Example: input "" gives YES.')

    assert spec.edge_cases == ["empty input line", "incorrect closing order"]
    assert spec.examples == [{"input": "", "output": "YES",
                             "source_excerpt": 'Example: input "" gives YES.'}]
    assert len(client.requests) == 2
    assert client.schemas == [TASK_SPEC_RESPONSE_SCHEMA] * 2
    assert client.schemas[0]["properties"]["edge_cases"]["items"]["minLength"] == 1
    assert client.requests[0]["temperature"] == 0.0
    assert "non-empty string" in client.requests[1]["prompt"]


def test_retries_invented_required_operation_and_accepts_grounded_correction():
    client = SequentialTaskAnalysisClient([
        {
            "problem_type": "string",
            "required_operations": ["stack"],
            "edge_cases": ["incorrect closing order"],
        },
        {
            "problem_type": "string",
            "required_operations": [],
            "edge_cases": ["incorrect closing order"],
        },
    ])
    problem = "Determine whether a bracket string is balanced and properly nested."

    spec = parse_task(client, problem)

    assert spec.required_operations == []
    assert len(client.requests) == 2
    retry_prompt = client.requests[1]["prompt"]
    assert "required_operations" in retry_prompt
    assert "'stack'" in retry_prompt
    assert "not grounded in an explicit operation" in retry_prompt


def test_accepts_operations_explicitly_grounded_in_problem_statement():
    client = TaskAnalysisClient({
        "problem_type": "other",
        "prohibited_operations": ["built-in sorting functions"],
        "required_operations": ["recursion"],
    })
    problem = "Use a recursive solution without built-in sort() or sorted()."

    spec = parse_task(client, problem)

    assert spec.prohibited_operations == ["built-in sorting functions"]
    assert spec.required_operations == ["recursion"]
    assert spec.discarded_inferred_operations == []
    assert len(client.requests) == 1


def test_discards_repeated_ungrounded_operation_after_bounded_retries():
    response = {
        "problem_type": "string",
        "required_operations": ["stack"],
        "edge_cases": ["incorrect closing order"],
    }
    client = TaskAnalysisClient(response)

    spec = parse_task(
        client,
        "Determine whether a bracket string is balanced and properly nested.",
    )

    assert spec.required_operations == []
    assert spec.discarded_inferred_operations == [{
        "field": "required_operations",
        "value": "stack",
        "reason": "not explicitly grounded in the problem statement",
    }]
    assert len(client.requests) == 3
    assert "remove that operation" in client.requests[1]["prompt"]


def test_task_spec_retry_is_bounded_and_reports_last_validation_error():
    client = TaskAnalysisClient({"edge_cases": [""]})

    with pytest.raises(
        ValueError,
        match=(
            "Could not generate a valid TaskSpec after 3 attempts: .*"
            "field 'edge_cases' item 1"
        ),
    ):
        parse_task(client, "Determine whether brackets are balanced.")

    assert len(client.requests) == 3


def test_task_spec_retry_context_caps_rejected_response():
    invalid_response = "x" * 3_000
    client = SequentialTaskAnalysisClient([
        invalid_response,
        {"problem_type": "string", "edge_cases": ["single character"]},
    ])

    parse_task(client, "Read a string.")

    retry_prompt = client.requests[1]["prompt"]
    assert "x" * 2_000 in retry_prompt
    assert "x" * 2_001 not in retry_prompt


def test_rejects_invalid_task_spec_attempt_limit():
    client = TaskAnalysisClient({})

    with pytest.raises(ValueError, match="max_generation_attempts must be at least 1"):
        parse_task(client, "Print hello.", max_generation_attempts=0)

    assert client.requests == []


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ([], "TaskSpec response must be a JSON object"),
        ({"input_format": []}, "field 'input_format' must be a string"),
        (
            {"prohibited_operations": "sort()"},
            "field 'prohibited_operations' must be a list of strings",
        ),
        (
            {"required_operations": ["recursion", 3]},
            "field 'required_operations' item 2 must be a non-empty string",
        ),
        ({"edge_cases": [""]}, "field 'edge_cases' item 1"),
        ({"examples": {}}, "field 'examples' must be a list"),
        ({"examples": ["1 -> 1"]}, "example 1 must be an object"),
        (
            {"examples": [{"input": "1"}]},
            "example 1 must contain 'input' and 'output'",
        ),
    ],
)
def test_rejects_malformed_task_spec_fields(response, message):
    with pytest.raises(ValueError, match=message):
        parse_task(TaskAnalysisClient(response), "Any programming problem")


def test_http_analysis_does_not_require_stdin_serialization_quotes():
    client=TaskAnalysisClient(dict(problem_type='other',input_format='HTTP requests',
        input_format_origin='stated',input_format_source_excerpt='',examples=[]))
    task=parse_task(client,'Implement the supplied HTTP API.',language='scala',execution_mode='http')
    assert len(client.requests)==1
    assert task.input_format_origin=='selected'
    assert 'original API contract' in task.input_format
    assert 'No input_format/output_format fields' in client.requests[0]['system']


def test_selected_delivery_rejects_unspecified_content_placeholders():
    from src.agents.task_spec import _parse_stdio_contracts, TaskFormatError
    contract = dict(explicit_in_statement=False, quoted_rules='', serialization='plain text',
                    layout='header then sequence content', whitespace='tokens separated by spaces',
                    empty_values='not allowed')
    with pytest.raises(TaskFormatError, match='layout is ambiguous'):
        _parse_stdio_contracts(json.dumps(dict(stdin=contract, stdout=dict(contract, layout='one integer'))),
                               'Process a counted sequence.')
    contract['layout'] = 'First line: length. Next line: length space-separated integer tokens.'
    result = _parse_stdio_contracts(json.dumps(dict(stdin=contract, stdout=dict(contract, layout='one integer'))),
                                   'Process a counted sequence.')
    assert 'space-separated' in result['input_format']


def test_format_quote_recovers_exact_source_after_whitespace_only_changes():
    from src.agents.task_spec import _parse_format
    original = 'Read the first record. Then read the remaining records.'
    value, origin, quote = _parse_format(dict(input_format='ignored paraphrase',
        input_format_origin='stated', input_format_source_excerpt='Read the first record.\nThen read the remaining records.'),
        'input', original)
    assert value == quote == original
    assert origin == 'stated'
