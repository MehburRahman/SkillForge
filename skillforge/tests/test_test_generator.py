import json

import pytest

from src.agents.task_spec import TaskSpec
from src.agents.test_review import test_review_schema as review_schema
from src.agents.test_generator import (
    TEST_SUITE_RESPONSE_SCHEMA,
    generate_tests,
    load_test_suite,
)


class ResponseClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0
        self.prompts = []
        self.last_cases = None

    def generate(self, prompt, system=None, temperature=0.2):
        self.calls += 1
        self.prompts.append(prompt)
        if "review generated programming tests" in (system or ""):
            return json.dumps([
                dict(test_number=i, input_valid=True, output_verified=True,
                     expected_output=case['expected_output'], category_valid=True,
                     reason='Fixture review of the input and answer.')
                for i, case in enumerate(self.last_cases, 1)
            ])
        response = next(self.responses)
        try:
            self.last_cases = json.loads(response)
        except ValueError:
            pass
        return response


class StructuredResponseClient:
    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.schema = None
        self.schemas = []

    def generate_json(
        self,
        prompt,
        system=None,
        temperature=0.2,
        schema=None,
    ):
        self.calls += 1
        self.schema = schema
        self.schemas.append(schema)
        if schema == review_schema(len(json.loads(self.response))):
            return json.dumps([
                dict(test_number=i, input_valid=True, output_verified=True,
                     expected_output=case['expected_output'], category_valid=True,
                     reason='Fixture review of the input and answer.')
                for i, case in enumerate(json.loads(self.response), 1)
            ])
        return self.response


def addition_spec():
    return TaskSpec(
        problem_statement="Read two integers and print their sum.",
        problem_type="math",
        constraints="exactly two integers",
        examples=[],
    )


def test_retries_after_invalid_tests():
    invalid = json.dumps([
        {"input": "", "expected_output": None, "category": "edge"},
    ])
    valid = json.dumps([
        {"input": "2 3", "expected_output": "5", "category": "normal"},
    ])
    client = ResponseClient([invalid, valid])

    suite = generate_tests(client, addition_spec())

    assert client.calls == 3
    assert suite.cases[0].input == "2 3"
    assert "previous test response was rejected" in client.prompts[1].lower()


def test_uses_native_json_schema_for_generated_tests():
    response = json.dumps([
        {
            "input": "2 3",
            "expected_output": "5",
            "category": "normal",
        },
    ])
    client = StructuredResponseClient(response)

    suite = generate_tests(client, addition_spec())

    assert client.calls == 2
    assert client.schemas == [TEST_SUITE_RESPONSE_SCHEMA, review_schema(1)]
    assert suite.cases[0].expected_output == "5"


def test_stops_after_two_retries():
    client = ResponseClient(["not json", "still invalid", "also invalid"])

    with pytest.raises(ValueError, match="after 3 attempts"):
        generate_tests(client, addition_spec())

    assert client.calls == 3


def test_rejects_duplicate_inputs():
    duplicate = json.dumps([
        {"input": "2 3", "expected_output": "5", "category": "normal"},
        {"input": "2 3", "expected_output": "5", "category": "edge"},
    ])
    client = ResponseClient([duplicate])

    with pytest.raises(ValueError, match="duplicates an earlier input"):
        generate_tests(client, addition_spec(), max_generation_attempts=1)


def test_accepts_generated_empty_line_when_problem_explicitly_allows_it():
    client = ResponseClient([json.dumps([
        {"input": "\n", "expected_output": "YES", "category": "edge"},
        {"input": "([])", "expected_output": "YES", "category": "normal"},
    ])])
    spec = TaskSpec(
        problem_statement="Check bracket nesting. An empty line is valid.",
        problem_type="string",
        constraints="",
    )

    suite = generate_tests(client, spec)

    assert suite.cases[0].input == "\n"
    assert suite.cases[0].expected_output == "YES"


@pytest.mark.parametrize("contract", ["Read a non-empty line.", "An empty line is invalid."])
def test_rejects_generated_empty_line_when_problem_does_not_allow_it(contract):
    client = ResponseClient([json.dumps([
        {"input": "\n", "expected_output": "YES", "category": "edge"},
    ])])
    spec = TaskSpec(problem_statement=contract, problem_type="string", constraints="")

    with pytest.raises(ValueError, match="empty input not allowed"):
        generate_tests(client, spec, max_generation_attempts=1)


def test_loads_trusted_tests(tmp_path):
    path = tmp_path / "tests.json"
    path.write_text(json.dumps([
        {"input": "2 3", "expected_output": "5", "category": "normal"},
    ]))

    suite = load_test_suite(path)

    assert len(suite.cases) == 1
    assert suite.cases[0].expected_output == "5"
