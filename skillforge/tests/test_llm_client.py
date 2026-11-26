from types import SimpleNamespace

import pytest

import src.agents.llm_client as llm_client_module
from src.agents.llm_client import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_MODEL,
    CallCountingClient,
    LLMClient,
    build_clients,
)


def _ollama_stub(chat):
    return SimpleNamespace(Client=lambda **kwargs: SimpleNamespace(chat=chat))


def test_generate_json_passes_schema_to_ollama(monkeypatch):
    captured = {}

    def fake_chat(**request):
        captured.update(request)
        return {"message": {"content": '{"answer":1}'}}

    monkeypatch.setattr(
        llm_client_module,
        "ollama",
        _ollama_stub(fake_chat),
    )
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "number"}},
        "required": ["answer"],
    }

    response = LLMClient("test-model").generate_json(
        "Return JSON",
        system="JSON only",
        schema=schema,
    )

    assert response == '{"answer":1}'
    assert captured["model"] == "test-model"
    assert captured["format"] == schema
    assert captured["options"] == {"temperature": 0.0}
    assert captured["messages"] == [
        {"role": "system", "content": "JSON only"},
        {"role": "user", "content": "Return JSON"},
    ]


def test_backend_failure_keeps_its_diagnostic_and_original_cause(monkeypatch):
    error = ValueError('Failed to initialize samplers: failed to parse grammar')

    def reject(**request):
        raise error

    monkeypatch.setattr(llm_client_module, 'ollama', _ollama_stub(reject))
    with pytest.raises(RuntimeError, match='failed to parse grammar') as caught:
        LLMClient('fixture-model').generate_json('Return JSON')
    assert caught.value.__cause__ is error


def test_explicit_context_is_sent_for_text_and_json_and_remains_reproducible(monkeypatch):
    requests = []
    monkeypatch.setattr(llm_client_module, 'ollama', _ollama_stub(
        lambda **kwargs: requests.append(kwargs) or {'message': {'content': '{}'}}))
    client = LLMClient('test', context_window=8192)
    client.generate('repair', seed=42)
    client.generate_json('analysis', seed=0)
    assert [r['options']['num_ctx'] for r in requests] == [8192, 8192]
    assert [r['options']['seed'] for r in requests] == [42, 0]


def test_explicit_output_budget_is_sent_to_ollama_without_mutating_client_options(monkeypatch):
    requests = []
    monkeypatch.setattr(llm_client_module, 'ollama', _ollama_stub(
        lambda **kwargs: requests.append(kwargs) or {'message': {'content': 'ok'}}))
    client = LLMClient('test', context_window=8192)
    client.generate('plan', max_output_tokens=1500)
    client.generate_json('review', max_output_tokens=500)
    client.generate('legacy')
    assert requests[0]['options']['num_predict'] == 1500
    assert requests[1]['options']['num_predict'] == 500
    assert 'num_predict' not in requests[2]['options']
    assert client.generation_options == {'num_ctx': 8192}


def test_same_model_test_client_has_a_budget_without_limiting_candidate_generation(monkeypatch):
    requests = []
    monkeypatch.setattr(llm_client_module, 'ollama', _ollama_stub(
        lambda **kwargs: requests.append(kwargs) or {'message': {'content': '{}'}}))
    solver = LLMClient('candidate:7b', context_window=8192)
    tests = solver.for_tests()
    tests.generate_json('Generate test data')
    solver.generate('Generate candidate source')
    assert requests[0]['model'] == requests[1]['model'] == 'candidate:7b'
    assert requests[0]['options']['num_predict'] == 4096
    assert 'num_predict' not in requests[1]['options']
    assert solver.generation_options == {'num_ctx': 8192}


@pytest.mark.parametrize('budget', [True, 0, -1, 1.5, 32769])
def test_invalid_output_budget_is_rejected_before_model_calls(budget):
    with pytest.raises(ValueError, match='max_output_tokens'):
        LLMClient('test').generate('plan', max_output_tokens=budget)


@pytest.mark.parametrize('context', [True, 0, 4096.0, 65536])
def test_invalid_context_is_rejected_before_model_calls(context):
    with pytest.raises(ValueError, match='context_window'):
        LLMClient(context_window=context)


def test_judge_defaults_to_a_different_family_with_a_separate_client():
    generator, judge = build_clients(llm_judge=True)

    assert generator.model == DEFAULT_MODEL
    assert judge.model == DEFAULT_JUDGE_MODEL
    assert generator.model.split(':')[0] != judge.model.split(':')[0]
    assert generator is not judge


def test_no_judge_client_when_the_judge_is_off():
    generator, judge = build_clients()

    assert generator.model == DEFAULT_MODEL
    assert judge is None


def test_naming_a_judge_model_enables_the_judge():
    generator, judge = build_clients(
        model="qwen2.5-coder:3b",
        judge_model="llama3.2:3b",
        llm_judge=False,
    )

    assert generator.model == "qwen2.5-coder:3b"
    assert judge.model == "llama3.2:3b"


@pytest.mark.parametrize("generator, judge", [
    ("qwen2.5-coder:7b", "qwen2.5-coder:7b"),
    ("qwen2.5-coder:7b", "qwen2.5:3b"),
    ("qwen2.5-coder:7b", "namespace/Qwen3:8b"),
    ("llama3.1:8b", "llama3.2:3b"),
    ("custom-model", "custom-model:latest"),
])
def test_allows_same_family_advisory_judge(generator, judge):
    solver, reviewer = build_clients(model=generator, judge_model=judge)
    assert solver is not reviewer
    assert not llm_client_module.validate_judge_pair(solver, reviewer)['independent_family']


class _GenerateOnlyClient:
    model = "generate-only:1b"

    def __init__(self):
        self.prompts = []

    def generate(self, prompt, system=None, temperature=0.2):
        self.prompts.append(prompt)
        return "ok"


class _JsonCapableClient(_GenerateOnlyClient):
    def generate_json(self, prompt, system=None, temperature=0.0, schema=None):
        self.prompts.append(prompt)
        return "{}"


def test_counts_both_generate_and_generate_json():
    counted = CallCountingClient(_JsonCapableClient())

    counted.generate("one")
    counted.generate_json("two")
    counted.generate("three")

    assert counted.call_count == 3


def test_forwards_attributes_of_the_wrapped_client():
    inner = _GenerateOnlyClient()
    counted = CallCountingClient(inner)

    assert counted.model == "generate-only:1b"
    counted.generate("one")
    assert inner.prompts == ["one"]


def test_hides_generate_json_when_the_wrapped_client_lacks_it():
    """The judge and the test generator branch on this attribute existing."""
    counted = CallCountingClient(_GenerateOnlyClient())

    assert getattr(counted, "generate_json", None) is None
    with pytest.raises(AttributeError):
        counted.generate_json


def test_counts_a_call_that_raises():
    class Failing:
        def generate(self, prompt, system=None, temperature=0.2):
            raise RuntimeError("model unavailable")

    counted = CallCountingClient(Failing())
    with pytest.raises(RuntimeError):
        counted.generate("one")

    assert counted.call_count == 1


@pytest.mark.parametrize("method", ["generate", "generate_json"])
def test_seed_reaches_ollama_for_text_and_json(monkeypatch, method):
    requests = []

    def fake_chat(**request):
        requests.append(request)
        return {"message": {"content": "{}"}}

    monkeypatch.setattr(llm_client_module, "ollama", _ollama_stub(fake_chat))
    getattr(LLMClient("test"), method)("prompt", seed=0)
    assert requests[0]["options"]["seed"] == 0


def test_stage_seeds_repeat_across_clients_and_change_with_task_and_base_seed(monkeypatch):
    requests = []

    def fake_chat(**request):
        requests.append(request)
        return {"message": {"content": "{}"}}

    monkeypatch.setattr(llm_client_module, "ollama", _ollama_stub(fake_chat))

    def seeds(base, task):
        requests.clear()
        client = CallCountingClient(LLMClient(), seed=base, task_key=task)
        client.set_stage("analysis")
        client.generate_json("analyze")
        client.generate_json("retry")
        client.set_stage("candidate:1")
        client.generate("code")
        return [r["options"]["seed"] for r in requests]

    baseline = seeds(42, "task-a")
    assert seeds(42, "task-a") == baseline
    assert seeds(43, "task-a") != baseline
    assert seeds(42, "task-b") != baseline
    assert len(set(baseline)) == 3


def test_thinking_is_disabled_so_reasoning_models_return_content(monkeypatch):
    captured = {}

    def fake_chat(**request):
        captured.update(request)
        return {"message": {"content": "ok"}}

    monkeypatch.setattr(llm_client_module, "ollama", _ollama_stub(fake_chat))
    assert LLMClient("granite4.2:8b").generate("Say ok") == "ok"
    assert captured["think"] is False


def test_a_rejected_thinking_flag_is_dropped_and_retried(monkeypatch):
    requests = []

    def fake_chat(**request):
        requests.append(dict(request))
        if "think" in request:
            raise ValueError('"model" does not support thinking')
        return {"message": {"content": "ok"}}

    monkeypatch.setattr(llm_client_module, "ollama", _ollama_stub(fake_chat))
    assert LLMClient("old-model").generate("Say ok") == "ok"
    assert "think" in requests[0] and "think" not in requests[1]


def test_large_models_are_unloaded_after_each_call():
    from src.agents.llm_client import LLMClient

    class Backend:
        def __init__(self, size):
            self.size, self.requests = size, []

        def show(self, model):
            return {'details': {'parameter_size': self.size}}

        def chat(self, **request):
            self.requests.append(request)
            return {'message': {'content': 'ok'}}

    for size, unloads in (('14.8B', True), ('7.6B', False)):
        client = LLMClient('m')
        client._backend = Backend(size)
        client.generate('hi')
        assert ('keep_alive' in client._backend.requests[0]) is unloads


def test_context_window_grows_when_prompt_and_output_do_not_fit():
    from src.agents.llm_client import LLMClient, fitted_context_window

    assert fitted_context_window({'num_ctx': 8192, 'num_predict': 1024}, [{'content': 'x' * 3000}]) is None
    assert fitted_context_window({'num_ctx': 8192}, [{'content': 'x' * 21000}]) == 16384
    assert fitted_context_window({'num_ctx': 8192, 'num_predict': 8192}, [{'content': 'x' * 2400}]) == 16384
    assert fitted_context_window({'num_ctx': 8192}, [{'content': 'x' * 400000}]) == 32768
    assert fitted_context_window({}, [{'content': 'x' * 400000}]) is None

    class Backend:
        requests = []

        def show(self, model):
            return {'details': {'parameter_size': '7.6B'}}

        def chat(self, **request):
            self.requests.append(request)
            return {'message': {'content': 'ok'}, 'done_reason': 'stop'}

    client = LLMClient('m', context_window=8192)
    client._backend = Backend()
    client.generate('x' * 21000)
    assert client._backend.requests[0]['options']['num_ctx'] == 16384
    assert client.last_usage['context_window'] == 16384
