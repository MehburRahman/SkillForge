import json
import math
import os
import re
import time

import httpx

from src.experiments import request_seed, validate_seed

try:
    import ollama
except Exception as exc:  # pragma: no cover - depends on environment
    ollama = None
    _OLLAMA_IMPORT_ERROR = exc
else:
    _OLLAMA_IMPORT_ERROR = None


DEFAULT_MODEL = "qwen2.5-coder:7b"
# Different family from the Qwen generator: a model must not grade its own family.
# llama3.1:8b ranked correct above broken Scala solutions where granite4.2:8b inverted them.
DEFAULT_JUDGE_MODEL = "llama3.1:8b"
DEFAULT_CONTEXT_WINDOW = 8192
DEFAULT_TEST_MODEL_OUTPUT_TOKENS = 4096
DEFAULT_ANALYSIS_OUTPUT_TOKENS = 2048
DEFAULT_TASK_FORMAT_OUTPUT_TOKENS = 1024
# Local 14B models write a few tokens per second; a several-hundred-line
# program or test plan can take well over five minutes on a 16 GB laptop.
DEFAULT_REQUEST_TIMEOUT_SECONDS = 900
# Room kept for the answer when a call sets no output limit, and the largest
# context the client will grow to so a long prompt cannot crowd out the reply.
OUTPUT_RESERVE_TOKENS = 4096
MAX_CONTEXT_WINDOW = 32768


def fitted_context_window(options, messages):
    """Return a larger num_ctx when prompt + expected output would not fit, else None.

    Ollama silently drops the oldest prompt tokens or stops the reply early when a
    request outgrows num_ctx, which shows up as truncated code or ignored repair
    evidence. The prompt size is estimated conservatively at three characters per token.
    """
    num_ctx = options.get('num_ctx')
    if not isinstance(num_ctx, int):
        return None
    prompt_tokens = sum(len(m.get('content') or '') for m in messages) // 3 + 64
    output = options.get('num_predict')
    output = output if isinstance(output, int) and output > 0 else OUTPUT_RESERVE_TOKENS
    needed = prompt_tokens + output
    if needed <= num_ctx or num_ctx >= MAX_CONTEXT_WINDOW:
        return None
    size = num_ctx
    while size < needed and size < MAX_CONTEXT_WINDOW:
        size *= 2
    return min(size, MAX_CONTEXT_WINDOW)


class LLMClient:
    analysis_output_budget = DEFAULT_ANALYSIS_OUTPUT_TOKENS
    task_format_output_budget = DEFAULT_TASK_FORMAT_OUTPUT_TOKENS
    structured_test_review = True
    executable_input_validation = True
    structured_test_inputs = True
    supports_source_edit_repairs = True
    source_edit_style = 'lines'
    disable_thinking = True

    def __init__(self, model=DEFAULT_MODEL, *, context_window=None, request_timeout=None):
        if context_window is not None and (type(context_window) is not int or not 1024 <= context_window <= 32768):
            raise ValueError('context_window must be an integer between 1024 and 32768')
        self.model = model
        self.generation_options = {'num_ctx': context_window} if context_window is not None else {}
        if request_timeout is None:
            try:
                request_timeout = float(os.environ.get('SKILLFORGE_LLM_TIMEOUT_SECONDS', DEFAULT_REQUEST_TIMEOUT_SECONDS))
            except ValueError as exc:
                raise ValueError('SKILLFORGE_LLM_TIMEOUT_SECONDS must be a positive finite number') from exc
        if (isinstance(request_timeout, bool) or not isinstance(request_timeout, (int, float))
                or not math.isfinite(request_timeout) or request_timeout <= 0):
            raise ValueError('request_timeout must be a positive finite number of seconds')
        self.request_timeout = request_timeout
        self._backend = None
        self._unload_after_call = None

    def for_tests(self):
        """Use the same model with separate, bounded test-generation settings."""
        client = LLMClient(self.model, request_timeout=self.request_timeout)
        client.generation_options = {'num_predict': DEFAULT_TEST_MODEL_OUTPUT_TOKENS, **self.generation_options}
        return client

    UNLOAD_PARAMETER_BILLIONS = 10.0

    def _should_unload(self):
        """True for models of at least UNLOAD_PARAMETER_BILLIONS parameters (checked once)."""
        if self._unload_after_call is None:
            self._unload_after_call = False
            try:
                details = self._backend.show(self.model).get('details') or {}
                size = str(details.get('parameter_size', '')).strip().upper()
                if size.endswith('B'):
                    self._unload_after_call = float(size[:-1]) >= self.UNLOAD_PARAMETER_BILLIONS
            except Exception:
                pass
        return self._unload_after_call

    def generate(self, prompt, system=None, temperature=0.2, *, seed=None, max_output_tokens=None):
        return self._generate(
            prompt,
            system=system,
            temperature=temperature,
            seed=seed,
            max_output_tokens=max_output_tokens,
        )

    def generate_json(
        self, prompt, system=None, temperature=0.0, schema=None, *, seed=None, max_output_tokens=None,
    ):
        """Generate JSON using Ollama's native JSON/JSON-schema mode."""
        return self._generate(
            prompt,
            system=system,
            temperature=temperature,
            response_format=schema or "json",
            seed=seed,
            max_output_tokens=max_output_tokens,
        )

    def _generate(
        self,
        prompt,
        system=None,
        temperature=0.2,
        response_format=None,
        seed=None,
        max_output_tokens=None,
    ):
        validate_seed(seed)
        if max_output_tokens is not None and (type(max_output_tokens) is not int
                                              or not 1 <= max_output_tokens <= 32768):
            raise ValueError('max_output_tokens must be an integer between 1 and 32768')
        if ollama is None:
            return self._fallback_response(prompt, system)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        self.last_usage = {}

        try:
            request = dict(
                model=self.model,
                messages=messages,
                options={**self.generation_options, "temperature": temperature},
            )
            if self.disable_thinking:
                # Reasoning models (e.g. granite4.2) otherwise spend the whole
                # output budget on hidden thinking and return empty content.
                request['think'] = False
            if response_format is not None:
                request["format"] = response_format
            if max_output_tokens is not None:
                request['options']['num_predict'] = max_output_tokens
            if seed is not None:
                request["options"]["seed"] = seed
            grown = fitted_context_window(request['options'], messages)
            if grown is not None:
                request['options']['num_ctx'] = grown
            if self._backend is None:
                self._backend = ollama.Client(timeout=httpx.Timeout(self.request_timeout,
                                                                   connect=min(10, self.request_timeout)))
            if self._should_unload():
                # A large model left resident starves the Docker/compiler
                # sandbox on a 16 GB machine. Release it after each response.
                request['keep_alive'] = 0
            retries = []
            transient_retries = 0
            while True:
                try:
                    response = self._backend.chat(**request)
                    break
                except Exception as exc:
                    message = str(exc).lower()
                    status = getattr(exc, 'status_code', None)
                    if 'think' in request and 'think' in message and status in (None, 400):
                        # Older servers or models that reject the flag: send without it.
                        request.pop('think')
                        retries.append(dict(kind='thinking_flag_fallback', error=str(exc)[:500]))
                        continue
                    if ('format' in request and status in (None, 400)
                            and any(word in message for word in ('grammar', 'schema', 'format', 'sampler'))):
                        previous_format = request['format']
                        if isinstance(previous_format, dict):
                            request['messages'] = [dict(m) for m in request['messages']]
                            request['messages'][-1]['content'] += (
                                '\nReturn JSON matching this schema (host validation still applies):\n'
                                + json.dumps(previous_format))
                            request['format'] = 'json'
                        else:
                            request.pop('format')
                        retries.append(dict(kind='structured_output_fallback', error=str(exc)[:500]))
                        continue
                    if (isinstance(exc, (httpx.TimeoutException, httpx.TransportError))
                            or status in (429, 502, 503, 504)) and transient_retries < 2:
                        transient_retries += 1
                        retries.append(dict(kind='transport_retry', error=str(exc)[:500]))
                        time.sleep(0.25 * transient_retries)
                        continue
                    self.last_usage = {'infrastructure_retries': retries, 'backend_request_count': len(retries) + 1}
                    raise
            self.last_usage = {
                field: response.get(field) if isinstance(response, dict) else getattr(response, field, None)
                for field in ('prompt_eval_count', 'eval_count', 'total_duration', 'done_reason')
            }
            self.last_usage['infrastructure_retries'] = retries
            if grown is not None:
                self.last_usage['context_window'] = grown
            self.last_usage['backend_request_count'] = len(retries) + 1
            return response["message"]["content"]
        except httpx.TimeoutException as exc:
            raise RuntimeError(
                f'LLM request for {self.model} timed out after {self.request_timeout:g} seconds. '
                'Check local model activity and memory pressure, or increase SKILLFORGE_LLM_TIMEOUT_SECONDS.'
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f'LLM client failed to generate a response: {type(exc).__name__}: {exc}'
            ) from exc

    def _fallback_response(self, prompt, system=None):
        """
        Fallback responses when LLM is not available.
        Raise an error instead of returning bogus data that corrupts testing.
        """
        if ollama is None:
            raise RuntimeError(
                f"ollama module is not available. Cannot generate response. "
                f"Original import error: {_OLLAMA_IMPORT_ERROR}"
            )
        raise RuntimeError(
            "LLM client failed to generate a response. Check your LLM service is running."
        )


def validate_judge_pair(generator, judge):
    """Describe whether an advisory judge shares the generator's provenance.

    Recognizes the plan's Qwen/Llama families across versions, sizes and
    namespaces, and matching base names for other models. Arbitrary local
    aliases do not disclose their family. This describes model families only;
    using a different family is not an independent correctness proof either.
    """
    if judge is None:
        return
    if generator is judge:
        raise ValueError('generator and judge require separate client instances')

    def family(client):
        model = getattr(client, "model", None)
        if not isinstance(model, str) or not model.strip():
            return None
        name = model.strip().lower().rsplit("/", 1)[-1].split(":", 1)[0]
        known = re.match(r"(qwen|llama)(?=$|[\d._-])", name)
        return known.group(1) if known else name

    generator_family, judge_family = family(generator), family(judge)
    # A same-family judge is permitted as advisory evidence. Keep its provenance
    # explicit; it cannot supply independent verification for generated tests.
    return dict(independent_family=generator_family != judge_family,
                same_model=getattr(generator, 'model', None) == getattr(judge, 'model', None))


def build_clients(model=None, judge_model=None, llm_judge=False, *, context_window=None):
    """Return (generator_client, judge_client) for one run.

    The judge is a separate client. It may share the generator's model, in
    which case its opinions are correlated. The judge client is None when the judge
    is off, and naming a judge model is enough to turn it on.
    """
    generator = LLMClient(model or DEFAULT_MODEL, context_window=context_window)
    if not (llm_judge or judge_model):
        return generator, None
    judge = LLMClient(judge_model or DEFAULT_JUDGE_MODEL, context_window=context_window)
    validate_judge_pair(generator, judge)
    return generator, judge


class RecordingClient:
    """Persist calls immediately, including failures before a run is created.

    Only callers with visible data use this wrapper. Hidden execution never
    calls the model and is not recorded here.
    """

    def __init__(self, inner, store, owner_id, role):
        self._inner = inner
        self._store, self._owner_id, self._role = store, owner_id, role
        self._recording_stage = "default"

    def set_recording_stage(self, stage):
        self._recording_stage = stage

    def __getattr__(self, name):
        method = getattr(self._inner, name)
        if name not in ("generate", "generate_json") or not callable(method):
            return method

        def recorded(prompt, *args, **kwargs):
            started = time.perf_counter()
            response, error = None, None
            request = {"method": name, "prompt": prompt, "arguments": list(args), **kwargs,
                       "model": getattr(self._inner, "model", "unknown"),
                       "options": dict(getattr(self._inner, "generation_options", {}))}
            try:
                response = method(prompt, *args, **kwargs)
                return response
            except BaseException as exc:
                error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                self._store.record_model_request(
                    self._owner_id, self._role, self._recording_stage, request,
                    response, error, getattr(self._inner, "last_usage", {}),
                    time.perf_counter() - started,
                )
        return recorded


class CallCountingClient:
    """Count calls and optionally inject seeds at the model-client boundary.

    The counter sits at the client boundary rather than inside LLMClient so
    that it also counts calls made by test doubles, and so that the retries
    inside the judge and the test generator are counted without either
    module knowing about it.

    Attribute access forwards to the wrapped client, which matters for more
    than convenience: the judge and the test generator both branch on
    whether `generate_json` exists. Forwarding through __getattr__ keeps a
    missing method missing instead of pushing a `generate`-only client down
    the JSON path.

    Seeded clients must accept the `seed` keyword. A stage gets its own
    reproducible sequence, including retries; unseeded calls retain the
    original method signature for compatibility with older clients.
    """

    def __init__(self, inner, *, seed=None, task_key=""):
        # Bind first, so no attribute lookup reaches __getattr__ before
        # _inner exists and recurses forever.
        self._inner = inner
        validate_seed(seed)
        self._seed = seed
        self._task_key = task_key
        self.set_stage("default")
        self.call_count = 0

    def set_stage(self, stage):
        """Start an independent seed sequence so retries cannot shift later stages."""
        self._stage = stage
        self._stage_calls = 0
        recorder = getattr(self._inner, "set_recording_stage", None)
        if recorder is not None:
            recorder(stage)

    def __getattr__(self, name):
        attribute = getattr(self._inner, name)
        if name in ("generate", "generate_json") and callable(attribute):
            return self._counted(attribute)
        return attribute

    def _counted(self, method):
        def call(*args, **kwargs):
            if self._seed is not None:
                kwargs["seed"] = request_seed(
                    self._seed, self._task_key, self._stage, self._stage_calls,
                )
            self._stage_calls += 1
            # Count before delegating, so a call that raises still counts as
            # compute spent.
            self.call_count += 1
            return method(*args, **kwargs)

        return call
