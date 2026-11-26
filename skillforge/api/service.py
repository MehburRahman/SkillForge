from src.evolution.scope import validate_evolution_scope
from dataclasses import asdict
import hashlib
import json
import threading
from uuid import uuid4

from fastapi.encoders import jsonable_encoder

from api.models import EvolutionRequest, SolveRequest
from src.agents.llm_client import LLMClient, build_clients, validate_judge_pair
from src.agents.task_spec_cache import TaskSpecCache
from src.agents.test_generator import build_test_suite, validate_generation_tests, reference_cases
from src.benchmark.loader import load_benchmark, select_task
from src.benchmark.runner import select_benchmark_tasks
from src.evolution.prompt_registry import PromptRegistry
from src.evolution.workflow import run_evolution_round
from src.evolution.trigger import evolution_readiness
from src.experiments import preflight_execution, validate_experiment, validate_generation_execution
from src.history import ExperimentStore
from src.reinforcement_loop import run_candidate_refinement_loop


class Service:
    def __init__(self, settings, client_factory=build_clients, meta_factory=LLMClient, test_client_factory=LLMClient):
        self.settings = settings
        self.client_factory, self.meta_factory = client_factory, meta_factory
        self.test_client_factory = test_client_factory
        self._catalogs = {}
        self._catalog_lock = threading.Lock()
        identity = {"protocol": 1, "history": str(settings.history_db), "registry": str(settings.registry),
                    "catalogs": {k: str(v) for k, v in settings.catalogs.items()}}
        self.context = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    def catalog(self, name):
        if name not in self.settings.catalogs:
            raise KeyError(f"unknown catalog: {name}")
        path = self.settings.catalogs[name]
        with self._catalog_lock:
            stat = path.stat()
            key = (stat.st_mtime_ns, stat.st_size)
            cached = self._catalogs.get(name)
            if cached is None or cached[0] != key:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                cached = (key, digest, load_benchmark(path))
                self._catalogs[name] = cached
            return cached[1], cached[2]

    def clients(self, request):
        generator, judge = self.client_factory(model=request.model, judge_model=request.judge_model,
                                                llm_judge=request.llm_judge)
        validate_judge_pair(generator, judge)
        for client in (generator, judge):
            if client is not None:
                client.generation_options = {**getattr(client, 'generation_options', {}), 'num_ctx': request.context_window}
        return generator, judge

    def readiness(self, catalog):
        registry = PromptRegistry(self.settings.registry)
        current = registry.active_prompt
        _, dataset = self.catalog(catalog)
        tasks = select_benchmark_tasks(dataset, splits=['validation'])
        evidence = dict(training_runs=0, failure_counts={}, pending_attempt_id=None, has_previous_attempt=False)
        scope_error, scopes, last_proposal = None, [], None
        if self.settings.history_db.exists():
            with ExperimentStore(self.settings.history_db, readonly=True) as history:
                evidence = history.evolution_evidence(current.prompt_id)
                scopes = history.training_execution_scopes()
                attempts = [check for check in history.evolution_checks(current.prompt_id) if check['attempt_started']]
                if attempts:
                    last_proposal = {key: attempts[-1].get(key) for key in
                                     ('status', 'reason', 'failure_kind', 'proposal_retry_count', 'created_at')}
                try:
                    validate_evolution_scope(tasks, store=history)
                except ValueError as exc:
                    scope_error = str(exc)
        else:
            try:
                validate_evolution_scope(tasks)
            except ValueError as exc:
                scope_error = str(exc)
        eligible, reason = evolution_readiness(current, evidence)
        return dict(active_prompt_id=current.prompt_id, eligible=eligible and scope_error is None,
                    reason=scope_error or reason, training_runs=evidence['training_runs'],
                    failure_counts=evidence['failure_counts'], qualifying_categories=evidence['qualifying_categories'],
                    policy=evidence['policy'], training_scopes=scopes, catalog=catalog, last_proposal=last_proposal)

    def solve_task(self, request):
        if request.task_id is not None:
            _, dataset = self.catalog(request.catalog)
            try:
                task = select_task(dataset, request.task_id)
            except ValueError as exc:
                raise KeyError(f"unknown task: {request.task_id}") from exc
            for name in ("language", "execution_mode", "entry_point"):
                supplied = getattr(request, name)
                if supplied is not None and supplied != getattr(task, name):
                    raise ValueError(f"{name} conflicts with the catalog task")
            resolved = dict(problem_statement=task.problem_statement, trusted_test_suite=task.visible_tests,
                            task_id=task.task_id, benchmark_split=task.split, language=task.language,
                            execution_mode=task.execution_mode, entry_point=task.entry_point)
        else:
            from src.benchmark.execution_mode import suggested_execution_mode
            mode = request.execution_mode or suggested_execution_mode(request.problem, request.language or 'python')
            suite = (build_test_suite([t.model_dump() for t in request.tests], allow_empty_input=True,
                                      execution_mode=mode) if request.tests is not None else None)
            resolved = dict(problem_statement=request.problem, trusted_test_suite=suite, task_id=None,
                            benchmark_split="adhoc", language=request.language or "python",
                            execution_mode=mode, entry_point=request.entry_point)
        validate_experiment(request.config, request.seed, resolved["language"])
        validate_generation_execution(resolved["language"], resolved["execution_mode"], resolved["entry_point"])
        validate_generation_tests(resolved["trusted_test_suite"], resolved["execution_mode"])
        if request.reference_tests is not None:
            resolved['reference_test_suite'] = reference_cases(None, build_test_suite(
                [case.model_dump() for case in request.reference_tests], allow_empty_input=True))
        try:
            preflight_execution(resolved["language"], resolved["execution_mode"])
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc
        return resolved

    def prepare(self, kind, request):
        self.clients(request)  # Validate model families without making model calls.
        registry = PromptRegistry(self.settings.registry)
        catalog_hash = None
        if kind == "solve":
            self.solve_task(request)
            if request.prompt_id is not None:
                registry.get(request.prompt_id)
            if request.task_id is not None:
                catalog_hash, _ = self.catalog(request.catalog)
        else:
            catalog_hash, dataset = self.catalog(request.catalog)
            tasks = select_benchmark_tasks(dataset, splits=["validation"])
            validate_evolution_scope(tasks)
            try:
                for task in tasks:
                    preflight_execution(task.language, task.execution_mode)
            except RuntimeError as exc:
                raise ValueError(str(exc)) from exc
            request = request.model_copy(update={"run_id": request.run_id or str(uuid4())})
        payload = {"request": request.model_dump(exclude_unset=True), "context": self.context,
                   "catalog_hash": catalog_hash}
        if kind == "evolution":
            payload["run_id"] = request.run_id
        return payload

    def execute(self, kind, payload, emit):
        if payload["context"] != self.context:
            raise ValueError("server data paths changed since this job was queued; submit a new job")
        if kind not in {"solve", "evolution"}:
            raise ValueError("unknown job kind")
        request = (SolveRequest if kind == "solve" else EvolutionRequest).model_validate(payload["request"])
        if payload["catalog_hash"] is not None:
            digest, _ = self.catalog(request.catalog)
            if digest != payload["catalog_hash"]:
                raise ValueError("catalog changed since this job was queued; submit a new job")
        client, judge = self.clients(request)
        registry = PromptRegistry(self.settings.registry)
        cache = TaskSpecCache(self.settings.cache_dir)
        with ExperimentStore(self.settings.history_db) as history:
            if kind == "evolution":
                _, dataset = self.catalog(request.catalog)
                meta = self.meta_factory(request.meta_model) if request.meta_model else client
                meta.generation_options = {**getattr(meta, 'generation_options', {}), 'num_ctx': request.context_window}
                return run_evolution_round(
                    client, dataset, registry, history,
                    meta_client=meta,
                    judge_client=judge, seed=request.seed, max_attempts=request.max_attempts,
                    round_id=request.run_id, resume=request.resume, max_observations=request.max_observations,
                    proposal_attempts=request.proposal_attempts, task_spec_cache=cache,
                    on_progress=lambda message: emit("progress", {"message": message}),
                    retry_proposal=request.retry_proposal,
                )
            resolved = self.solve_task(request)
            test_client = None
            if resolved['trusted_test_suite'] is None and request.test_model and request.test_model != request.model:
                test_client = self.test_client_factory(request.test_model)
                test_client.generation_options = {
                    **getattr(test_client, 'generation_options', {}), 'num_ctx': request.context_window,
                }
            prompt_id = request.prompt_id or registry.active_prompt.prompt_id
            emit("stage", {"stage": "preparing", "prompt_id": prompt_id,
                           "language": resolved["language"], "task_id": resolved["task_id"]})
            def attempt_done(attempt):
                emit("attempt", jsonable_encoder({
                    "attempt_number": attempt.attempt_number, "code": attempt.candidate.raw_code,
                    "reward": attempt.reward, "evaluation": asdict(attempt.evaluation_report),
                    "duplicate_of": attempt.duplicate_of, "reused_from": attempt.reused_from,
                }))
            result = run_candidate_refinement_loop(
                client, **resolved, prompt_registry=registry, prompt_id=prompt_id, judge_client=judge,
                history_store=history, task_spec_cache=cache, config_name=request.config,
                seed=request.seed, max_attempts=request.max_attempts, on_attempt=attempt_done,
                reasoning_mode=request.reasoning_mode,
                judge_only_passing=request.judge_scope == 'passing',
                test_client=test_client,
                test_generation_method=request.test_generation_method,
                differential_cases=request.differential_cases,
                on_stage=lambda stage, data: emit("stage", {"stage": stage, **data}),
            )
            return jsonable_encoder({
                "history_run_id": result.history_run_id, "prompt_id": prompt_id,
                "request_log_id": result.request_log_id,
                "reasoning_mode": result.reasoning_mode,
                "test_model": result.test_model, "test_model_calls": result.test_model_calls,
                "input_format": result.task_spec.input_format,
                "output_format": result.task_spec.output_format,
                "status": result.best_attempt.evaluation_report.status if result.best_attempt else 'unverified',
                "code": result.best_code, "best_attempt_number": result.best_attempt_number,
                "attempt_count": len(result.attempts), "attempt_budget": result.attempt_budget,
                "attempt_budget_source": result.attempt_budget_source,
                "stop_reason": result.stop_reason, "test_source": result.test_source,
                "model_calls": result.model_calls, "wall_clock_seconds": result.wall_clock_seconds,
                "evaluation": asdict(result.best_attempt.evaluation_report) if result.best_attempt else None,
                "validation_error": result.validation_error,
                "test_validation": result.test_suite.validation,
                "stage_metrics": result.stage_metrics,
            })
