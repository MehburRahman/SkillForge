# Source navigation index

This index lists source-level classes, functions, and methods, including private helpers. Read the [guided explanation](../README.md) first; use this index to locate a definition afterward. Line numbers describe the source inspected for this guide and may change after edits.

Descriptions below are taken from source docstrings when available. A signature describes inputs and return annotations; it does not guarantee runtime type validation. Nested implementation functions are listed under their enclosing name.

## src/__init__.py

[Open this file](../../../src/__init__.py)

Package marker or import/export definitions; no functions or classes are defined here.

## src/agents/__init__.py

[Open this file](../../../src/agents/__init__.py)

Package marker or import/export definitions; no functions or classes are defined here.

## src/agents/code_generator.py

[Open this file](../../../src/agents/code_generator.py)

- `class CodeCandidate` — line 8. Fields: `raw_code`, `problem_type`, `prompt_version`, `model_name`, `generation_temperature`.

- `generate_code(client, task_spec, temperature=0.2, prompt_registry=None, prompt_id=None)` — line 16. Generate code with the active or explicitly selected prompt version.

## src/agents/llm_client.py

[Open this file](../../../src/agents/llm_client.py)

- `class LLMClient` — line 18.

- `LLMClient.__init__(self, model=DEFAULT_MODEL)` — line 19.

- `LLMClient.generate(self, prompt, system=None, temperature=0.2, *, seed=None)` — line 22.

- `LLMClient.generate_json(self, prompt, system=None, temperature=0.0, schema=None, *, seed=None)` — line 30. Generate JSON using Ollama's native JSON/JSON-schema mode.

- `LLMClient._generate(self, prompt, system=None, temperature=0.2, response_format=None, seed=None)` — line 42.

- `LLMClient._fallback_response(self, prompt, system=None)` — line 74. Raises a RuntimeError when Ollama is unavailable or generation fails; it does not return a fabricated response.

- `validate_judge_pair(generator, judge)` — line 89. Reject self-judging in the solve pipeline before spending compute.

- `validate_judge_pair.family(client)` — line 100.

- `build_clients(model=None, judge_model=None, llm_judge=False)` — line 113. Return (generator_client, judge_client) for one run.

- `class CallCountingClient` — line 129. Count calls and optionally inject seeds at the model-client boundary.

- `CallCountingClient.__init__(self, inner, *, seed=None, task_key='')` — line 148.

- `CallCountingClient.set_stage(self, stage)` — line 158. Start an independent seed sequence so retries cannot shift later stages.

- `CallCountingClient.__getattr__(self, name)` — line 163.

- `CallCountingClient._counted(self, method)` — line 169.

- `CallCountingClient._counted.call(*args, **kwargs)` — line 170.

## src/agents/metaprompt_agent.py

[Open this file](../../../src/agents/metaprompt_agent.py)

- `_required_string(data: dict, field_name: str) -> str` — line 48.

- `_string_list(data: dict, field_name: str) -> list[str]` — line 55.

- `_normalized_prompt(prompt: str) -> str` — line 71.

- `_validate_prompt_contract(prompt: str) -> None` — line 75.

- `_parse_proposal(raw, current_prompt: str, forbidden_task_ids: Iterable[str]) -> PromptProposal` — line 90.

- `propose_prompt(client, current_prompt_id: str, current_prompt: str, training_context: dict, forbidden_task_ids: Iterable[str], max_attempts: int=2, temperature: float=0.2) -> PromptProposal` — line 151. Propose one generalized coding prompt from training-only aggregates.

## src/agents/refinement_agent.py

[Open this file](../../../src/agents/refinement_agent.py)

- `_task_context(task_spec, previous_candidate)` — line 7. Keep the task information identical across both repair arms.

- `refine_code(client, task_spec, previous_candidate: CodeCandidate, verification_result, temperature: float=0.2, evaluation_report=None, feedback_bundle=None, feedback_mode='full') -> CodeCandidate` — line 33. Generate a repaired candidate from bounded multi-grader feedback.

- `_generate_repair(client, previous_candidate, prompt, system, temperature)` — line 148.

## src/agents/task_spec.py

[Open this file](../../../src/agents/task_spec.py)

- `class TaskSpec` — line 40. Fields: `problem_statement`, `problem_type`, `constraints`, `examples`, `input_format`, `output_format`, `expected_complexity`, `prohibited_operations`, `required_operations`, `edge_cases`, `discarded_inferred_operations`, `selected_max_attempts`, `attempt_budget_reason`.

- `class UngroundedOperationError` — line 56. Raised when a claimed operation is absent from the source problem.

- `_optional_string(data: dict, key: str, default: str='') -> str` — line 60. Read an optional JSON string while rejecting ambiguous field types.

- `_required_positive_integer(data: dict, key: str) -> int` — line 70.

- `_required_non_empty_string(data: dict, key: str) -> str` — line 77.

- `_optional_string_list(data: dict, key: str) -> List[str]` — line 84. Read an optional list of non-empty strings from the model response.

- `_optional_examples(data: dict) -> List[dict]` — line 102. Read optional input/output examples without inventing missing values.

- `_words(value: str) -> List[str]` — line 122.

- `_operation_word_matches(operation_word: str, problem_word: str) -> bool` — line 126.

- `_operation_is_grounded(operation: str, problem_statement: str) -> bool` — line 138. Require operation terms to be traceable to the source problem.

- `_grounded_operation_list(data: dict, key: str, problem_statement: str, discard_ungrounded: bool=False)` — line 155.

- `_parse_task_response(raw, problem_statement: str, discard_ungrounded_operations: bool=False) -> TaskSpec` — line 181. Validate one model response and construct its task specification.

- `parse_task(client, problem_statement: str, max_generation_attempts: int=3) -> TaskSpec` — line 233. Generate a validated TaskSpec with bounded correction retries.

## src/agents/task_spec_cache.py

[Open this file](../../../src/agents/task_spec_cache.py)

- `analyzer_fingerprint()` — line 20.

- `task_spec_from_dict(data, problem_statement)` — line 25. Revalidate disk/checkpoint data with the same grounding rules as fresh analysis.

- `class TaskSpecCache` — line 46.

- `TaskSpecCache.__init__(self, directory=DEFAULT_TASK_SPEC_CACHE)` — line 47.

- `TaskSpecCache.identity(self, problem_statement, model, language)` — line 50.

- `TaskSpecCache.get_or_analyze(self, client, problem_statement, language='python')` — line 60. Return (spec, cache_hit, actual_analysis_calls). Uses a fixed analysis seed.

- `add_cache_arguments(parser, default=None)` — line 93.

## src/agents/test_generator.py

[Open this file](../../../src/agents/test_generator.py)

- `class TestCase` — line 28. Fields: `input`, `expected_output`, `category`.

- `class TestSuite` — line 35. Fields: `cases`.

- `_problem_allows_empty_input(task_spec) -> bool` — line 39.

- `build_test_suite(data, *, allow_empty_input: bool=False) -> TestSuite` — line 53. Validate decoded test data before it can influence a reward.

- `load_test_suite(path) -> TestSuite` — line 100. Load a manually verified test suite from a JSON file.

- `generate_tests(client, task_spec, max_generation_attempts: int=3) -> TestSuite` — line 110. Generate validated tests, retrying twice after malformed responses.

## src/benchmark/__init__.py

[Open this file](../../../src/benchmark/__init__.py)

Package marker or import/export definitions; no functions or classes are defined here.

## src/benchmark/loader.py

[Open this file](../../../src/benchmark/loader.py)

- `class BenchmarkFormatError` — line 33. Raised when a trusted benchmark file has an invalid structure.

- `_required_string(record: dict, field_name: str, task_number: int) -> str` — line 37.

- `_test_suite(record: dict, field_name: str, task_id: str)` — line 46.

- `load_benchmark(path) -> BenchmarkDataset` — line 59. Load the shared problem catalog for single-task or benchmark execution.

- `select_task(dataset: BenchmarkDataset, task_id: str) -> BenchmarkTask` — line 137. Look up one problem without separating its statement from its tests.

## src/benchmark/models.py

[Open this file](../../../src/benchmark/models.py)

- `class BenchmarkTask` — line 8. Fields: `task_id`, `title`, `problem_statement`, `category`, `difficulty`, `split`, `visible_tests`, `hidden_tests`.

- `class BenchmarkDataset` — line 20. Fields: `tasks`, `source_path`.

- `class TaskBenchmarkResult` — line 26. Fields: `task_id`, `title`, `category`, `difficulty`, `split`, `prompt_version`, `history_run_id`, `attempt_budget`, `attempt_budget_source`, `attempt_budget_reason`, `attempt_count`, `repair_attempts`, `first_attempt_score`, `final_visible_score`, `hidden_score`, `final_visible_status`, `hidden_status`, `stop_reason`, `config_name`, `seed`, `language`, `model_calls`, `wall_clock_seconds`, `task_spec_cache_hit`, `analysis_model_calls`, `analysis_wall_clock_seconds`.

- `class BenchmarkSummary` — line 56. Fields: `prompt_version`, `task_results`, `run_id`, `resumed_tasks`, `interrupted_model_calls`, `interrupted_wall_clock_seconds`.

- `BenchmarkSummary.task_count(self) -> int` — line 65.

- `BenchmarkSummary._mean(values) -> float` — line 69.

- `BenchmarkSummary.first_attempt_pass_rate(self) -> float` — line 74.

- `BenchmarkSummary.final_visible_pass_rate(self) -> float` — line 80.

- `BenchmarkSummary.hidden_pass_rate(self) -> float` — line 86.

- `BenchmarkSummary.mean_first_attempt_score(self) -> float` — line 92.

- `BenchmarkSummary.mean_final_visible_score(self) -> float` — line 98.

- `BenchmarkSummary.mean_hidden_score(self) -> float` — line 104.

- `BenchmarkSummary.average_attempt_count(self) -> float` — line 108.

- `BenchmarkSummary.average_attempt_budget(self) -> float` — line 112.

- `BenchmarkSummary.to_dict(self) -> dict` — line 115.

## src/benchmark/runner.py

[Open this file](../../../src/benchmark/runner.py)

- `_selection(values: Optional[Iterable[str]])` — line 20.

- `run_benchmark(client, dataset, splits: Optional[Iterable[str]]=None, categories: Optional[Iterable[str]]=None, prompt_registry=None, prompt_id: Optional[str]=None, max_attempts: Optional[int]=None, max_feedback_failures: int=3, max_consecutive_repeats: int=2, judge_client=None, judge_max_attempts: int=3, judge_review_threshold: float=0.5, history_store=None, config_name: Optional[str]=None, seed: Optional[int]=None, language: Optional[str]=None, on_task_complete: Optional[Callable[[TaskBenchmarkResult], None]]=None, run_id: Optional[str]=None, resume: bool=False, task_spec_cache=None, on_sweep_start=None) -> BenchmarkSummary` — line 24. Persist task results and resume a fixed experiment from task boundaries.

- `_fingerprint(value)` — line 202.

- `_visible_checkpoint(task, result)` — line 206. Only the fixed selected candidate crosses into the hidden phase.

## src/evaluation/__init__.py

[Open this file](../../../src/evaluation/__init__.py)

Package marker or import/export definitions; no functions or classes are defined here.

## src/evaluation/aggregator.py

[Open this file](../../../src/evaluation/aggregator.py)

- `_functional_failures(functional_grade) -> List[EvaluationReason]` — line 19.

- `_runtime_failures(runtime_grade) -> List[EvaluationReason]` — line 42.

- `_constraint_failures(constraint_grade) -> List[EvaluationReason]` — line 73.

- `_review_reasons(constraint_grade, judge_grade, judge_review_threshold: float, judge_error=None) -> List[EvaluationReason]` — line 93.

- `build_evaluation_report(functional_grade, runtime_grade, constraint_grade, judge_grade=None, judge_review_threshold: float=0.5, judge_error=None) -> EvaluationReport` — line 185. Combine grader results without averaging away deterministic failures.

## src/evaluation/constraint_grader.py

[Open this file](../../../src/evaluation/constraint_grader.py)

- `_violation(constraint: str, operation: str, node, message: str) -> dict` — line 11.

- `_find_sort_method(tree: ast.AST, constraint: str) -> List[dict]` — line 20.

- `_find_sorted_builtin(tree: ast.AST, constraint: str) -> List[dict]` — line 35.

- `_find_builtin_sorting(tree: ast.AST, constraint: str) -> List[dict]` — line 50.

- `_find_dictionary_use(tree: ast.AST, constraint: str) -> List[dict]` — line 54.

- `_library_target(constraint: str) -> Optional[str]` — line 86.

- `_find_library_import(tree: ast.AST, constraint: str, target: str) -> List[dict]` — line 105.

- `_prohibited_finder(constraint: str) -> Optional[ViolationFinder]` — line 131.

- `_direct_recursion_sites(tree: ast.AST) -> List[ast.AST]` — line 165.

- `_required_recursion_violations(tree: ast.AST, constraint: str) -> List[dict]` — line 186.

- `_required_finder(constraint: str) -> Optional[ViolationFinder]` — line 197.

- `grade_constraint_adherence(code: str, task_spec) -> ConstraintGrade` — line 204. Check only constraints that can be established conservatively via AST.

## src/evaluation/functional_grader.py

[Open this file](../../../src/evaluation/functional_grader.py)

- `grade_functional_correctness(verification_result) -> FunctionalGrade` — line 6. Convert trusted-test verification into a structured functional grade.

## src/evaluation/judge_sanity.py

[Open this file](../../../src/evaluation/judge_sanity.py)

- `class SanityCase` — line 25. Fields: `case_id`, `task_spec`, `tests`, `correct`, `broken`, `defect_explanation`, `expected_failure_codes`.

- `load_sanity_cases(path=DEFAULT_SANITY_CASES)` — line 35. Reject malformed fixtures before executing code or making model requests.

- `verify_sanity_cases(cases)` — line 83. Check every fixture first, reusing these executions during judging.

- `class _AuditedJudge` — line 106. Observe responses with the existing validator, without changing retries.

- `_AuditedJudge.__init__(self, inner, **kwargs)` — line 109.

- `_AuditedJudge._counted(self, method)` — line 114.

- `_AuditedJudge._counted.call(*args, **kwargs)` — line 117.

- `run_judge_sanity(client, history_store, *, cases=None, seed=42, max_attempts=3, on_result=None, on_start=None)` — line 132.

- `summarize_sanity_results(results, case_ids=())` — line 192. Missing/invalid judgments are unavailable, never zero scores or correct calls.

- `load_sanity_report(store, run_id)` — line 236.

## src/evaluation/llm_judge.py

[Open this file](../../../src/evaluation/llm_judge.py)

- `_bounded(value: Any) -> Any` — line 87.

- `_task_payload(task_spec) -> Dict[str, Any]` — line 99.

- `_grade_payload(grade) -> Dict[str, Any]` — line 103.

- `_judge_payload(task_spec, code: str, functional_grade, runtime_grade, constraint_grade) -> str` — line 107.

- `_score(data: dict, field_name: str) -> float` — line 129.

- `_string_list(data: dict, field_name: str) -> list[str]` — line 139.

- `_parse_judge_grade(raw: str) -> JudgeGrade` — line 153.

- `judge_code(client, task_spec, code: str, functional_grade, runtime_grade, constraint_grade, max_attempts: int=3, temperature: float=0.0) -> JudgeGrade` — line 196. Request an advisory semantic grade with bounded schema retries.

## src/evaluation/models.py

[Open this file](../../../src/evaluation/models.py)

- `class FunctionalGrade` — line 7. Fields: `score`, `passed`, `total`, `passed_all`, `failed_categories`, `failure_evidence`.

- `class RuntimeGrade` — line 17. Fields: `score`, `total_executions`, `successful_executions`, `timeout_count`, `runtime_error_count`, `nonzero_exit_count`, `total_duration_seconds`, `max_duration_seconds`, `critical_failure`, `failure_evidence`.

- `class ConstraintGrade` — line 31. Fields: `score`, `coverage`, `checked_constraints`, `passed_constraints`, `violations`, `unverified_constraints`, `critical_violation`, `analysis_error`.

- `class JudgeGrade` — line 43. Fields: `problem_understanding`, `algorithm_suitability`, `edge_case_handling`, `requirement_adherence`, `code_quality`, `overall_score`, `critical_issues`, `feedback`, `likely_failure_category`.

- `class EvaluationStatus` — line 55.

- `class EvaluationReason` — line 62. Fields: `code`, `message`, `source`, `details`.

- `class EvaluationReport` — line 70. Fields: `functional_grade`, `runtime_grade`, `constraint_grade`, `judge_grade`, `status`, `hard_gate_passed`, `failure_reasons`, `review_reasons`, `secondary_score`, `judge_error`.

- `EvaluationReport.passed(self) -> bool` — line 83.

## src/evaluation/pipeline.py

[Open this file](../../../src/evaluation/pipeline.py)

- `evaluate_candidate(task_spec, code: str, verification_result, test_source: str, judge_client=None, judge_max_attempts: int=3, judge_review_threshold: float=0.5) -> Tuple[EvaluationReport, FeedbackBundle]` — line 16. Build all grades from one existing candidate execution trace.

## src/evaluation/runtime_grader.py

[Open this file](../../../src/evaluation/runtime_grader.py)

- `grade_runtime_robustness(verification_result) -> RuntimeGrade` — line 5. Grade crashes and timeouts using already-collected execution evidence.

## src/evaluation/taxonomy.py

[Open this file](../../../src/evaluation/taxonomy.py)

- `class FailureCategory` — line 4.

## src/evolution/__init__.py

[Open this file](../../../src/evolution/__init__.py)

- `__getattr__(name)` — line 22. Load the evolver lazily so proposal models remain cycle-free imports.

## src/evolution/models.py

[Open this file](../../../src/evolution/models.py)

- `class PromptProposal` — line 8. Fields: `proposed_prompt`, `change_summary`, `targeted_failure_categories`, `evidence_used`, `expected_benefits`, `regression_risks`, `preserved_behaviors`.

- `class PromptCandidateResult` — line 19. Fields: `proposal`, `prompt_version`.

## src/evolution/prompt_evolver.py

[Open this file](../../../src/evolution/prompt_evolver.py)

- `propose_candidate_prompt(meta_client, prompt_registry, history_store, prompt_id=None, max_observations: int=50, max_attempts: int=2, temperature: float=0.2) -> PromptCandidateResult` — line 13. Create and register one candidate from training-only history.

## src/evolution/prompt_registry.py

[Open this file](../../../src/evolution/prompt_registry.py)

- `class PromptRegistryError` — line 17. Raised when prompt registry data or an operation is invalid.

- `class PromptVersion` — line 22. Fields: `prompt_id`, `prompt_text`, `parent_prompt_id`, `created_at`, `change_reason`, `status`, `training_metrics`, `validation_metrics`.

- `_metric_map(value, field_name: str) -> Dict[str, float]` — line 33.

- `_required_string(record: dict, field_name: str) -> str` — line 51.

- `_prompt_from_record(record, index: int) -> PromptVersion` — line 58.

- `_validate_versions(versions: Iterable[PromptVersion]) -> Tuple[PromptVersion, ...]` — line 103.

- `_utc_timestamp() -> str` — line 132.

- `class PromptRegistry` — line 136. Persistent version store for code-generator system prompts.

- `PromptRegistry.__init__(self, path)` — line 139.

- `PromptRegistry._read(self) -> Tuple[PromptVersion, ...]` — line 143.

- `PromptRegistry._write(self, versions: Iterable[PromptVersion]) -> None` — line 171.

- `PromptRegistry.versions(self) -> Tuple[PromptVersion, ...]` — line 208.

- `PromptRegistry.active_prompt(self) -> PromptVersion` — line 212.

- `PromptRegistry.get(self, prompt_id: str) -> PromptVersion` — line 217.

- `PromptRegistry._next_prompt_id(self) -> str` — line 223.

- `PromptRegistry.register_candidate(self, prompt_text: str, change_reason: str, parent_prompt_id: Optional[str]=None, training_metrics: Optional[Dict[str, float]]=None, validation_metrics: Optional[Dict[str, float]]=None, created_at: Optional[str]=None) -> PromptVersion` — line 230. Persist a candidate without changing the current active prompt.

- `PromptRegistry.activate(self, prompt_id: str) -> PromptVersion` — line 262. Manually activate a candidate or archived prompt version.

- `PromptRegistry.reject(self, prompt_id: str) -> PromptVersion` — line 281. Mark a non-active prompt as rejected.

- `load_default_registry() -> PromptRegistry` — line 299.

## src/experiments.py

[Open this file](../../../src/experiments.py)

- `validate_seed(seed)` — line 13.

- `validate_experiment(config_name=None, seed=None, language=None)` — line 23. Reject unsupported work before any model calls or history writes.

- `request_seed(base_seed, task_key, stage, call_index)` — line 42. Stable across processes, task ordering, configurations and prompt versions.

- `add_experiment_arguments(parser)` — line 52.

## src/feedback/__init__.py

[Open this file](../../../src/feedback/__init__.py)

Package marker or import/export definitions; no functions or classes are defined here.

## src/feedback/collector.py

[Open this file](../../../src/feedback/collector.py)

- `_bounded_text(value) -> str` — line 12.

- `_functional_category(functional_grade) -> FailureCategory` — line 19.

- `_functional_signal(report) -> Optional[FeedbackSignal]` — line 27.

- `_runtime_signals(report) -> List[FeedbackSignal]` — line 51.

- `_constraint_signals(report) -> List[FeedbackSignal]` — line 76.

- `_judge_signal(report, test_source: str) -> Optional[FeedbackSignal]` — line 109.

- `_candidate_feedback(report, signals, max_evidence_items: int) -> str` — line 152.

- `_primary_category(signals: List[FeedbackSignal])` — line 188.

- `_unique_categories(signals: List[FeedbackSignal]) -> List[FailureCategory]` — line 200.

- `_evolution_observations(signals: List[FeedbackSignal]) -> List[str]` — line 208.

- `collect_feedback(report, test_source: str, stop_reason: Optional[str]=None, max_evidence_items: int=3) -> FeedbackBundle` — line 223. Convert one evaluation report into repair and evolution feedback.

## src/feedback/models.py

[Open this file](../../../src/feedback/models.py)

- `class SignalStrength` — line 9.

- `class FeedbackSignal` — line 16. Fields: `category`, `strength`, `source`, `message`, `evidence`.

- `class FeedbackBundle` — line 25. Fields: `evaluation_status`, `primary_category`, `categories`, `candidate_feedback`, `evolution_observations`, `signals`.

## src/history/__init__.py

[Open this file](../../../src/history/__init__.py)

Package marker or import/export definitions; no functions or classes are defined here.

## src/history/experiment_store.py

[Open this file](../../../src/history/experiment_store.py)

- `class ExperimentStoreError` — line 45. Raised when history data or an experiment-store operation is invalid.

- `_timestamp() -> str` — line 49.

- `_json_default(value)` — line 53.

- `_json_dump(value) -> str` — line 61.

- `_candidate_fingerprint(code: str) -> str` — line 70.

- `_default_task_id(problem_statement: str) -> str` — line 77.

- `class ExperimentStore` — line 82. SQLite-backed storage for runs, attempts, reports, and failure signals.

- `ExperimentStore.__init__(self, path=DEFAULT_HISTORY_PATH)` — line 85.

- `ExperimentStore.__enter__(self)` — line 105.

- `ExperimentStore.__exit__(self, exc_type, exc_value, traceback)` — line 108.

- `ExperimentStore.close(self) -> None` — line 111.

- `ExperimentStore._initialize_schema(self) -> None` — line 116.

- `ExperimentStore._is_supported_version(value) -> bool` — line 270. Accept any past version this build can still migrate forward.

- `ExperimentStore._ensure_runs_columns(self) -> None` — line 278. Add any RUNS_ADDED_COLUMNS the database does not have yet.

- `ExperimentStore.start_judge_sanity(self, judge_model, seed, fixture_hash, config)` — line 295.

- `ExperimentStore.record_judge_sanity_result(self, run_id, result)` — line 306. Commit each candidate independently so interruptions keep prior results.

- `ExperimentStore.complete_judge_sanity(self, run_id, wall_clock_seconds)` — line 324.

- `ExperimentStore._get_judge_sanity_run(self, run_id)` — line 339.

- `ExperimentStore.get_judge_sanity(self, run_id)` — line 349.

- `ExperimentStore.start_run(self, problem_statement: str, prompt_version: str, model_name: str, test_source: str, judge_enabled: bool, task_id: Optional[str]=None, judge_model: Optional[str]=None, config_name: Optional[str]=None, seed: Optional[int]=None, language: Optional[str]=None, benchmark_split: str='adhoc', run_id: Optional[str]=None, started_at: Optional[str]=None, attempt_budget: Optional[int]=None, attempt_budget_source: Optional[str]=None, attempt_budget_reason: Optional[str]=None, benchmark_run_id: Optional[str]=None) -> str` — line 360.

- `ExperimentStore.record_attempt(self, run_id: str, attempt, created_at: Optional[str]=None) -> None` — line 477.

- `ExperimentStore.complete_run(self, run_id: str, result, completed_at: Optional[str]=None) -> None` — line 574.

- `ExperimentStore.begin_benchmark(self, manifest, run_id=None, resume=False)` — line 612. Pin settings once; an existing sweep can only be opened explicitly.

- `ExperimentStore.benchmark_task(self, run_id, task_id)` — line 641.

- `ExperimentStore.checkpoint_benchmark_task(self, run_id, task_id, checkpoint)` — line 650. Freeze visible selection before any hidden execution.

- `ExperimentStore.complete_benchmark_task(self, run_id, result, best_attempt_number)` — line 661. Commit hidden results and the task-run completion atomically.

- `ExperimentStore.interrupt_benchmark_task(self, run_id, task_id, model_calls, seconds, error)` — line 684.

- `ExperimentStore.finish_benchmark(self, run_id)` — line 695.

- `ExperimentStore.record_prompt_experiment(self, baseline_prompt_id: str, candidate_prompt_id: str, benchmark_split: str, task_count: int, baseline_pass_rate: float, candidate_pass_rate: float, candidate_only_passes: int, baseline_only_passes: int, p_value: float, decision: str, reason: str, experiment_id: Optional[str]=None, created_at: Optional[str]=None) -> str` — line 711. Record one baseline-versus-candidate prompt comparison.

- `ExperimentStore.prompt_experiments(self, candidate_prompt_id: Optional[str]=None, limit: int=50)` — line 810. Return recorded prompt comparisons, newest first.

- `ExperimentStore.get_run(self, run_id: str)` — line 828.

- `ExperimentStore.get_attempts(self, run_id: str)` — line 834.

- `ExperimentStore.get_signals(self, run_id: str)` — line 844.

- `ExperimentStore._split_clause(splits: Iterable[str]) -> Tuple[Tuple[str, ...], str]` — line 855.

- `ExperimentStore.failure_counts(self, prompt_version: Optional[str]=None, splits: Iterable[str]=('train',))` — line 867. Count affected runs by category using eligible splits only.

- `ExperimentStore.training_observations(self, prompt_version: Optional[str]=None, limit: int=100)` — line 891. Return only training-split signals for future prompt reflection.

- `ExperimentStore.training_run_summary(self, prompt_version: Optional[str]=None)` — line 919. Return aggregate outcomes without exposing task content or test data.

## src/reinforcement_loop.py

[Open this file](../../../src/reinforcement_loop.py)

- `class AttemptResult` — line 19. Fields: `attempt_number`, `candidate`, `verification`, `reward`, `evaluation_report`, `feedback`, `attempt_budget`, `attempt_budget_source`, `attempt_budget_reason`.

- `class CandidateRefinementResult` — line 32. Complete result of the bounded, per-task candidate repair loop. Fields: `task_spec`, `test_suite`, `attempts`, `success`, `best_attempt_number`, `stop_reason`, `test_source`, `history_run_id`, `attempt_budget`, `attempt_budget_source`, `attempt_budget_reason`, `model_calls`, `wall_clock_seconds`, `config_name`, `seed`, `language`, `task_spec_cache_hit`, `analysis_model_calls`, `analysis_wall_clock_seconds`.

- `CandidateRefinementResult.best_attempt(self) -> Optional[AttemptResult]` — line 59.

- `CandidateRefinementResult.best_code(self) -> str` — line 65.

- `CandidateRefinementResult.best_reward(self) -> float` — line 70.

- `_candidate_fingerprint(code: str) -> str` — line 75. Normalize irrelevant trailing whitespace when detecting repeated code.

- `_attempt_rank(attempt: AttemptResult)` — line 80. Rank hard status before deterministic and advisory secondary scores.

- `_review_has_actionable_feedback(report: EvaluationReport) -> bool` — line 108. Only judge-backed review concerns are actionable without human input.

- `_resolve_attempt_budget(task_spec: TaskSpec, max_attempts: Optional[int], config_name=None)` — line 116.

- `run_candidate_refinement_loop(client, problem_statement: str, max_attempts: Optional[int]=None, max_feedback_failures: int=3, max_consecutive_repeats: int=2, on_attempt: Optional[Callable[[AttemptResult], None]]=None, trusted_test_suite: Optional[TestSuite]=None, prompt_registry=None, prompt_id: Optional[str]=None, judge_client=None, judge_max_attempts: int=3, judge_review_threshold: float=0.5, history_store=None, task_id: Optional[str]=None, benchmark_split: str='adhoc', config_name: Optional[str]=None, seed: Optional[int]=None, language: Optional[str]=None, complete_history: bool=True, task_spec_cache=None, benchmark_run_id: Optional[str]=None) -> CandidateRefinementResult` — line 140. Generate, evaluate, and repair candidates for one programming task.

- `run_reinforcement_loop(client, problem_statement: str, max_attempts: Optional[int]=None, max_feedback_failures: int=3, max_consecutive_repeats: int=2, on_attempt: Optional[Callable[[AttemptResult], None]]=None, trusted_test_suite: Optional[TestSuite]=None, prompt_registry=None, prompt_id: Optional[str]=None, judge_client=None, judge_max_attempts: int=3, judge_review_threshold: float=0.5, history_store=None, task_id: Optional[str]=None, benchmark_split: str='adhoc', config_name: Optional[str]=None, seed: Optional[int]=None, language: Optional[str]=None, task_spec_cache=None) -> CandidateRefinementResult` — line 399. Compatibility wrapper for :func:`run_candidate_refinement_loop`.

## src/utils/__init__.py

[Open this file](../../../src/utils/__init__.py)

Package marker or import/export definitions; no functions or classes are defined here.

## src/utils/parsing.py

[Open this file](../../../src/utils/parsing.py)

- `strip_markdown_fence(text: str) -> str` — line 5. Remove one optional Markdown fence while preserving response content.

- `parse_json_response(text: str) -> Any` — line 19. Parse JSON returned either directly or inside a Markdown fence.

## src/verifier/outcome_verifier.py

[Open this file](../../../src/verifier/outcome_verifier.py)

- `class TestCaseVerification` — line 9. Fields: `input`, `expected`, `actual`, `category`, `passed`, `execution`.

- `class VerificationResult` — line 19. Fields: `pass_rate`, `total`, `passed`, `first_failure`, `failures`, `case_results`.

- `VerificationResult.execution_results(self) -> List[ExecutionResult]` — line 28. Expose the exact executions shared by deterministic graders.

- `_display_output(value: Any) -> str` — line 33. Return a stable, human-readable representation of a test output.

- `_outputs_match(actual: str, expected: Any) -> bool` — line 40. Compare plain-text output, while allowing JSON arrays and objects.

- `verify(code: str, test_suite, max_failures: int=3, timeout: float=5) -> VerificationResult` — line 52. Score a candidate and retain a bounded set of repair examples.

## src/verifier/sandbox.py

[Open this file](../../../src/verifier/sandbox.py)

- `class Status` — line 11.

- `class ExecutionResult` — line 19. Fields: `status`, `stdout`, `stderr`, `exit_code`, `duration_seconds`.

- `_normalize_input(stdin_input: Any) -> str` — line 27.

- `run_code_many(code: str, stdin_inputs: List[Any], timeout: float=5) -> List[ExecutionResult]` — line 37. Write a candidate once, then execute it independently for every input.

- `run_code(code: str, stdin_input: str='', timeout: float=5) -> ExecutionResult` — line 86.

## run_benchmark.py

[Open this file](../../../run_benchmark.py)

- `print_task(result) -> None` — line 11.

- `build_parser() -> argparse.ArgumentParser` — line 20.

- `main() -> None` — line 76.

## run_judge_sanity.py

[Open this file](../../../run_judge_sanity.py)

- `build_parser()` — line 16.

- `_score(value)` — line 31.

- `_concern(value)` — line 35.

- `print_result(result)` — line 39.

- `print_report(report)` — line 47.

- `main()` — line 95.

## run_reinforcement.py

[Open this file](../../../run_reinforcement.py)

- `print_attempt(attempt: AttemptResult) -> None` — line 15.

- `print_task_spec_warnings(task_spec) -> None` — line 62.

- `build_parser() -> argparse.ArgumentParser` — line 75.

- `resolve_task(args)` — line 169. Resolve one source of task data before opening history or creating clients.

- `main() -> None` — line 189.
