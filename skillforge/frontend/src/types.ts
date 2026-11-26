export type Verdict = 'pass' | 'fail' | 'review'
export type JobStatus = 'queued' | 'running' | 'completed' | 'failed' | 'interrupted'
export type ReasoningMode = 'direct' | 'plan_critique'
export interface Reason { code: string; message: string; source: string; details?: unknown }
export interface Evaluation {
  status: Verdict
  hard_gate_passed: boolean | number
  functional_grade: { score?: number; passed?: number; total?: number; failure_evidence?: unknown[] }
  runtime_grade: { score?: number; timeout_count?: number; runtime_error_count?: number; compilation_error_count?: number; compilation_seconds?: number; total_duration_seconds?: number; execution_environment?: { scala?: string; scala_cli?: string; jvm?: string; offline?: boolean } }
  constraint_grade: { score?: number | null; coverage?: number; unverified_constraints?: string[]; violations?: unknown[]; requirement_ledger?: { coverage: number; scope: string; requirements: { id: string; text: string; state: 'verified' | 'failed' | 'unverified'; evidence: {type:string;test_id:string|null;result:string;independent:boolean;detail:unknown}[] }[] } }
  evidence_confidence?: number | null
  judge_grade?: { overall_score: number; feedback: string; critical_issues?: string[] } | null
  judge_error?: string | null
  failure_reasons: Reason[]
  review_reasons: Reason[]
}
export interface SolveResult {
  history_run_id: string | null; prompt_id: string; status: Verdict | 'unverified'; code: string
  best_attempt_number: number; attempt_count: number; attempt_budget: number
  attempt_budget_source: string; stop_reason: string; test_source: string
  model_calls: number; wall_clock_seconds: number; evaluation: Evaluation | null
  validation_error?: string | null
  test_validation?: { status?: string; generated_tests_error?: string; coverage_complete?: boolean; coverage_warning?: string; reference_anchors_verified?: boolean; independent_anchor_count?: number; request_count?: number; scenario_count?: number; differential?: { seed: number; requested: number; added: number }; differential_error?: string }
  stage_metrics?: { evaluated_candidates: number; compiled_candidates: number; compile_rate: number | null; all_tests_passed_candidates: number; repaired_after_initial_failure: boolean; diagnostic_candidates?: number; diagnostic_compilation_failures?: number }
  reasoning_mode?: ReasoningMode
  test_model?: string | null; test_model_calls?: number
  input_format?: string; output_format?: string
}
export interface Job {
  job_id: string; kind: string; status: JobStatus; resource_id: string | null
  result: SolveResult | null; error: string | null
  request?: Partial<SolveInput>
  draft_code?: string | null
}
export interface Task {
  task_id: string; title: string; problem_statement: string; split: string
  language: 'python' | 'javascript' | 'scala' | 'cpp' | 'java'; execution_mode: string; entry_point: string | null
}
export interface Run {
  run_id: string; task_id: string | null; problem_statement: string; benchmark_split: string
  prompt_version: string; model_name: string; judge_model: string | null; language: string
  execution_mode?: string; entry_point?: string | null
  reasoning_mode?: ReasoningMode
  test_model?: string | null; test_model_calls?: number
  started_at: string; completed_at: string | null; final_status: Verdict | null
  best_attempt_number: number | null; attempt_budget: number; attempt_budget_source: string
  model_calls: number; wall_clock_seconds: number; test_source: string; stop_reason: string | null
}
export interface Attempt extends Omit<Evaluation, 'status'> {
  attempt_number: number; candidate_code: string; reward: number; evaluation_status: Verdict
  duplicate_of?: number | null; reused_from?: number | null; code_diff: string | null; feedback?: { candidate_feedback?: string }
}
export interface Signal {
  attempt_number: number; category: string; strength: string; source: string; message: string; evidence: unknown
}
export interface RunDetail extends Run { request_log_id?: string | null; attempts: Attempt[]; failure_signals: Signal[] }
export interface Page<T> { items: T[]; total: number }
export interface SolveInput {
  problem?: string; task_id?: string; catalog?: string; language?: string
  execution_mode?: string; entry_point?: string; tests?: unknown[]
  reference_tests?: unknown[]
  model: string; context_window?: number; judge_model?: string; max_attempts?: number; seed?: number
  test_model?: string
  differential_cases?: number
  reasoning_mode?: ReasoningMode
  config?: 'single_shot' | 'best_of_n' | 'scalar_repair' | 'full'
}
