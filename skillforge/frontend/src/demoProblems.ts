// Demo problems with independently verified tests. Supplying tests skips
// test generation, so a live demo runs in minutes instead of ~20 minutes.
// Answers were checked against brute-force solutions before inclusion.
export interface DemoProblem { id: string; title: string; language: string; mode?: 'stdio' | 'http'; model?: string; problem: string; tests: unknown[] }

// Generator choices. Only one model is resident at a time on a 16 GB Mac;
// the test and judge models stay separate and are swapped in when needed.
export const DEFAULT_GENERATOR = 'qwen2.5-coder:7b'
export const GENERATOR_MODELS = [
  { name: 'qwen2.5-coder:7b', note: 'Fast default. Solves most demo problems in a few minutes.' },
  { name: 'qwen2.5-coder:14b', note: 'Stronger and about 2× slower (~9 GB). Use for hard multi-requirement tasks.' },
]

export const DEMO_PROBLEMS: DemoProblem[] = []
