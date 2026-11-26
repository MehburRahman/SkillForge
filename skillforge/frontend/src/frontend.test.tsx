import { act, fireEvent, render, renderHook, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi, test, expect } from 'vitest'
import { Evidence } from './components'
import { Solve } from './Solve'
import { Runs } from './Runs'
import { useJob } from './useJob'
import { catalogTasks, request } from './api'
import type { Evaluation, Job, RunDetail, SolveResult } from './types'

const evaluation: Evaluation = {
  status: 'pass', hard_gate_passed: true,
  functional_grade: { passed: 1, total: 1, score: 1 },
  runtime_grade: { runtime_error_count: 0, timeout_count: 0 },
  constraint_grade: { coverage: 1 }, judge_grade: null, failure_reasons: [], review_reasons: [],
}
const result: SolveResult = {
  history_run_id: 'r1', prompt_id: 'P0', status: 'pass', code: 'print(5)', best_attempt_number: 1,
  attempt_count: 1, attempt_budget: 3, attempt_budget_source: 'task_analysis_llm', stop_reason: 'passed',
  test_source: 'generated', model_calls: 3, wall_clock_seconds: 2, evaluation,
}
const job: Job = { job_id: 'j1', kind: 'solve', status: 'running', resource_id: null, result: null, error: null }
const unverified: SolveResult = {
  ...result, history_run_id: null, status: 'unverified', evaluation: null,
  best_attempt_number: 0, attempt_count: 0, stop_reason: 'test_validation_failed',
  validation_error: 'The reference disagrees with independently verified cases.',
  input_format: 'One value per line.', output_format: 'One result per line.',
}
const response = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } })

class FakeStream extends EventTarget {
  static instances: FakeStream[] = []
  onopen: (() => void) | null = null
  onerror: (() => void) | null = null
  closed = false
  constructor(public url: string) { super(); FakeStream.instances.push(this) }
  close() { this.closed = true }
  emit(type: string, data: unknown, id: number) { this.dispatchEvent(new MessageEvent(type, { data: JSON.stringify(data), lastEventId: String(id) })) }
}
function streams() { FakeStream.instances = []; vi.stubGlobal('EventSource', FakeStream) }

test('custom solve preserves model-selected budget and avoids accidental judge activation', async () => {
  const user = userEvent.setup()
  const onJob = vi.fn()
  const fetch = vi.fn().mockResolvedValue(response(job, 202))
  vi.stubGlobal('fetch', fetch)
  render(<Solve jobId="" onJob={onJob} onRun={vi.fn()} />)
  await user.type(screen.getByLabelText('Problem statement'), 'Read two integers and print their sum.')
  await user.click(screen.getByRole('button', { name: /Start solve/ }))
  expect(onJob).toHaveBeenCalledWith('j1')
  const body = JSON.parse(fetch.mock.calls[0][1].body)
  expect(body).toMatchObject({ problem: 'Read two integers and print their sum.', language: 'python', model: 'qwen2.5-coder:7b', test_model: 'granite4.2:8b', differential_cases: 32 })
  expect(body).not.toHaveProperty('max_attempts')
  expect(body).not.toHaveProperty('judge_model')
  expect(body).not.toHaveProperty('catalog')
  expect(body).not.toHaveProperty('config')
})

test('function submission requires assertions and invalid JSON never reaches the API', async () => {
  const user = userEvent.setup()
  const fetch = vi.fn()
  vi.stubGlobal('fetch', fetch)
  render(<Solve jobId="" onJob={vi.fn()} onRun={vi.fn()} />)
  await user.type(screen.getByLabelText('Problem statement'), 'Add two integers from solve(a, b).')
  await user.click(screen.getByText('Run settings & tests'))
  await user.selectOptions(screen.getByLabelText('Execution mode'), 'function')
  await user.click(screen.getByRole('button', { name: /Start solve/ }))
  expect(await screen.findByRole('alert')).toHaveTextContent('Function mode needs assertion tests')
  await user.type(screen.getByLabelText(/Assertion tests \(required\)/), 'invalid')
  await user.click(screen.getByRole('button', { name: /Start solve/ }))
  expect(screen.getByRole('alert')).toHaveTextContent('invalid JSON')
  expect(fetch).not.toHaveBeenCalled()
})

test('test generation model leaves the 7B candidate model unchanged', async () => {
  const user = userEvent.setup()
  const fetch = vi.fn().mockResolvedValue(response(job, 202))
  vi.stubGlobal('fetch', fetch)
  render(<Solve jobId="" onJob={vi.fn()} onRun={vi.fn()} />)
  await user.type(screen.getByLabelText('Problem statement'), 'Read integers and print their sum.')
  await user.clear(screen.getByLabelText(/Test generation model/))
  await user.type(screen.getByLabelText(/Test generation model/), 'reference-model:large')
  await user.click(screen.getByRole('button', { name: /Start solve/ }))
  expect(JSON.parse(fetch.mock.calls[0][1].body)).toMatchObject({
    model: 'qwen2.5-coder:7b', test_model: 'reference-model:large',
  })
})

test('supplied trusted tests bypass the selected test generation model', async () => {
  const user = userEvent.setup()
  const fetch = vi.fn().mockResolvedValue(response(job, 202))
  vi.stubGlobal('fetch', fetch)
  render(<Solve jobId="" onJob={vi.fn()} onRun={vi.fn()} />)
  await user.type(screen.getByLabelText('Problem statement'), 'Read two integers and print their sum.')
  await user.clear(screen.getByLabelText(/Test generation model/))
  await user.type(screen.getByLabelText(/Test generation model/), 'unused-model')
  await user.click(screen.getByText('Run settings & tests'))
  const cases = [{ input: '2 3', expected_output: '5', category: 'normal' }]
  fireEvent.change(screen.getByLabelText(/^Tests \(optional\)/), { target: { value: JSON.stringify(cases) } })
  expect(screen.getByLabelText(/Test generation model/)).toBeDisabled()
  await user.click(screen.getByRole('button', { name: /Start solve/ }))
  const body = JSON.parse(fetch.mock.calls[0][1].body)
  expect(body.tests).toEqual(cases)
  expect(body).not.toHaveProperty('test_model')
})

test('solve forwards explicit strategy and reproducible budget settings', async () => {
  const user = userEvent.setup()
  const fetch = vi.fn().mockResolvedValue(response(job, 202))
  vi.stubGlobal('fetch', fetch)
  render(<Solve jobId="" onJob={vi.fn()} onRun={vi.fn()} />)
  await user.type(screen.getByLabelText('Problem statement'), 'Sum integers from standard input.')
  await user.selectOptions(screen.getByLabelText('Language'), 'scala')
  await user.click(screen.getByText('Run settings & tests'))
  await user.selectOptions(screen.getByRole('combobox', { name: /^Experiment strategy/ }), 'full')
  await user.selectOptions(screen.getByLabelText('Random reference checks'), '128')
  await user.type(screen.getByLabelText(/^Maximum attempts/), '3')
  await user.type(screen.getByLabelText(/Seed/), '42')
  await user.click(screen.getByRole('button', { name: /Start solve/ }))
  expect(JSON.parse(fetch.mock.calls[0][1].body)).toMatchObject({
    language: 'scala', config: 'full', max_attempts: 3, seed: 42, differential_cases: 128,
  })
})

test('duplicate clicks cannot submit two jobs', async () => {
  const fetch = vi.fn(() => new Promise<Response>(() => {}))
  vi.stubGlobal('fetch', fetch)
  render(<Solve jobId="" onJob={vi.fn()} onRun={vi.fn()} />)
  fireEvent.change(screen.getByLabelText('Problem statement'), { target: { value: 'Add two integers.' } })
  const submit = screen.getByRole('button', { name: /Start solve/ })
  fireEvent.click(submit); fireEvent.click(submit)
  expect(fetch).toHaveBeenCalledTimes(1)
  expect(screen.getByRole('button', { name: 'Submitting…' })).toBeDisabled()
})

test('live events deduplicate on reconnect and stop after the final verdict', async () => {
  streams()
  vi.stubGlobal('fetch', vi.fn().mockImplementation(async () => response(job)))
  const { result: progress, unmount } = renderHook(() => useJob('j1'))
  await waitFor(() => expect(progress.current.status).toBe('running'))
  const stream = FakeStream.instances[0]
  act(() => {
    stream.emit('stage', { stage: 'evaluation', attempt_number: 1, attempt_budget: 3 }, 10)
    stream.emit('attempt', { attempt_number: 1, code: 'print(5)', reward: 1, evaluation }, 11)
    stream.emit('attempt', { attempt_number: 1, code: 'print(5)', reward: 1, evaluation }, 11)
  })
  expect(progress.current.attempts).toHaveLength(1)
  expect(progress.current.attempt).toBe(1)
  act(() => stream.emit('completed', { result, error: null }, 12))
  expect(progress.current.status).toBe('completed')
  expect(progress.current.result?.code).toBe('print(5)')
  expect(stream.closed).toBe(true)
  unmount()
})

test('opening a completed job restores its result without resubmitting', async () => {
  streams()
  const fetch = vi.fn().mockImplementation(async () => response({ ...job, status: 'completed', result, resource_id: 'r1', request: { problem: 'Saved original problem.', model: 'custom-generator', judge_model: 'independent-judge', max_attempts: 5 } }))
  vi.stubGlobal('fetch', fetch)
  render(<Solve jobId="j1" onJob={vi.fn()} onRun={vi.fn()} />)
  expect(await screen.findByText('Solve completed')).toBeInTheDocument()
  expect(screen.getByLabelText('Best solution · attempt 1')).toHaveTextContent('print(5)')
  expect(screen.getByText(/not independent verification/)).toBeInTheDocument()
  expect(screen.getByText('Saved original problem.')).toBeInTheDocument()
  expect(screen.getByText('custom-generator')).toBeInTheDocument()
  expect(screen.queryByLabelText('Generator model')).not.toBeInTheDocument()
  expect(fetch.mock.calls.every(([, options]) => !options?.method)).toBe(true)
})

test('an unverified result restores the draft and explains the failure without a verdict or history link', async () => {
  streams()
  vi.stubGlobal('fetch', vi.fn(async () => response({ ...job, status: 'completed', result: unverified })))
  render(<Solve jobId="j1" onJob={vi.fn()} onRun={vi.fn()} />)
  expect(await screen.findByText('Draft saved; verification incomplete')).toBeInTheDocument()
  expect(screen.getByLabelText('Unverified draft')).toHaveTextContent(unverified.code)
  expect(screen.getByText(unverified.validation_error!)).toBeInTheDocument()
  expect(screen.queryByText('Solve completed')).not.toBeInTheDocument()
  expect(screen.queryByLabelText('Verdict evidence')).not.toBeInTheDocument()
  expect(screen.queryByText('View saved run')).not.toBeInTheDocument()
  fireEvent.click(screen.getByText('Input and output format'))
  expect(screen.getByText(unverified.input_format!)).toBeVisible()
})

test('verifier infrastructure failure is distinct from a failed candidate', async () => {
  streams()
  vi.stubGlobal('fetch', vi.fn(async () => response({ ...job, status: 'failed',
    error: 'Infrastructure error: HTTP verifier crashed', draft_code: 'saved draft', result: null })))
  render(<Solve jobId="j1" onJob={vi.fn()} onRun={vi.fn()} />)
  expect(await screen.findByText('Verification infrastructure failed; draft not graded')).toBeInTheDocument()
  expect(screen.getByText('The verifier failed; this is not a verdict on your code.')).toBeInTheDocument()
  expect(screen.queryByLabelText('Verdict evidence')).not.toBeInTheDocument()
})

test('live test generation keeps the candidate draft visible through an unverified completion', async () => {
  streams()
  vi.stubGlobal('fetch', vi.fn(async () => response(job)))
  render(<Solve jobId="j1" onJob={vi.fn()} onRun={vi.fn()} />)
  await screen.findByText('Progress is saved while the worker runs.')
  act(() => {
    FakeStream.instances[0].emit('stage', { stage: 'draft_ready', code: unverified.code }, 1)
    FakeStream.instances[0].emit('stage', { stage: 'test_generation' }, 2)
  })
  expect(screen.getByLabelText('Unverified draft')).toHaveTextContent(unverified.code)
  expect(screen.getByText('Verification is pending. This draft has not been graded.')).toBeInTheDocument()
  act(() => FakeStream.instances[0].emit('completed', { result: unverified }, 3))
  expect(screen.getByLabelText('Unverified draft')).toHaveTextContent(unverified.code)
  expect(screen.queryByLabelText('Verdict evidence')).not.toBeInTheDocument()
  expect(FakeStream.instances[0].closed).toBe(true)
})

test('an interrupted job restores its saved draft from the snapshot before event replay', async () => {
  streams()
  vi.stubGlobal('fetch', vi.fn(async () => response({
    ...job, status: 'interrupted', draft_code: unverified.code, error: 'Server stopped during verification.',
  })))
  render(<Solve jobId="j1" onJob={vi.fn()} onRun={vi.fn()} />)
  expect(await screen.findByLabelText('Unverified draft')).toHaveTextContent(unverified.code)
  expect(screen.getByText('Verification did not finish. This saved draft has not been graded.')).toBeInTheDocument()
  expect(screen.queryByLabelText('Verdict evidence')).not.toBeInTheDocument()
  expect(FakeStream.instances[0].closed).toBe(true)
})

test('SSE disconnect falls back to saved terminal job state', async () => {
  streams()
  let finished = false
  vi.stubGlobal('fetch', vi.fn(async () => response(finished ? { ...job, status: 'failed', error: 'Model unavailable' } : job)))
  const { result: progress } = renderHook(() => useJob('j1'))
  await waitFor(() => expect(progress.current.status).toBe('running'))
  finished = true
  act(() => FakeStream.instances[0].onerror?.())
  await waitFor(() => expect(progress.current.error).toBe('Model unavailable'))
  expect(FakeStream.instances[0].closed).toBe(true)
})

test('missing jobs stop reconnecting and let the user start again', async () => {
  streams()
  vi.stubGlobal('fetch', vi.fn(async () => response({ detail: 'unknown job' }, 404)))
  render(<Solve jobId="missing" onJob={vi.fn()} onRun={vi.fn()} />)
  expect(await screen.findByRole('alert')).toHaveTextContent('job was not found')
  expect(screen.getByRole('button', { name: 'New solve' })).toBeEnabled()
  expect(FakeStream.instances[0].closed).toBe(true)
})

test('REVIEW is displayed separately from PASS even when all tests pass', () => {
  render(<Evidence evaluation={{ ...evaluation, status: 'review', review_reasons: [{ code: 'JUDGE_ERROR', source: 'judge', message: 'The judge could not assess this candidate.' }] }} />)
  expect(screen.getByText('REVIEW')).toBeInTheDocument()
  expect(screen.queryByText('PASS')).not.toBeInTheDocument()
  expect(screen.getByText('The judge could not assess this candidate.')).toBeInTheDocument()
})

const run: RunDetail = {
  run_id: 'r1', task_id: null, problem_statement: 'Sum two numbers.', benchmark_split: 'adhoc',
  prompt_version: 'P0', model_name: 'test-model', judge_model: null, language: 'python',
  started_at: '2026-11-11T00:00:00Z', completed_at: '2026-11-11T00:01:00Z', final_status: 'pass',
  best_attempt_number: 2, attempt_budget: 3, attempt_budget_source: 'task_analysis_llm',
  model_calls: 3, wall_clock_seconds: 2, test_source: 'trusted', stop_reason: 'passed',
  attempts: [
    { ...evaluation, attempt_number: 1, candidate_code: 'print(0)', evaluation_status: 'fail', reward: 0, code_diff: null },
    { ...evaluation, attempt_number: 2, candidate_code: 'print(5)', evaluation_status: 'pass', reward: 1, code_diff: '-print(0)\n+print(5)' },
  ],
  failure_signals: [{ attempt_number: 1, category: 'LOGIC_ERROR', strength: 'inferred', source: 'functional', message: 'Expected five.', evidence: { expected: '5', actual: '0' } }],
}

test('history selects the best attempt and shows earlier signals and code changes', async () => {
  const user = userEvent.setup()
  vi.stubGlobal('fetch', vi.fn(async (path: string) => response(path.endsWith('/r1') ? run : { items: [run], total: 1 })))
  render(<Runs runId="r1" onRun={vi.fn()} />)
  expect(await screen.findByLabelText('Candidate code')).toHaveTextContent('print(5)')
  await user.click(screen.getByRole('button', { name: 'Changes' }))
  expect(screen.getByLabelText('Changes from previous attempt')).toHaveTextContent('-print(0)')
  await user.click(screen.getByRole('button', { name: /Attempt 1/ }))
  expect(screen.getByLabelText('Candidate code')).toHaveTextContent('print(0)')
  expect(screen.getByText('Expected five.')).toBeInTheDocument()
  expect(screen.getByRole('button', { name: 'Changes' })).toBeDisabled()
})

test('history pagination and split filters make server-side queries', async () => {
  const user = userEvent.setup()
  const fetch = vi.fn(async () => response({ items: [run], total: 21 }))
  vi.stubGlobal('fetch', fetch)
  render(<Runs runId="" onRun={vi.fn()} />)
  await screen.findByText('Sum two numbers.')
  await user.click(screen.getByRole('button', { name: /Next/ }))
  await waitFor(() => expect(fetch).toHaveBeenCalledWith('/api/runs?limit=20&offset=20', expect.anything()))
  await user.selectOptions(screen.getByLabelText('Split'), 'train')
  await waitFor(() => expect(fetch).toHaveBeenCalledWith('/api/runs?limit=20&offset=0&split=train', expect.anything()))
})

test('catalog loading retrieves tasks beyond the first API page', async () => {
  const fetch = vi.fn(async (path: string) => response(path.includes('offset=0')
    ? { items: Array.from({ length: 100 }, (_, i) => ({ task_id: String(i) })), total: 101 }
    : { items: [{ task_id: 'last' }], total: 101 }))
  vi.stubGlobal('fetch', fetch)
  const tasks = await catalogTasks('mbpp', new AbortController().signal)
  expect(tasks).toHaveLength(101)
  expect(tasks.at(-1)?.task_id).toBe('last')
})

test('API validation and offline errors are human-readable', async () => {
  vi.stubGlobal('fetch', vi.fn(async () => response({ detail: [{ loc: ['body', 'seed'], msg: 'Must be positive' }] }, 422)))
  await expect(request('/solves')).rejects.toThrow('seed: Must be positive')
  vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')))
  await expect(request('/runs')).rejects.toThrow('Cannot reach the API')
})

test('Scala HTTP accepts task-defined request scenarios without assuming routes', async () => {
  const user = userEvent.setup()
  const fetch = vi.fn().mockResolvedValue(response(job, 202))
  vi.stubGlobal('fetch', fetch)
  render(<Solve jobId="" onJob={vi.fn()} onRun={vi.fn()} />)
  await user.type(screen.getByLabelText('Problem statement'), 'Serve the supplied HTTP contract in Scala.')
  await user.selectOptions(screen.getByLabelText('Language'), 'scala')
  await user.click(screen.getByText('Run settings & tests'))
  await user.selectOptions(screen.getByLabelText('Execution mode'), 'http')
  expect(screen.getByLabelText(/HTTP scenarios \(optional\)/)).toHaveValue('')
  await user.click(screen.getByRole('button', { name: /Start solve/ }))
  const body = JSON.parse(fetch.mock.calls[0][1].body)
  expect(body).toMatchObject({language:'scala',execution_mode:'http'})
  expect(body).not.toHaveProperty('tests')
})

test('switching away from Scala clears unsupported HTTP mode', async () => {
  const user = userEvent.setup()
  render(<Solve jobId="" onJob={vi.fn()} onRun={vi.fn()} />)
  await user.selectOptions(screen.getByLabelText('Language'), 'scala')
  await user.click(screen.getByText('Run settings & tests'))
  await user.selectOptions(screen.getByLabelText('Execution mode'), 'http')
  await user.selectOptions(screen.getByLabelText('Language'), 'python')
  expect(screen.getByLabelText('Execution mode')).toHaveValue('stdio')
  expect(screen.queryByRole('option', {name:'HTTP service with request tests'})).toBeNull()
})

test('requirement evidence remains visible and advisory scores are capped by coverage', () => {
  const report: Evaluation = { ...evaluation, status: 'review', evidence_confidence: .5,
    judge_grade: { overall_score: 1, feedback: 'Model opinion only' },
    constraint_grade: { coverage: .5, requirement_ledger: { coverage: .5,
      scope: 'Evidence from executed inputs only', requirements: [
        {id:'R-tested',text:'Tested obligation',state:'verified',evidence:[{type:'executed_test',test_id:'T1',result:'pass',independent:false,detail:{}}]},
        {id:'R-gap',text:'Missing obligation',state:'unverified',evidence:[]}] } } }
  render(<Evidence evaluation={report} testSource="generated" />)
  expect(screen.getByLabelText('Requirement evidence ledger')).toHaveTextContent('R-gap · unverified · Missing obligation')
  expect(screen.getByText('Advisory model judge')).toBeInTheDocument()
  expect(screen.queryByText('100%')).not.toBeInTheDocument()
  expect(screen.getByText(/Judge scores cannot exceed this coverage/)).toBeInTheDocument()
})

test('an empty requirement ledger from the API does not crash the evidence panel', () => {
  // Stdio runs without stated requirements store the ledger as {}.
  const report = { ...evaluation, constraint_grade: { coverage: 1, requirement_ledger: {} } } as unknown as Evaluation
  render(<Evidence evaluation={report} testSource="generated" />)
  expect(screen.queryByLabelText('Requirement evidence ledger')).toBeNull()
})

test('a problem that names Scala and an HTTP server selects Scala HTTP mode', async () => {
  const user = userEvent.setup()
  const fetch = vi.fn().mockResolvedValue(response(job, 202))
  vi.stubGlobal('fetch', fetch)
  render(<Solve jobId="" onJob={vi.fn()} onRun={vi.fn()} />)
  await user.type(screen.getByLabelText('Problem statement'), 'Build an HTTP server in Scala that stores notes.')
  await user.click(screen.getByRole('button', { name: /Start solve/ }))
  const body = JSON.parse(fetch.mock.calls[0][1].body)
  expect(body).toMatchObject({ language: 'scala', execution_mode: 'http' })
})
