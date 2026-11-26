import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi, test, expect } from 'vitest'
import { Evolution } from './Evolution'
import { ModelRequests } from './ModelRequests'

const response = (value: unknown) => new Response(JSON.stringify(value), { headers: { 'Content-Type': 'application/json' } })
const blocked = {
  active_prompt_id: 'P0', eligible: false, reason: 'Not ready: 0/20 new completed training runs.',
  training_runs: 0, qualifying_categories: [], failure_counts: {}, training_scopes: [],
  policy: { min_training_runs: 20, min_failure_categories: 3, min_category_runs: 3 },
}

test('readiness explains excluded ad hoc runs and blocks proposals without evidence', async () => {
  const fetch = vi.fn((url: string) => Promise.resolve(response(url.includes('readiness') ? blocked : { items: ['scala_http'] })))
  vi.stubGlobal('fetch', fetch)
  render(<Evolution />)
  expect(await screen.findByText(blocked.reason)).toBeInTheDocument()
  expect(screen.getByText(/Custom ad hoc solves do not count/)).toBeInTheDocument()
  expect(screen.getByRole('button', { name: 'Propose and validate' })).toBeDisabled()
  expect(fetch).toHaveBeenCalledTimes(2)
})

test('eligible proposal submits matching validation budgets and shows its result', async () => {
  const fetch = vi.fn((url: string, options?: RequestInit) => {
    if (options?.method === 'POST') return Promise.resolve(response({ job_id: 'e1' }))
    if (url.includes('/jobs/')) return Promise.resolve(response({ status: 'completed', result: { decision: 'rejected', active_prompt_id: 'P0' } }))
    return Promise.resolve(response(url.includes('readiness') ? { ...blocked, eligible: true, training_runs: 20, reason: 'Ready' } : { items: ['scala_http'] }))
  })
  vi.stubGlobal('fetch', fetch)
  render(<Evolution />)
  await screen.findByText('Ready')
  await userEvent.click(screen.getByRole('button', { name: 'Propose and validate' }))
  expect(await screen.findByText('rejected. Active prompt: P0.')).toBeInTheDocument()
  const posted = fetch.mock.calls.filter(([, options]) => options?.method === 'POST')
  expect(posted).toHaveLength(1)
  expect(JSON.parse(String(posted[0][1]?.body))).toMatchObject({ context_window: 8192, seed: 42, max_attempts: 3, catalog: 'scala_http' })
})

test('model requests are loaded on demand and show request, response, and usage', async () => {
  const fetch = vi.fn().mockResolvedValue(response({ total: 1, items: [{ call_id: 1, role: 'generator', stage: 'candidate:2',
    request: { prompt: 'Fix the compiler error', options: { num_ctx: 8192 } }, response_text: 'corrected source', usage: { prompt_eval_count: 600 }, error: null }] }))
  vi.stubGlobal('fetch', fetch)
  render(<ModelRequests ownerId="log-1" />)
  expect(fetch).not.toHaveBeenCalled()
  await userEvent.click(screen.getByRole('button', { name: 'Load saved model requests' }))
  await waitFor(() => expect(screen.getByText(/generator · candidate:2/)).toBeInTheDocument())
  expect(screen.getByLabelText('Model request')).toHaveTextContent('Fix the compiler error')
  expect(screen.getByLabelText('Model response')).toHaveTextContent('corrected source')
})
