import type { Page, Task } from './types'

export class ApiError extends Error {
  constructor(message: string, public status: number) { super(message) }
}

export async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  let response: Response
  try {
    response = await fetch(`/api${path}`, options)
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') throw error
    throw new ApiError('Cannot reach the API. Check that the backend is running, then retry.', 0)
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    const detail = body.detail
    const message = Array.isArray(detail)
      ? detail.map((item: { loc?: string[]; msg?: string }) => `${item.loc?.slice(1).join('.') || 'Request'}: ${item.msg}`).join('; ')
      : typeof detail === 'string' ? detail : `The API returned an error (${response.status}).`
    throw new ApiError(message, response.status)
  }
  return response.json()
}

export async function catalogTasks(catalog: string, signal: AbortSignal): Promise<Task[]> {
  const items: Task[] = []
  let total = 1
  while (items.length < total) {
    const page = await request<Page<Task>>(`/tasks?catalog=${encodeURIComponent(catalog)}&limit=100&offset=${items.length}`, { signal })
    items.push(...page.items)
    total = page.total
    if (!page.items.length) break
  }
  return items
}

export function message(error: unknown) { return error instanceof Error ? error.message : 'An unexpected error occurred.' }
export function aborted(error: unknown) { return error instanceof Error && error.name === 'AbortError' }
export function pathId(value: string) { return encodeURIComponent(value) }
