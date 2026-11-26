import { useEffect, useState } from 'react'
import { ApiError, aborted, message, pathId, request } from './api'
import type { Evaluation, Job, JobStatus, SolveInput, SolveResult } from './types'

export const terminal = (status?: JobStatus) => status === 'completed' || status === 'failed' || status === 'interrupted'
export interface Progress {
  status?: JobStatus; stage: string; attempt?: number; budget?: number; runId?: string | null
  result?: SolveResult | null; error?: string | null; connectionError?: string
  request?: Partial<SolveInput>
  draftCode?: string
  attempts: { attempt_number: number; duplicate_of?: number | null; reused_from?: number | null; code: string; reward: number; evaluation: Evaluation }[]
}
const initial: Progress = { stage: 'queued', attempts: [] }

export function useJob(jobId: string) {
  const [progress, setProgress] = useState<Progress>(initial)
  useEffect(() => {
    setProgress(initial)
    if (!jobId) return
    const controller = new AbortController()
    const stream = new EventSource(`/api/jobs/${pathId(jobId)}/events`)
    let disposed = false
    let done = false
    let cursor = 0
    let checking = false
    let retry: ReturnType<typeof setTimeout> | undefined
    const stop = () => { done = true; stream.close(); clearTimeout(retry) }
    const snapshot = async () => {
      if (checking || disposed || done) return
      checking = true
      try {
        const job = await request<Job>(`/jobs/${pathId(jobId)}`, { signal: controller.signal })
        if (disposed) return
        const alreadyDone = done
        setProgress(previous => alreadyDone ? { ...previous, request: job.request } :
          { ...previous, request: job.request, status: previous.status === 'running' && job.status === 'queued' ? previous.status : job.status, runId: job.resource_id ?? previous.runId,
            result: job.result, error: job.error, connectionError: undefined,
            draftCode: job.draft_code ?? previous.draftCode })
        if (terminal(job.status)) stop()
      } catch (error) {
        if (disposed || aborted(error)) return
        setProgress(previous => ({ ...previous, connectionError: message(error) }))
        if (error instanceof ApiError && error.status === 404) {
          setProgress(previous => ({ ...previous, status: 'failed', error: 'This job was not found. You can start a new solve.', connectionError: undefined }))
          stop()
        }
      } finally { checking = false }
    }
    const checkAfterDisconnect = async () => {
      await snapshot()
      if (!disposed && !done) retry = setTimeout(checkAfterDisconnect, 3000)
    }
    const receive = (type: string) => (event: MessageEvent) => {
      if (disposed || done) return
      const id = Number(event.lastEventId)
      if (id && id <= cursor) return
      try {
        const data = JSON.parse(event.data)
        if (id) cursor = id
        setProgress(previous => {
          const base = { ...previous, connectionError: undefined }
          if (type === 'stage') return { ...base, status: 'running', stage: data.stage,
            attempt: data.attempt_number ?? previous.attempt, budget: data.attempt_budget ?? previous.budget,
            runId: data.history_run_id ?? previous.runId,
            draftCode: data.stage === 'draft_ready' ? data.code : previous.draftCode }
          if (type === 'attempt') return { ...base, attempts: [...previous.attempts.filter(a => a.attempt_number !== data.attempt_number), data] }
          if (type === 'completed' || type === 'failed' || type === 'interrupted') return {
            ...base, status: type, result: data.result ?? previous.result, error: data.error ?? data.message,
            runId: data.result?.history_run_id ?? previous.runId,
          }
          return { ...base, status: type as JobStatus }
        })
        if (['completed', 'failed', 'interrupted'].includes(type)) stop()
      } catch {
        setProgress(previous => ({ ...previous, connectionError: 'An update could not be read. Checking saved job status.' }))
        void snapshot()
      }
    }
    for (const type of ['queued', 'running', 'stage', 'attempt', 'completed', 'failed', 'interrupted']) {
      stream.addEventListener(type, receive(type) as EventListener)
    }
    stream.onopen = () => { clearTimeout(retry); void snapshot() }
    stream.onerror = () => {
      if (disposed || done) return
      setProgress(previous => ({ ...previous, connectionError: 'Connection interrupted. Reconnecting to saved progress…' }))
      clearTimeout(retry)
      void checkAfterDisconnect()
    }
    void snapshot()
    return () => { disposed = true; stop(); controller.abort() }
  }, [jobId])
  return progress
}
