import { useEffect, useState } from 'react'
import { aborted, message, request } from './api'
import { ErrorBox, Status } from './components'
import { ModelRequests } from './ModelRequests'

interface Readiness {
  active_prompt_id: string; eligible: boolean; reason: string; training_runs: number
  qualifying_categories: string[]; failure_counts: Record<string, number>
  policy: { min_training_runs: number; min_failure_categories: number; min_category_runs: number }
  training_scopes: [string, string][]
  last_proposal?: { status: string; reason: string } | null
}

export function Evolution() {
  const [catalogs, setCatalogs] = useState<string[]>([])
  const [catalog, setCatalog] = useState('scala_http')
  const [readiness, setReadiness] = useState<Readiness | null>(null)
  const [error, setError] = useState('')
  const [refresh, setRefresh] = useState(0)
  const [model, setModel] = useState('qwen2.5-coder:7b')
  const [context, setContext] = useState(8192)
  const [jobId, setJobId] = useState('')
  const [roundId, setRoundId] = useState('')
  const [busy, setBusy] = useState(false)
  const [outcome, setOutcome] = useState('')

  useEffect(() => {
    const controller = new AbortController()
    request<{ items: string[] }>('/catalogs', { signal: controller.signal }).then(data => setCatalogs(data.items))
      .catch(err => { if (!aborted(err)) setError(message(err)) })
    return () => controller.abort()
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    setReadiness(null); setError('')
    request<Readiness>(`/evolutions/readiness?catalog=${encodeURIComponent(catalog)}`, { signal: controller.signal })
      .then(setReadiness).catch(err => { if (!aborted(err)) setError(message(err)) })
    return () => controller.abort()
  }, [catalog, refresh])

  useEffect(() => {
    if (!jobId) return
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout>
    const poll = async () => {
      try {
        const job = await request<{ status: string; error?: string; request?: { run_id?: string }; result?: { round_id?: string; decision?: string; reason?: string; active_prompt_id?: string } }>(`/jobs/${encodeURIComponent(jobId)}`, { signal: controller.signal })
        setRoundId(job.result?.round_id || job.request?.run_id || '')
        if (['completed', 'failed', 'interrupted'].includes(job.status)) {
          setBusy(false)
          setOutcome(job.error || job.result?.reason || `${job.result?.decision || job.status}. Active prompt: ${job.result?.active_prompt_id || 'unchanged'}.`)
          setRefresh(value => value + 1)
        } else { timer = setTimeout(poll, 2000) }
      } catch (err) { if (!aborted(err)) { setError(message(err)); setBusy(false) } }
    }
    void poll()
    return () => { controller.abort(); clearTimeout(timer) }
  }, [jobId])

  async function evolve(event: React.FormEvent) {
    event.preventDefault()
    if (busy || !readiness?.eligible) return
    setBusy(true); setError(''); setOutcome('')
    try {
      const job = await request<{ job_id: string }>('/evolutions', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ catalog, model, context_window: context, seed: 42, max_attempts: 3 }),
      })
      setJobId(job.job_id)
    } catch (err) { setError(message(err)); setBusy(false) }
  }

  return <>
    <div className="page-heading"><div><p className="eyebrow">PROMPT EVOLUTION</p><h1>Evolution readiness</h1><p>Use training evidence to propose instructions, then compare them on separate validation tasks.</p></div></div>
    <section className="panel input-panel evolution-panel">
      <form onSubmit={evolve}><fieldset disabled={busy}>
        <label className="field">Validation catalog<select value={catalog} onChange={event => setCatalog(event.target.value)}>{Array.from(new Set([catalog, ...catalogs])).map(name => <option key={name}>{name}</option>)}</select></label>
        {readiness && <>
          <div className="grade-grid"><div><span>Active prompt</span><strong>{readiness.active_prompt_id}</strong></div><div><span>New training runs</span><strong>{readiness.training_runs} / {readiness.policy.min_training_runs}</strong></div><div><span>Recurring categories</span><strong>{readiness.qualifying_categories.length} / {readiness.policy.min_failure_categories}</strong></div></div>
          <p role="status">{readiness.reason}</p>
          {readiness.last_proposal && <p className="notice">Last proposal ({readiness.last_proposal.status}): {readiness.last_proposal.reason}</p>}
          <p className="form-note">Each category must occur in at least {readiness.policy.min_category_runs} distinct training runs. Custom ad hoc solves do not count. Use a separate workspace for each language and execution mode.</p>
          {readiness.training_scopes.length > 0 && <p>Training workspace: {readiness.training_scopes.map(pair => pair.join(' / ')).join(', ')}</p>}
          {Object.keys(readiness.failure_counts).length > 0 && <ul>{Object.entries(readiness.failure_counts).map(([category, count]) => <li key={category}>{category}: {count} runs</li>)}</ul>}
        </>}
        <label className="field">Evolution model<input required value={model} onChange={event => setModel(event.target.value)} /></label>
        <label className="field">Model context (tokens)<input type="number" required min={1024} max={32768} value={context} onChange={event => setContext(Number(event.target.value))} /></label>
        <p className="form-note">Uses seed 42 and three candidates per validation task for both prompts. A candidate is promoted only when it meets the existing validation criteria.</p>
        <div className="evolution-actions"><button className="button primary" disabled={!readiness?.eligible || busy}>{busy ? 'Evolution in progress…' : 'Propose and validate'}</button>
        <button type="button" className="button secondary" onClick={() => setRefresh(value => value + 1)}>Refresh readiness</button></div>
      </fieldset></form>
      {jobId && <p>Job: {jobId} {busy && <Status value="running" />}</p>}
      {outcome && <p role="status">{outcome}</p>}
      {error && <ErrorBox>{error}</ErrorBox>}
    </section>
    <ModelRequests key={roundId} ownerId={roundId} />
  </>
}
