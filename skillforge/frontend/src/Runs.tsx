import { ModelRequests } from './ModelRequests'
import { useEffect, useState } from 'react'
import { ArrowLeft, ArrowRight, History, RefreshCw, Signal } from 'lucide-react'
import { aborted, message, pathId, request } from './api'
import { Code, ErrorBox, Evidence, Status, date, duration } from './components'
import type { Page, Run, RunDetail } from './types'

function useResource<T>(path: string, refresh: number) {
  const [state, setState] = useState<{ data?: T; error?: string; loading: boolean }>({ loading: true })
  useEffect(() => {
    const controller = new AbortController()
    setState({ loading: true })
    request<T>(path, { signal: controller.signal }).then(data => setState({ data, loading: false }))
      .catch(error => { if (!aborted(error)) setState({ error: message(error), loading: false }) })
    return () => controller.abort()
  }, [path, refresh])
  return state
}

export function Runs({ runId, onRun }: { runId: string; onRun: (id: string) => void }) {
  const [offset, setOffset] = useState(0)
  const [split, setSplit] = useState('')
  const [refresh, setRefresh] = useState(0)
  const page = useResource<Page<Run>>(`/runs?limit=20&offset=${offset}${split ? `&split=${split}` : ''}`, refresh)
  if (runId) return <RunView key={runId} runId={runId} onBack={() => onRun('')} />
  return <>
    <div className="page-heading"><div><p className="eyebrow">EXPERIMENT HISTORY</p><h1>Every attempt, recorded.</h1><p>Explore saved runs and see how each solution changed.</p></div><button className="button secondary" disabled={page.loading} onClick={() => setRefresh(n => n + 1)}><RefreshCw size={16} />Refresh</button></div>
    <section className="panel">
      <div className="table-toolbar"><h2>Runs <span className="count">{page.data?.total ?? '—'}</span></h2><label className="filter">Split<select value={split} onChange={e => { setSplit(e.target.value); setOffset(0) }}><option value="">All splits</option><option value="adhoc">Ad hoc</option><option value="train">Training</option><option value="validation">Validation</option><option value="test">Test</option></select></label></div>
      {page.error && <div className="panel-padding"><ErrorBox>{page.error}</ErrorBox></div>}
      {page.loading ? <p className="loading" role="status">Loading runs…</p> : page.data?.items.length ? <>
        <div className="table-scroll"><table><thead><tr><th>Problem</th><th>Verdict</th><th>Model & language</th><th>Prompt</th><th>Started</th><th><span className="sr-only">Open run</span></th></tr></thead><tbody>{page.data.items.map(run => <tr key={run.run_id}>
          <td><button className="run-title" onClick={() => onRun(run.run_id)}>{run.benchmark_split === 'adhoc' ? run.problem_statement : run.task_id || run.problem_statement}</button><small className="table-subtitle">{run.benchmark_split === 'adhoc' ? run.run_id.slice(0, 12) : run.problem_statement}</small><span className="split-tag">{run.benchmark_split}</span></td>
          <td><Status value={run.final_status} /></td><td><span className="model-name">{run.model_name}</span><small className="table-subtitle">{run.language || 'python'}</small></td><td><span className="mono muted">{run.prompt_version}</span></td><td className="date-cell">{date(run.started_at)}</td><td><button className="icon-button" onClick={() => onRun(run.run_id)} aria-label={`Open run ${run.run_id}`}><ArrowRight size={17} /></button></td>
        </tr>)}</tbody></table></div>
      </> : !page.error && <div className="empty-progress"><div className="empty-icon"><History size={27} /></div><h3>No runs here yet</h3><p>{split ? 'Try another split, or start a new solve.' : 'Complete your first solve to start building a history.'}</p></div>}
      <div className="pagination"><span>{page.data?.total ? `${offset + 1}–${Math.min(offset + 20, page.data.total)} of ${page.data.total}` : '0 runs'}</span><div><button className="button secondary compact" disabled={page.loading || offset === 0} onClick={() => setOffset(n => Math.max(0, n - 20))}><ArrowLeft size={14} />Previous</button><button className="button secondary compact" disabled={page.loading || !page.data || offset + 20 >= page.data.total} onClick={() => setOffset(n => n + 20)}>Next<ArrowRight size={14} /></button></div></div>
    </section>
  </>
}

function RunView({ runId, onBack }: { runId: string; onBack: () => void }) {
  const [refresh, setRefresh] = useState(0)
  const [selected, setSelected] = useState<number | null>(null)
  const [showDiff, setShowDiff] = useState(false)
  const { data: run, error, loading } = useResource<RunDetail>(`/runs/${pathId(runId)}`, refresh)
  const attempt = run?.attempts.find(a => a.attempt_number === selected)
    || run?.attempts.find(a => a.attempt_number === run.best_attempt_number) || run?.attempts[0]
  const signals = run?.failure_signals.filter(s => s.attempt_number === attempt?.attempt_number) || []
  return <>
    <button className="text-button back-link" onClick={onBack}><ArrowLeft size={15} />All runs</button>
    <div className="page-heading"><div><p className="eyebrow">RUN DETAIL</p><h1>{run?.benchmark_split === 'adhoc' ? 'Solution history' : run?.task_id || 'Solution history'}</h1><p className="mono run-id">{runId}</p></div><button className="button secondary" disabled={loading} onClick={() => setRefresh(n => n + 1)}><RefreshCw size={16} />Refresh</button></div>
    {loading && <p className="loading" role="status">Loading run…</p>}
    {error && <ErrorBox>{error}</ErrorBox>}
    {run && <>
      <ModelRequests ownerId={run.request_log_id} />
      <section className="panel run-overview"><div className="section-heading"><h2>The problem</h2><Status value={run.final_status} /></div><p className="problem-text">{run.problem_statement}</p>
        <div className="run-metadata"><div><span>Generator</span><strong>{run.model_name}</strong></div><div><span>Language / prompt</span><strong>{run.language || 'python'} / {run.prompt_version}</strong></div><div><span>Attempts</span><strong>{run.attempts.length} / {run.attempt_budget ?? '—'}</strong></div><div><span>Duration / calls</span><strong>{duration(run.wall_clock_seconds)} / {run.model_calls ?? '—'}</strong></div></div>
        <p className="form-note">{date(run.started_at)} · {run.benchmark_split} · {run.execution_mode || 'stdio'}{run.entry_point ? ` (${run.entry_point})` : ''} · {run.test_source} tests{run.judge_model ? ` · Judge: ${run.judge_model}` : ''}</p>
        {!run.completed_at && <p className="notice">This run is unfinished. It may still be running or have been interrupted. Refresh to check for saved attempts.</p>}
      </section>
      {!attempt ? <div className="panel empty-progress"><h3>No saved attempts yet</h3><p>Refresh after the worker finishes an attempt.</p></div> : <div className="history-grid">
        <aside className="panel attempt-picker"><div className="panel-heading"><h2>Attempts</h2><span className="count">{run.attempts.length}</span></div><div aria-label="Select an attempt">{run.attempts.map(item => <button key={item.attempt_number} aria-pressed={item === attempt} className={item === attempt ? 'attempt-choice active' : 'attempt-choice'} onClick={() => { setSelected(item.attempt_number); setShowDiff(false) }}><span><strong>Attempt {item.attempt_number}{item.duplicate_of != null && <small>Repeated candidate · attempt {item.duplicate_of}</small>}</strong><small>{item.attempt_number === run.best_attempt_number ? 'Best candidate' : `Reward ${item.reward.toFixed(2)}`}</small></span><Status value={item.evaluation_status} /></button>)}</div></aside>
        <div className="attempt-detail"><section className="panel result-panel"><Evidence evaluation={{ ...attempt, status: attempt.evaluation_status }} testSource={run.test_source} />
          <div className="code-controls"><h3>Attempt {attempt.attempt_number}</h3><div className="segmented small"><button className={!showDiff ? 'selected' : ''} aria-pressed={!showDiff} onClick={() => setShowDiff(false)}>Code</button><button disabled={attempt.code_diff === null} className={showDiff ? 'selected' : ''} aria-pressed={showDiff} onClick={() => setShowDiff(true)}>Changes</button></div></div>
          {showDiff && !attempt.code_diff ? <p className="notice">No code changes from the previous attempt.</p> : <Code key={`${attempt.attempt_number}-${showDiff}`} code={showDiff ? attempt.code_diff! : attempt.candidate_code} diff={showDiff} title={showDiff ? 'Changes from previous attempt' : 'Candidate code'} />}
        </section>
        <section className="panel signals-panel"><div className="section-heading"><h3><Signal size={17} />Failure signals</h3><span className="count">{signals.length}</span></div>
          {signals.length ? signals.map((signal, index) => <div className="signal" key={index}><div className="tags"><span>{signal.category.replaceAll('_', ' ')}</span><span>{signal.strength}</span></div><p>{signal.message}</p><small>Source: {signal.source}</small><details><summary>View evidence</summary><pre>{JSON.stringify(signal.evidence, null, 2)}</pre></details></div>) : <p className="muted">No failure signals were recorded for this attempt.</p>}
          {attempt.feedback?.candidate_feedback && <details className="feedback-detail"><summary>Feedback sent for refinement</summary><pre>{attempt.feedback.candidate_feedback}</pre></details>}
        </section></div>
      </div>}
    </>}
  </>
}
