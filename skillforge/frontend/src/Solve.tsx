import { useEffect, useRef, useState } from 'react'
import { ArrowRight, Check, Circle, LoaderCircle, Play, Plus, Terminal, Workflow } from 'lucide-react'
import { aborted, catalogTasks, message, request } from './api'
import { Code, ErrorBox, Evidence, Status, duration } from './components'
import { terminal, useJob } from './useJob'
import type { Job, ReasoningMode, SolveInput, Task } from './types'
import { DEFAULT_GENERATOR, DEMO_PROBLEMS, GENERATOR_MODELS } from './demoProblems'

const stageLabels: Record<string, string> = {
  request_logging: 'Preparing model request records', repeated_candidate: 'Reusing an identical compilation failure', queued: 'Waiting for the model worker', preparing: 'Preparing the task', analysis: 'Analysing the problem',
  test_generation: 'Generating tests', tests_ready: 'Tests ready', generation: 'Generating a solution',
  test_validation: 'Checking generated inputs and deriving answers',
  http_test_generation: 'Constructing HTTP request scenarios',
  http_reference_generation: 'Building an HTTP reference',
  http_reference_execution: 'Computing expected HTTP responses',
  tests_rejected: 'Regenerating tests after review', repair_rejected: 'Source edits rejected; checking repair limits',
  trusted_test_fallback: 'Continuing with independently verified cases',
  test_oracle_generation: 'Building a reference to check generated tests',
  test_oracle_execution: 'Running the test reference',
  test_oracle_result: 'Checking the reference results',
  test_oracle_rejected: 'Repairing the test reference within its limit',
  test_builder_execution: 'Building complete test inputs',
  test_builder_result: 'Checking the constructed tests',
  run_started: 'Saving run', verification: 'Running tests', evaluation: 'Evaluating the evidence',
  requirement_gaps: 'Generating tests for unverified requirements', requirement_evidence: 'Updating requirement evidence', requirement_gaps_exhausted: 'Coverage investigation limit reached',
  refinement: 'Refining the solution', finished: 'Saving results',
  compilation: 'Compiling the Scala candidate', test_execution: 'Running compiled tests',
  server_testing: 'Starting the server and testing requests',
  algorithm_plan: 'Planning the algorithm', algorithm_review: 'Reviewing the algorithm plan',
  algorithm_repair: 'Revising the algorithm after a failed test',
  draft_ready: 'Draft saved; preparing verification', unverified: 'Draft saved; verification incomplete',
  reference_anchor_validation: 'Checking the reference against verified cases',
  reference_anchors_verified: 'Reference passed the supplied cases',
}

export function Solve({ jobId, onJob, onRun }: { jobId: string; onJob: (id: string) => void; onRun: (id: string) => void }) {
  const [source, setSource] = useState('custom')
  const [demoId, setDemoId] = useState('')
  const [problem, setProblem] = useState('')
  const [language, setLanguage] = useState('python')
  const [catalog, setCatalog] = useState('problems')
  const [catalogs, setCatalogs] = useState<string[]>([])
  const [tasks, setTasks] = useState<Task[]>([])
  const [taskId, setTaskId] = useState('')
  const [catalogError, setCatalogError] = useState('')
  const [loadingTasks, setLoadingTasks] = useState(false)
  const [reload, setReload] = useState(0)
  const [context, setContext] = useState('8192')
  const [reasoningMode, setReasoningMode] = useState<ReasoningMode>('direct')
  const [model, setModel] = useState(DEFAULT_GENERATOR)
  // A separate family checks the 7B generator's code (tests, references, judge).
  const [testModel, setTestModel] = useState('granite4.2:8b')
  const [useJudge, setUseJudge] = useState(false)
  const [judge, setJudge] = useState('llama3.1:8b')
  // Random inputs answered by the executable reference catch wrong algorithms that
  // pass the few small base tests; 32 keeps a solve within a few extra seconds.
  const [differentialCases, setDifferentialCases] = useState('32')
  const [budget, setBudget] = useState('')
  const [config, setConfig] = useState('')
  const [seed, setSeed] = useState('')
  const [mode, setMode] = useState('stdio')
  const modeManuallySelected = useRef(false)
  const languageManuallySelected = useRef(false)
  const [entry, setEntry] = useState('solve')
  const [tests, setTests] = useState('')
  const [referenceTests, setReferenceTests] = useState('')
  const [error, setError] = useState('')
  const [submitting, setSubmitting] = useState(false)
  function loadDemo(id: string) {
    setDemoId(id)
    const demo = DEMO_PROBLEMS.find(item => item.id === id)
    if (!demo) return
    // A fast, honest demo: verified tests (no test generation), plan-and-review
    // reasoning (fixes wrong algorithms that repair patches cannot), a fixed
    // budget and a separate judge that only reviews candidates passing every test.
    setProblem(demo.problem); setLanguage(demo.language); setMode(demo.mode || 'stdio')
    setTests(JSON.stringify(demo.tests, null, 2)); setReasoningMode('plan_critique')
    setBudget('6'); setUseJudge(true); setModel(demo.model || DEFAULT_GENERATOR)
  }

  const submittingRef = useRef(false)
  const progress = useJob(jobId)
  const busy = submitting || (!!jobId && !terminal(progress.status))
  const selected = tasks.find(task => task.task_id === taskId)
  const result = progress.result

  useEffect(() => {
    // Follow a language the problem names ("in Scala") until the user picks one.
    if (languageManuallySelected.current) return
    const named = [['scala', /\bscala\b/i], ['javascript', /\b(?:javascript|node\.?js)\b/i], ['python', /\bpython\b/i]]
      .filter(([, pattern]) => (pattern as RegExp).test(problem)).map(([name]) => name as string)
    if (named.length === 1) setLanguage(named[0])
  }, [problem])

  useEffect(() => {
    if (modeManuallySelected.current) return
    setMode(language === 'scala' && /\bhttps?\s+(?:web\s+)?(?:server|service)\b/i.test(problem) ? 'http' : 'stdio')
  }, [problem, language])

  useEffect(() => {
    if (source !== 'catalog') return
    const controller = new AbortController()
    setCatalogError('')
    request<{ items: string[] }>('/catalogs', { signal: controller.signal }).then(data => setCatalogs(data.items))
      .catch(error => { if (!aborted(error)) setCatalogError(message(error)) })
    return () => controller.abort()
  }, [source, reload])

  useEffect(() => {
    if (source !== 'catalog') return
    const controller = new AbortController()
    setLoadingTasks(true); setTasks([]); setTaskId(''); setCatalogError('')
    catalogTasks(catalog, controller.signal).then(items => { setTasks(items); setLoadingTasks(false) })
      .catch(error => { if (!aborted(error)) { setCatalogError(message(error)); setLoadingTasks(false) } })
    return () => controller.abort()
  }, [source, catalog, reload])

  async function submit(event: React.FormEvent) {
    event.preventDefault()
    if (submittingRef.current || busy) return
    setError('')
    submittingRef.current = true; setSubmitting(true)
    try {
      const body: SolveInput = source === 'catalog'
        ? { task_id: taskId, catalog, model: model.trim() }
        : { problem: problem.trim(), language, model: model.trim(), execution_mode: mode }
      if (source === 'custom') {
        if (mode === 'function') body.entry_point = entry.trim()
        if (tests.trim()) {
          const parsed = JSON.parse(tests)
          if (!Array.isArray(parsed) || parsed.length === 0) throw new Error('Tests must be a non-empty JSON array.')
          body.tests = parsed
        }
        if (mode === 'stdio' && !body.tests && referenceTests.trim()) {
          const parsed = JSON.parse(referenceTests)
          if (!Array.isArray(parsed) || parsed.length === 0) throw new Error('Reference cases must be a non-empty JSON array.')
          body.reference_tests = parsed
        }
        if (mode === 'function' && !body.tests) throw new Error('Function mode needs assertion tests. Add them in the test field below.')
      }
      body.context_window = Number(context)
      if (source === 'custom' && mode !== 'function' && !body.tests && testModel.trim()) body.test_model = testModel.trim()
      if (reasoningMode !== 'direct') body.reasoning_mode = reasoningMode
      if (source === 'custom' && mode === 'stdio' && !body.tests && differentialCases !== '0') body.differential_cases = Number(differentialCases)
      if (useJudge) body.judge_model = judge.trim()
      if (budget) body.max_attempts = Number(budget)
      if (config) body.config = config as SolveInput['config']
      if (seed) {
        const value = Number(seed)
        if (!Number.isSafeInteger(value) || value < 0) throw new Error('Seed must be a non-negative safe integer.')
        body.seed = value
      }
      const job = await request<Job>('/solves', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
      onJob(job.job_id)
    } catch (error) { setError(error instanceof SyntaxError ? 'Tests contain invalid JSON. Check the example format.' : message(error)) }
    finally { submittingRef.current = false; setSubmitting(false) }
  }

  return <>
    <div className="page-heading"><div><p className="eyebrow">CODE WORKSPACE</p><h1>Solve a problem</h1><p>Generate, test, and refine a solution. Follow the evidence at every step.</p></div>
      {jobId && <button className="button secondary" disabled={busy && !progress.connectionError} onClick={() => onJob('')}><Plus size={16} />New solve</button>}</div>
    <div className="solve-grid">
      <section className="panel input-panel"><div className="panel-heading"><span className="step-number">01</span><h2>{jobId ? 'Submitted problem' : 'The problem'}</h2></div>
        {jobId ? <div className="submitted-summary">
          {progress.request ? <>
            <p className="problem-text">{progress.request.problem || progress.request.task_id || 'Saved solve'}</p>
            <div className="tags"><span>{progress.request.language || 'Catalog contract'}</span>{progress.request.catalog && <span>{progress.request.catalog}</span>}</div>
            <dl className="submitted-settings">
              <div><dt>Generator</dt><dd>{progress.request.model || 'Default generator'}</dd></div>
              {(progress.request.test_model || result?.test_model) && <div><dt>Test generation</dt><dd>{result?.test_source === 'trusted' ? 'Bypassed (supplied tests)' : result?.test_model || progress.request.test_model}</dd></div>}
              <div><dt>Judge</dt><dd>{progress.request.judge_model || 'Not requested'}</dd></div>
              <div><dt>Reasoning</dt><dd>{progress.request.reasoning_mode === 'plan_critique' ? 'Plan and review' : 'Direct'}</dd></div>
              <div><dt>Attempt budget</dt><dd>{progress.request.max_attempts ?? 'Chosen by the model'}</dd></div>
              {progress.request.entry_point && <div><dt>Function</dt><dd>{progress.request.entry_point}</dd></div>}
            </dl>
          </> : <p className="muted">{terminal(progress.status) ? 'Request details are not available.' : 'Loading submitted settings…'}</p>}
          <p className="form-note">You can leave this page and reopen the job link to follow its progress.</p>
        </div> : <form onSubmit={submit}>
          <fieldset disabled={busy}>
            <div className="segmented" aria-label="Problem source"><button type="button" className={source === 'custom' ? 'selected' : ''} aria-pressed={source === 'custom'} onClick={() => setSource('custom')}>Write a problem</button><button type="button" className={source === 'catalog' ? 'selected' : ''} aria-pressed={source === 'catalog'} onClick={() => setSource('catalog')}>Choose from catalog</button></div>
            {source === 'custom' ? <>
              {DEMO_PROBLEMS.length > 0 && <label className="field">Demo problem <span className="optional">optional</span><select aria-label="Demo problem" value={demoId} onChange={e => loadDemo(e.target.value)}><option value="">Write my own</option>{DEMO_PROBLEMS.map(demo => <option key={demo.id} value={demo.id}>{demo.title}</option>)}</select><small>Fills in a problem with verified tests. Supplied tests skip test generation, so the run takes minutes.</small></label>}
              <label className="field">Problem statement<textarea required maxLength={20000} value={problem} onChange={e => setProblem(e.target.value)} rows={9} placeholder={'Describe the task, inputs, outputs, and constraints.\n\nExample: Read two integers from standard input and print their sum.'} /></label>
              <label className="field">Language<select value={language} onChange={e => { languageManuallySelected.current = true; setLanguage(e.target.value); if (mode === 'http') setMode('stdio') }}><option value="python">Python</option><option value="javascript">JavaScript</option><option value="scala">Scala</option></select></label>
            </> : <>
              <label className="field">Catalog<select value={catalog} onChange={e => setCatalog(e.target.value)}>{Array.from(new Set([catalog, ...catalogs])).map(name => <option key={name}>{name}</option>)}</select></label>
              <label className="field">Task<select required value={taskId} disabled={loadingTasks} onChange={e => setTaskId(e.target.value)}><option value="">{loadingTasks ? 'Loading tasks…' : 'Select a task'}</option>{tasks.map(task => <option key={task.task_id} value={task.task_id} disabled={!['python', 'javascript', 'scala'].includes(task.language)}>{task.task_id} · {task.title}</option>)}</select></label>
              {selected && <div className="task-preview"><p>{selected.problem_statement}</p><div className="tags"><span>{selected.language}</span><span>{selected.execution_mode}</span><span>{selected.split}</span></div><small>Uses the catalog’s visible tests.</small></div>}
              {catalogError && <ErrorBox>{catalogError} <button type="button" className="text-button" onClick={() => setReload(n => n + 1)}>Retry</button></ErrorBox>}
            </>}
            <label className="field">Generator model<select value={GENERATOR_MODELS.some(item => item.name === model) ? model : ''} onChange={e => setModel(e.target.value)}>{GENERATOR_MODELS.map(item => <option key={item.name} value={item.name}>{item.name}</option>)}<option value="">Other installed model…</option></select>{!GENERATOR_MODELS.some(item => item.name === model) && <input required maxLength={200} value={model} onChange={e => setModel(e.target.value)} placeholder="e.g. llama3.1:8b" spellCheck={false} />}<small>{GENERATOR_MODELS.find(item => item.name === model)?.note ?? 'Type any model installed in Ollama.'}</small></label>
            {source === 'custom' && mode !== 'function' && <label className="field">Test generation model <span className="optional">optional</span><input maxLength={200} value={testModel} onChange={e => setTestModel(e.target.value)} disabled={!!tests.trim()} spellCheck={false} placeholder="Same as generator" /><small>Choose an installed model to generate and check tests, including reference code. Supplied tests skip this step.</small></label>}
            <label className="field">Reasoning mode<select value={reasoningMode} onChange={e => setReasoningMode(e.target.value as ReasoningMode)}><option value="direct">Direct</option><option value="plan_critique">Plan and review</option></select><small>Plan and review adds two requests before coding and may add planning during repair. The candidate budget stays the same.</small></label>
            <div className="judge-option"><label><input type="checkbox" checked={useJudge} onChange={e => setUseJudge(e.target.checked)} /><span><strong>Use a model judge</strong><small>Add a separate model’s assessment to the checks.</small></span></label></div>
            {useJudge && <label className="field">Judge model<input required maxLength={200} value={judge} onChange={e => setJudge(e.target.value)} spellCheck={false} /><small>Advisory assessment. Using the same model gives correlated opinions; executable checks remain essential.</small></label>}
            <details className="advanced"><summary>Run settings & tests</summary>
              {source === 'custom' && mode === 'stdio' && <label className="field">Random reference checks<select aria-label="Random reference checks" disabled={!!tests.trim()} value={differentialCases} onChange={e => setDifferentialCases(e.target.value)}><option value="0">Base tests only</option><option value="32">32 small inputs</option><option value="128">128 small inputs</option><option value="1000">Up to 1000 small inputs (slow)</option></select><small>Random inputs use your run seed and are checked against an executable reference. Agreement is evidence, not a proof.</small></label>}
              <label className="field">Model context (tokens)<input required type="number" min={1024} max={32768} value={context} onChange={e => setContext(e.target.value)} /><small>Saved with model requests for reproducible runs.</small></label>
              <label className="field">Experiment strategy<select value={config} onChange={e => setConfig(e.target.value)}>
                <option value="">Adaptive repair (default)</option><option value="full">Full feedback, fixed candidate budget</option>
                <option value="single_shot">Single attempt</option><option value="best_of_n">Independent candidates</option>
                <option value="scalar_repair">Repair using pass count</option>
              </select><small>Fixed-budget strategies count every candidate, including repeated code.</small></label>
              <div className="two-fields"><label className="field">Maximum attempts<input type="number" min={1} max={20} step={1} value={budget} onChange={e => setBudget(e.target.value)} placeholder="Let model decide" /><small>Leave empty for the model’s attempt budget.</small></label>
                <label className="field">Seed <span className="optional">optional</span><input type="number" min={0} max={Number.MAX_SAFE_INTEGER} step={1} value={seed} onChange={e => setSeed(e.target.value)} placeholder="e.g. 42" /></label></div>
              {source === 'custom' && <>
                <label className="field">Execution mode<select value={mode} onChange={e => { modeManuallySelected.current = true; setMode(e.target.value) }}><option value="stdio">Standard input / output</option><option value="function">Function with assertion tests</option>{language === 'scala' && <option value="http">HTTP service with request tests</option>}</select></label>
                {mode === 'function' && <label className="field">Function name<input required value={entry} onChange={e => setEntry(e.target.value)} /></label>}
                <label className="field">{mode === 'http' ? 'HTTP scenarios (optional)' : mode === 'function' ? 'Assertion tests (required)' : 'Tests (optional)'}<textarea value={tests} onChange={e => setTests(e.target.value)} rows={4} spellCheck={false} placeholder={mode === 'http' ? '[{"requests":[{"method":"POST","path":"/your-route","body":"your input","expected_status":200,"expected_body":"expected result"}],"category":"normal"}]' : mode === 'function' ? (language === 'scala' ? '[{"assertion": "assert(solve(2, 3) == 5)", "category": "normal"}]' : language === 'javascript' ? '[{"assertion": "assert.strictEqual(solve(2, 3), 5);", "category": "normal"}]' : '[{"assertion": "assert solve(2, 3) == 5", "category": "normal"}]') : '[{"input": "2 3", "expected_output": "5", "category": "normal"}]'} /><small>{mode === 'http' ? 'Leave empty for model-generated request scenarios and executable reference answers. Each scenario uses a fresh server; its requests share state. HTTP execution requires a configured runner.' : mode === 'function' ? 'Provide assertions in the selected language.' : 'Paste official or otherwise trusted tests to skip test generation. Leave empty to use the test generation model.'}</small></label>
                {mode === 'stdio' && <label className="field">Verified reference cases <span className="optional">optional</span><textarea value={referenceTests} onChange={e => setReferenceTests(e.target.value)} disabled={!!tests.trim()} rows={3} spellCheck={false} placeholder="Paste independently verified input, expected_output and category objects as a JSON array." /><small>Checks the generated reference while retaining generated tests. Use official cases or independently checked answers.</small></label>}
              </>}
            </details>
          </fieldset>
          {error && <ErrorBox>{error}</ErrorBox>}
          <button className="button primary solve-button" disabled={busy || (source === 'catalog' && (!selected || loadingTasks))} type="submit">{busy ? <LoaderCircle size={17} className="spin" /> : <Play size={16} />}{submitting ? 'Submitting…' : busy ? 'Solve in progress' : 'Start solve'}{!busy && <ArrowRight size={16} className="ml-auto" />}</button>
          <p className="form-note">Runs locally through Ollama. Results are saved to your history.</p>
        </form>}
      </section>
      <div className="output-column">
        <section className="panel progress-panel"><div className="panel-heading"><span className="step-number">02</span><h2>Live progress</h2>{jobId && <Status value={result?.status === 'unverified' ? 'unverified' : progress.status || 'connecting'} />}</div>
          {!jobId ? <div className="empty-progress"><div className="empty-icon"><Workflow size={29} strokeWidth={1.5} /></div><h3>Ready when you are</h3><p>Your solution’s journey appears here,<br />from analysis to the final verdict.</p><div className="pipeline-preview"><span>Analyse</span><ArrowRight size={13} /><span>Generate</span><ArrowRight size={13} /><span>Verify</span></div></div> : <>
            <div className="live-state" role="status">{terminal(progress.status) ? progress.status === 'completed' && result?.status !== 'unverified' ? <Check size={22} /> : <Circle size={22} /> : <LoaderCircle className="spin" size={22} />}<div><strong>{result?.status === 'unverified' ? 'Draft saved; verification incomplete' : progress.status === 'completed' && result?.status === 'fail' ? 'Finished; solution failed verification' : progress.status === 'completed' && result?.status === 'review' ? 'Finished; solution needs review' : progress.status === 'completed' ? 'Solve completed' : progress.status === 'failed' ? progress.error?.startsWith('Infrastructure error:') ? 'Verification infrastructure failed; draft not graded' : 'Solve could not finish' : progress.status === 'interrupted' ? 'Solve interrupted' : stageLabels[progress.stage] || 'Working on your solution'}</strong><p>{result && terminal(progress.status) ? `${result.attempt_count} graded attempt${result.attempt_count === 1 ? '' : 's'}; graded-attempt budget ${result.attempt_budget}; preliminary compile repairs have a separate allowance.` : progress.attempt ? `Attempt ${progress.attempt}${progress.budget ? ` of ${progress.budget}` : ''}` : progress.status === 'failed' && progress.error?.startsWith('Infrastructure error:') ? 'The verifier failed; this is not a verdict on your code.' : 'Progress is saved while the worker runs.'}</p></div></div>
            {progress.connectionError && <ErrorBox>{progress.connectionError}</ErrorBox>}
            {progress.error && <ErrorBox>{progress.error}</ErrorBox>}
            {progress.attempts.length > 0 && <ol className="attempt-progress">{progress.attempts.map(attempt => <li key={attempt.attempt_number}><span>Attempt {attempt.attempt_number}{attempt.duplicate_of != null && ` · Repeated candidate from attempt ${attempt.duplicate_of}`}</span><Status value={attempt.evaluation.status} /></li>)}</ol>}
            {(result?.history_run_id || progress.runId) && <button className="text-button history-link" onClick={() => onRun(result?.history_run_id || progress.runId!)}>View saved run <ArrowRight size={14} /></button>}
          </>}
        </section>
        {result ? <section className="panel result-panel">{result.evaluation ? <Evidence evaluation={result.evaluation} testSource={result.test_source} /> : <><div className="section-heading"><h3>Verification incomplete</h3><Status value="unverified" /></div><p className="notice">The draft has not been graded. Supply verified tests or reference cases, or choose another test model.</p>{result.validation_error && <details><summary>View validation failure</summary><pre>{result.validation_error}</pre></details>}</>}
          <div className="result-meta"><span>{result.attempt_count} / {result.attempt_budget} attempts</span><span>{result.model_calls} model calls</span><span>{duration(result.wall_clock_seconds)}</span><span>{result.prompt_id}</span></div>
          {result.stage_metrics && <p className="form-note">Compiled candidates: {result.stage_metrics.compiled_candidates}/{result.stage_metrics.evaluated_candidates}. All tests passed: {result.stage_metrics.all_tests_passed_candidates}. {result.stage_metrics.repaired_after_initial_failure ? 'A later candidate passed after the first failed.' : ''}</p>}
          {!!result.stage_metrics?.diagnostic_candidates && <p className="form-note">Preliminary candidates checked: {result.stage_metrics.diagnostic_candidates}. Candidates with compiler failures: {result.stage_metrics.diagnostic_compilation_failures}. These checks are separate from graded attempts.</p>}
          {result.test_validation?.coverage_complete === false && <p className="notice">Test coverage is incomplete. The available cases were executed, but passing them does not establish the missing requirement. {result.test_validation.coverage_warning}</p>}
          {result.test_validation?.status === 'trusted_samples_only' && <p className="notice">Generated tests could not be verified. This result covers only the independently supplied cases. {result.test_validation.generated_tests_error}</p>}
          {result.test_validation?.differential && <p className="form-note">Added {result.test_validation.differential.added} reference checks with seed {result.test_validation.differential.seed}. These use a model-generated reference; agreement is not independent verification.</p>}
          {result.test_validation?.request_count != null && <p className="form-note">Generated HTTP checks: {result.test_validation.scenario_count} scenarios, {result.test_validation.request_count} requests. Expected responses came from an executed model-written reference; agreement is not independent verification.</p>}
          {result.test_validation?.differential_error && <p className="notice">Additional random checks could not be completed: {result.test_validation.differential_error}</p>}
          {(result.input_format || result.output_format) && <details><summary>Input and output format</summary><dl className="submitted-settings">
            {result.input_format && <div><dt>Input</dt><dd>{result.input_format}</dd></div>}
            {result.output_format && <div><dt>Output</dt><dd>{result.output_format}</dd></div>}
          </dl></details>}
          <Code key={result.code} code={result.code} title={result.status === 'unverified' ? 'Unverified draft' : `Best solution · attempt ${result.best_attempt_number}`} /></section>
          : progress.attempts.length > 0 ? <section className="panel result-panel"><p className="eyebrow">LATEST ATTEMPT · FINAL RESULT PENDING</p><Evidence evaluation={progress.attempts.at(-1)!.evaluation} /><Code key={progress.attempts.at(-1)!.attempt_number} code={progress.attempts.at(-1)!.code} title="Latest candidate" /></section>
          : progress.draftCode ? <section className="panel result-panel"><p className="notice">{terminal(progress.status) ? 'Verification did not finish. This saved draft has not been graded.' : 'Verification is pending. This draft has not been graded.'}</p><Code code={progress.draftCode} title="Unverified draft" /></section>
          : <div className="quiet-note"><Terminal size={18} /><p>Generated code, test results, and evaluation evidence will appear here.</p></div>}
      </div>
    </div>
  </>
}
