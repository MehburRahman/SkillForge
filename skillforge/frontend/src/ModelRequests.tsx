import { useState } from 'react'
import { message, request } from './api'
import { Code, ErrorBox } from './components'

interface ModelCall {
  call_id: number; role: string; stage: string; request: unknown
  response_text: string | null; error: string | null; usage: unknown
}

export function ModelRequests({ ownerId }: { ownerId?: string | null }) {
  const [calls, setCalls] = useState<ModelCall[]>([])
  const [loaded, setLoaded] = useState(false)
  const [error, setError] = useState('')
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(false)
  async function load() {
    if (loading) return
    setLoading(true)
    setError('')
    try {
      const data = await request<{ items: ModelCall[]; total: number }>(`/model-requests/${encodeURIComponent(ownerId!)}?limit=100&offset=${calls.length}`)
      setCalls(previous => [...previous, ...data.items]); setTotal(data.total); setLoaded(true)
    } catch (err) { setError(message(err)) }
    finally { setLoading(false) }
  }
  if (!ownerId) return null
  return <section className="panel signals-panel"><h3>Model requests</h3>
    {!loaded && <button className="button secondary" disabled={loading} onClick={load}>Load saved model requests</button>}
    {calls.map(call => <details key={call.call_id}><summary>{call.role} · {call.stage}</summary>
      <Code title="Model request" code={JSON.stringify(call.request, null, 2)} />
      <Code title="Model response" code={call.response_text || call.error || ''} />
      <pre>{JSON.stringify(call.usage, null, 2)}</pre>
    </details>)}
    {loaded && calls.length === 0 && <p>No saved requests for this run.</p>}
    {calls.length < total && <button className="button secondary" disabled={loading} onClick={load}>Load more requests</button>}
    {error && <ErrorBox>{error}</ErrorBox>}
  </section>
}
