import { useEffect, useState } from 'react'
import * as Tabs from '@radix-ui/react-tabs'
import { Braces, Code2, History, ArrowUpRight, Cpu } from 'lucide-react'
import { request } from './api'
import { Runs } from './Runs'
import { Solve } from './Solve'
import { Evolution } from './Evolution'

function currentRoute() {
  const [tab, rawId = ''] = window.location.hash.slice(1).split('/')
  let id = ''
  try { id = decodeURIComponent(rawId) } catch { /* A malformed link opens the empty tab. */ }
  return { tab: tab === 'runs' ? 'runs' : tab === 'evolution' ? 'evolution' : 'solve', id }
}

function lastJob() { try { return sessionStorage.getItem('skillforge.lastJob') || '' } catch { return '' } }

export default function App() {
  const [route, setRoute] = useState(currentRoute)
  const [savedJob, setSavedJob] = useState(() => {
    const opened = currentRoute()
    return opened.tab === 'solve' && opened.id ? opened.id : lastJob()
  })
  const [connected, setConnected] = useState<boolean | null>(null)
  useEffect(() => {
    const changed = () => {
      const opened = currentRoute()
      setRoute(opened)
      if (opened.tab === 'solve' && opened.id) setSavedJob(opened.id)
    }
    window.addEventListener('hashchange', changed)
    return () => window.removeEventListener('hashchange', changed)
  }, [])
  useEffect(() => {
    const controller = new AbortController()
    const check = () => request<{ worker_alive: boolean }>('/health', { signal: controller.signal })
      .then(data => setConnected(data.worker_alive)).catch(() => { if (!controller.signal.aborted) setConnected(false) })
    void check()
    const interval = setInterval(check, 15000)
    return () => { clearInterval(interval); controller.abort() }
  }, [])
  function navigate(tab: string, id = '') {
    window.location.hash = `${tab}${id ? `/${encodeURIComponent(id)}` : ''}`
    setRoute({ tab, id })
    if (tab === 'solve') {
      setSavedJob(id)
      try { if (id) sessionStorage.setItem('skillforge.lastJob', id); else sessionStorage.removeItem('skillforge.lastJob') } catch { /* The URL still supports reconnecting. */ }
    }
  }
  return <Tabs.Root value={route.tab} onValueChange={tab => navigate(tab, tab === 'solve' ? savedJob : '')} className="app-shell">
    <aside className="sidebar"><a className="brand" href="#solve" onClick={e => { e.preventDefault(); navigate('solve', savedJob) }}><span className="brand-symbol"><Braces size={24} /></span><span>SkillForge<small>CODE · EVALUATE · REFINE</small></span></a>
      <p className="nav-label">WORKSPACE</p><Tabs.List className="navigation" aria-label="Workspace navigation"><Tabs.Trigger value="solve"><Code2 size={19} />Solve<span>01</span></Tabs.Trigger><Tabs.Trigger value="runs"><History size={19} />Runs<span>02</span></Tabs.Trigger><Tabs.Trigger value="evolution"><Cpu size={19} />Evolution<span>03</span></Tabs.Trigger></Tabs.List>
      <div className="sidebar-bottom"><div className="local-label"><Cpu size={18} /><span>Local workspace<small>Powered by your models</small></span></div><a href="http://127.0.0.1:8000/docs" target="_blank" rel="noreferrer">API reference<ArrowUpRight size={14} /></a></div>
    </aside>
    <div className="workspace"><header className="topbar"><div><span>Workspace</span><span className="crumb">/</span><strong>{route.tab === 'runs' ? 'Runs' : route.tab === 'evolution' ? 'Evolution' : 'Solve'}</strong></div><span className={`connection ${connected ? 'online' : ''}`} role="status"><i />{connected === null ? 'Connecting to API' : connected ? 'API connected' : 'API offline'}</span></header>
      <main><Tabs.Content value="solve"><Solve jobId={route.tab === 'solve' ? route.id : ''} onJob={id => navigate('solve', id)} onRun={id => navigate('runs', id)} /></Tabs.Content><Tabs.Content value="runs"><Runs runId={route.tab === 'runs' ? route.id : ''} onRun={id => navigate('runs', id)} /></Tabs.Content><Tabs.Content value="evolution"><Evolution /></Tabs.Content></main>
      <footer>SkillForge<span>Every verdict has evidence.</span></footer>
    </div>
  </Tabs.Root>
}
