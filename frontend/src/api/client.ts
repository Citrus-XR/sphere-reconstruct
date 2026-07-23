// バックエンドの API 型と fetch ヘルパ.
// Phase 7 で OpenAPI から自動生成に置き換える予定. それまでは手書きで最小限持つ.

export type PipelineState =
  | 'created'
  | 'inspected'
  | 'extracted'
  | 'reprojected'
  | 'masked'
  | 'reconstructed'
  | 'exported'

export type SourceKind = 'insv' | 'erp_video' | 'erp_images'

export interface Project {
  id: string
  name: string
  created_at: string
  updated_at: string
  source_kind: SourceKind | null
  source_path: string | null
  state: PipelineState
}

export interface Job {
  id: string
  project_id: string
  kind: string
  stage: string | null
  status: 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled'
  created_at: string
  started_at: string | null
  finished_at: string | null
  error_text: string | null
  pid: number | null
}

export interface EventEnvelope {
  id: number
  job_id: string | null
  project_id: string | null
  stage: string | null
  level: 'debug' | 'info' | 'warn' | 'error'
  message: string
  progress: number | null
  ts: string
}

const BASE = ''

const jsonHeaders = { 'Content-Type': 'application/json' }

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init)
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`${res.status} ${res.statusText}: ${text}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  listProjects: () => req<Project[]>('/api/projects'),
  getProject: (id: string) => req<Project>(`/api/projects/${id}`),
  createProject: (name: string) =>
    req<Project>('/api/projects', {
      method: 'POST',
      headers: jsonHeaders,
      body: JSON.stringify({ name }),
    }),
  setSource: (id: string, kind: SourceKind, path: string) =>
    req<Project>(`/api/projects/${id}/source`, {
      method: 'POST',
      headers: jsonHeaders,
      body: JSON.stringify({ kind, path }),
    }),
  runPipeline: (id: string) =>
    req<{ job_id: string }>(`/api/projects/${id}/run`, { method: 'POST' }),
  rerunStage: (id: string, stage: string) =>
    req<{ job_id: string }>(`/api/projects/${id}/rerun/${stage}`, { method: 'POST' }),
  getJob: (id: string) => req<Job>(`/api/jobs/${id}`),
  cancelJob: (id: string) =>
    req<{ cancelled: boolean }>(`/api/jobs/${id}/cancel`, { method: 'POST' }),
  getSettings: () => req<Record<string, unknown>>('/api/settings'),
}

// WebSocket ヘルパ. dev では /api/events が Vite proxy 経由で ws:// にアップグレードされる.
export const openEventStream = (
  opts: { jobId?: string; projectId?: string; since?: number },
  onEvent: (e: EventEnvelope) => void,
): WebSocket => {
  const params = new URLSearchParams()
  if (opts.since !== undefined) params.set('since', String(opts.since))
  if (opts.jobId) params.set('job_id', opts.jobId)
  if (opts.projectId) params.set('project_id', opts.projectId)

  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const url = `${proto}//${window.location.host}/api/events?${params.toString()}`
  const ws = new WebSocket(url)
  ws.onmessage = ev => {
    try {
      onEvent(JSON.parse(ev.data) as EventEnvelope)
    } catch {
      // 壊れたメッセージは無視.
    }
  }
  return ws
}
