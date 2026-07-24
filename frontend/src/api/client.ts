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

export interface ReconstructionCamera {
  id: number
  model: string
  width: number
  height: number
  params: number[]
}

export interface ReconstructionImage {
  id: number
  name: string
  camera_id: number
  qvec: number[]
  tvec: number[]
  position: number[]
  num_points: number
}

export interface ReconstructionData {
  cameras: ReconstructionCamera[]
  images: ReconstructionImage[]
  stats: {
    num_cameras: number
    num_images: number
    num_points3D: number
    mean_reprojection_error: number
    mean_track_length: number
    registered_ratio?: number
  }
  points_file: string
  points_stride: number
}

export interface ParsedPoints {
  count: number
  positions: Float32Array // 3 * count
  colors: Float32Array // 3 * count, 0..1
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
  getReconstruction: (id: string) =>
    req<ReconstructionData>(`/api/projects/${id}/reconstruction`),
  getMasks: (id: string) => req<MasksManifest>(`/api/projects/${id}/masks`),
  getFrames: (id: string) => req<FramesManifest>(`/api/projects/${id}/frames`),
}

export interface FramesManifest {
  kind: string
  count: number
  width: number | null
  height: number | null
  fps: number | null
  frames: { index: number; timestamp_sec: number | null }[]
}

// 抽出フレーム (fisheye) の URL.
export const frameImageUrl = (id: string, index: number, lens: number) =>
  `/api/projects/${id}/frames/${index}/image?lens=${lens}`

export interface MaskViewRecord {
  view: string
  lens: number
  path: string
  coverage: number
  detections: Record<string, number>
  coverage_warning?: boolean
}

export interface MasksManifest {
  kind: string
  prompt: string[]
  max_inference_size: number
  frames: { index: number; views: MaskViewRecord[] }[]
}

// pinhole 画像 / mask の URL を組み立てる.
export const pinholeUrl = (id: string, index: number, view: string, lens: number) =>
  `/api/projects/${id}/pinhole/${index}/${view}?lens=${lens}`
export const maskUrl = (id: string, index: number, view: string, lens: number) =>
  `/api/projects/${id}/pinhole/${index}/${view}?lens=${lens}&mask=true`

// points.bin をパースする. フォーマット (backend/colmap/web_preview.py と一致):
//   header: u32 count, u32 stride(=20)
//   各点: 3*f32 xyz, 3*u8 rgb, 1 pad, f32 error  (= 20 bytes)
export const fetchPoints = async (projectId: string): Promise<ParsedPoints> => {
  const res = await fetch(`/api/projects/${projectId}/reconstruction/points`)
  if (!res.ok) throw new Error(`points fetch failed: ${res.status}`)
  const buf = await res.arrayBuffer()
  const dv = new DataView(buf)
  const count = dv.getUint32(0, true)
  const stride = dv.getUint32(4, true)
  const positions = new Float32Array(count * 3)
  const colors = new Float32Array(count * 3)
  let off = 8
  for (let i = 0; i < count; i++) {
    positions[i * 3] = dv.getFloat32(off, true)
    positions[i * 3 + 1] = dv.getFloat32(off + 4, true)
    positions[i * 3 + 2] = dv.getFloat32(off + 8, true)
    colors[i * 3] = dv.getUint8(off + 12) / 255
    colors[i * 3 + 1] = dv.getUint8(off + 13) / 255
    colors[i * 3 + 2] = dv.getUint8(off + 14) / 255
    off += stride
  }
  return { count, positions, colors }
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
