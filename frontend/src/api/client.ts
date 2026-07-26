// バックエンドの API 型と fetch ヘルパ. Browser 固有の binary points parser もここに集約する.

export type PipelineState =
  | 'created'
  | 'inspected'
  | 'extracted'
  | 'reprojected'
  | 'masked'
  | 'features_extracted'
  | 'matched'
  | 'reconstructed'
  | 'aligned'
  | 'denoised'
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
  // 工程に保存された UI 設定 (step パラメータ / モード / 無効化). リロードで復元する.
  ui_state: { params?: Record<string, unknown>; reconMode?: string; disabled?: string[] } | null
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
  // i18n キー + 補間引数. バックエンドが安定キーを付けた行のみ非 null. message は
  // 未 key 行 / 翻訳欠落時のフォールバック. msg_args はここで JSON 文字列からパース済み.
  msg_key: string | null
  msg_args: Record<string, unknown> | null
  progress: number | null
  // 'log' = Console 表示行 / 'progress' = 進捗のみの一時イベント (環形インジケータ駆動, Console 非表示).
  kind: 'log' | 'progress'
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
    camera_center_span?: number[]
    camera_trajectory_diameter?: number
    unique_camera_centers?: number
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
  runPipeline: (id: string, paramsByStage?: Record<string, Record<string, unknown>>, skip?: string[]) =>
    req<{ job_id: string }>(`/api/projects/${id}/run`, {
      method: 'POST',
      headers: jsonHeaders,
      body: JSON.stringify({ params_by_stage: paramsByStage ?? null, skip: skip ?? null }),
    }),
  rerunStage: (id: string, stage: string, paramsByStage?: Record<string, Record<string, unknown>>) =>
    req<{ job_id: string }>(`/api/projects/${id}/rerun/${stage}`, {
      method: 'POST',
      headers: jsonHeaders,
      body: JSON.stringify({ params_by_stage: paramsByStage ?? null }),
    }),
  getJob: (id: string) => req<Job>(`/api/jobs/${id}`),
  cancelJob: (id: string) =>
    req<{ cancelled: boolean }>(`/api/jobs/${id}/cancel`, { method: 'POST' }),
  getSettings: () => req<Record<string, unknown>>('/api/settings'),
  getReconstruction: (id: string) =>
    req<ReconstructionData>(`/api/projects/${id}/reconstruction`),
  getFrames: (id: string) => req<FramesManifest>(`/api/projects/${id}/frames`),
  getMasks: (id: string) => req<MasksManifest>(`/api/projects/${id}/masks`),
  getDenoise: (id: string) => req<DenoiseManifest>(`/api/projects/${id}/denoise`),
  getExportInfo: (id: string) => req<ExportInfo>(`/api/projects/${id}/export-info`),
  putUiState: (id: string, ui: Record<string, unknown>) =>
    req<Project>(`/api/projects/${id}/ui-state`, {
      method: 'PUT', headers: jsonHeaders, body: JSON.stringify({ ui }),
    }),
  getFisheyeRegion: (id: string) =>
    req<FisheyeRegion>(`/api/projects/${id}/fisheye-region`),
  putFisheyeRegion: (id: string, region: FisheyeRegion) =>
    req<FisheyeRegion>(`/api/projects/${id}/fisheye-region`, {
      method: 'PUT',
      headers: jsonHeaders,
      body: JSON.stringify(region),
    }),
  // system / filesystem / stages.
  getSystemStats: () => req<SystemStats>('/api/system/stats'),
  getDoctor: () => req<DoctorReport>('/api/system/doctor'),
  getFsRoots: () => req<{ roots: string[] }>('/api/fs/roots'),
  getFsDrives: () => req<{ drives: string[] }>('/api/fs/drives'),
  browseFs: (path: string) =>
    req<FsListing>(`/api/fs/browse?path=${encodeURIComponent(path)}`),
  getSourceInfo: (id: string) => req<SourceInfo>(`/api/projects/${id}/source-info`),
  getStages: (id: string) => req<StagesStatus>(`/api/projects/${id}/stages`),
  clearStage: (id: string, stage: string) =>
    req<{ cleared: string; invalidated: string[]; state: string }>(`/api/projects/${id}/stages/${stage}/clear`, {
      method: 'POST',
    }),
  clearOutputs: (id: string) =>
    req<{ cleared: string; state: string }>(`/api/projects/${id}/clear-outputs`, { method: 'POST' }),
  deleteProject: (id: string) =>
    req<{ deleted: string }>(`/api/projects/${id}`, { method: 'DELETE' }),
}

export interface SystemGpu {
  name: string
  util_percent: number | null
  mem_used_mb: number | null
  mem_total_mb: number | null
}
export interface SystemStats {
  cpu_percent: number | null
  ram: { percent: number; used_mb: number; total_mb: number } | null
  gpus: SystemGpu[]
}
export interface DoctorCheck {
  ok: boolean
  optional?: boolean
  message: string
  path?: string | null
  version?: string
  capabilities?: Record<string, boolean>
}
export interface DoctorReport {
  ready: boolean
  platform: Record<string, string>
  checks: Record<string, DoctorCheck>
}
export interface FsEntry {
  name: string
  path: string
  is_dir: boolean
  size?: number
  ext?: string
}
export interface FsListing {
  path: string
  parent: string | null
  dirs: FsEntry[]
  files: FsEntry[]
}
export interface SourceInfo {
  kind: string | null
  duration_sec: number | null
  fps?: number | null
  width?: number | null
  height?: number | null
  nb_frames?: number | null
}
export interface StageStatus {
  stage: string
  has_output: boolean
  status: string | null // null=未実行, 'succeeded'|'failed'|'running'|'cancelled'
  error_text: string | null
  job_id: string | null
  started_at: string | null
  finished_at: string | null
  params: Record<string, unknown> | null
  extra: Record<string, unknown> | null
}
export interface StagesStatus {
  project_id: string
  state: PipelineState
  stages: StageStatus[]
}

// 魚眼の円形有効領域 (正規化 cx/cy/r, 画像幅基準). lens0=front, lens1=back.
export interface LensCircle {
  cx: number
  cy: number
  r: number
}
export interface FisheyeRegion {
  lens0: LensCircle
  lens1: LensCircle
  saved?: boolean
}

export interface FrameSelection {
  mode: string
  selected: number
  candidates?: number
  fallback?: boolean
  reasons?: { blur: number; exposure: number; few_features: number }
}

export interface FrameInfo {
  index: number
  timestamp_sec: number | null
  score?: { sharpness: number; features?: number } | null
}

export interface FramesManifest {
  kind: string
  count: number
  width: number | null
  height: number | null
  fps: number | null
  selection: FrameSelection | null
  frames: FrameInfo[]
}

// generate_masks の manifest. kind により frame ごとの構造が異なる (fisheye は lenses, pinhole/erp は views).
export interface MaskLensRecord {
  lens: number
  path: string
  coverage: number
  coverage_warning?: boolean
}
export interface MaskFrameRecord {
  index: number
  lenses?: MaskLensRecord[]
  views?: Array<{ view: string; lens: number; path: string; coverage: number; coverage_warning?: boolean }>
}
export interface MasksManifest {
  kind: string
  prompt?: string[]
  frames: MaskFrameRecord[]
}

export interface DenoiseFrameRecord {
  index: number
  lens0?: string
  lens1?: string
  erp?: string
}
export interface DenoiseManifest {
  method: 'off' | 'fastdvdnet' | 'ffmpeg_adaptive'
  device: string | null
  frames: DenoiseFrameRecord[]
}

// export_dataset の出力ディレクトリ (絶対パス).
export interface ExportInfo {
  dir: string
  dataset_dir: string | null
  preview_dir: string | null
  train_configs_dir: string | null
  training_output_dir: string
  gui_integration: {
    train_configs_auto_applied: boolean
    warnings: string[]
    required_settings: {
      strategy: string
      gut: boolean
      undistort: boolean
      mask_mode: string
    }
  } | null
  command_template: {
    executable: string
    arguments: string[]
    run_name_placeholder: string
    requires_unique_run_name: boolean
  } | null
}

// 抽出フレーム (fisheye) の URL.
export const frameImageUrl = (id: string, index: number, lens: number) =>
  `/api/projects/${id}/frames/${index}/image?lens=${lens}`
// native fisheye の生成マスク PNG の URL.
export const fisheyeMaskUrl = (id: string, index: number, lens: number) =>
  `/api/projects/${id}/fisheye-mask/${index}?lens=${lens}`
export const denoisedImageUrl = (id: string, index: number, lens: number) =>
  `/api/projects/${id}/denoise/${index}/image?lens=${lens}`
export const pinholeImageUrl = (id: string, index: number, view: string, lens: number, mask = false) =>
  `/api/projects/${id}/pinhole/${index}/${encodeURIComponent(view)}?lens=${lens}${mask ? '&mask=true' : ''}`

export type ParsedImageName = {
  kind: 'native' | 'pinhole' | 'erp'
  view: string
  lens: number
  index: number
}

// COLMAP image name を入力 workspace の 3 layout に分解する.
export const parseImageName = (name: string): ParsedImageName | null => {
  const native = name.match(/^(front|back)\/frame_(\d+)\.(?:jpg|jpeg|png)$/i)
  if (native) {
    return { kind: 'native', view: native[1], lens: native[1] === 'front' ? 0 : 1, index: Number(native[2]) }
  }
  const pinhole = name.match(/^(.+)_lens(\d+)\/frame_(\d+)\.(?:jpg|jpeg|png)$/i)
  if (pinhole) {
    return { kind: 'pinhole', view: pinhole[1], lens: Number(pinhole[2]), index: Number(pinhole[3]) }
  }
  const erp = name.match(/^frame_(\d+)\.(?:jpg|jpeg|png)$/i)
  if (erp) return { kind: 'erp', view: 'erp', lens: 0, index: Number(erp[1]) }
  return null
}

// 再構成の画像名から frame index を抽出し, frame ごとの登録情報 (使われた画像数 / 3D 点数) を集計する.
// 抽出済みだが Map に無い frame = 再構成で未登録 (失敗).
export const frameReconMap = (recon: ReconstructionData | undefined): Map<number, { numPoints: number; images: number }> => {
  const m = new Map<number, { numPoints: number; images: number }>()
  if (!recon) return m
  for (const img of recon.images) {
    const mt = img.name.match(/frame_(\d+)/)
    if (!mt) continue
    const idx = Number(mt[1])
    const cur = m.get(idx) ?? { numPoints: 0, images: 0 }
    cur.numPoints += img.num_points
    cur.images += 1
    m.set(idx, cur)
  }
  return m
}

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
  onStatus?: (status: 'disconnected' | 'reconnected') => void,
): { close: () => void } => {
  // バックエンド再起動 / 一時的な切断でも進捗が止まらないよう自動再接続する.
  // 受信した最大 id を覚え, 再接続時に since=lastId で続きから取り (取りこぼし / 重複なし).
  // 初回 since<0 は「今から」tail (過去ログを Console に流さない).
  let ws: WebSocket | null = null
  let closed = false
  let lastId = opts.since ?? -1
  let everOpen = false
  let downNotified = false
  let retry: ReturnType<typeof setTimeout> | undefined

  const connect = () => {
    if (closed) return
    const params = new URLSearchParams()
    params.set('since', String(lastId))
    if (opts.jobId) params.set('job_id', opts.jobId)
    if (opts.projectId) params.set('project_id', opts.projectId)
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    ws = new WebSocket(`${proto}//${window.location.host}/api/events?${params.toString()}`)
    ws.onopen = () => {
      if (everOpen && downNotified) onStatus?.('reconnected')
      everOpen = true
      downNotified = false
    }
    ws.onmessage = ev => {
      try {
        const raw = JSON.parse(ev.data) as Omit<EventEnvelope, 'msg_args'> & { msg_args: string | null }
        if (typeof raw.id === 'number' && raw.id > lastId) lastId = raw.id
        // バックエンドは msg_args を JSON 文字列で送るので, ここでオブジェクトへパースする.
        let args: Record<string, unknown> | null = null
        if (raw.msg_args) {
          try {
            args = JSON.parse(raw.msg_args) as Record<string, unknown>
          } catch {
            args = null
          }
        }
        onEvent({ ...raw, msg_args: args })
      } catch {
        // 壊れたメッセージは無視.
      }
    }
    ws.onclose = () => {
      if (closed) return
      if (everOpen && !downNotified) { onStatus?.('disconnected'); downNotified = true }
      clearTimeout(retry)
      retry = setTimeout(connect, 1000)
    }
    ws.onerror = () => { try { ws?.close() } catch { /* onclose が再接続を張る */ } }
  }

  connect()
  return {
    close: () => {
      closed = true
      clearTimeout(retry)
      try { ws?.close() } catch { /* noop */ }
    },
  }
}
