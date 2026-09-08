// バックエンドの API 型と fetch ヘルパ. Browser 固有の binary points parser もここに集約する.
import type { IJsonModel } from 'flexlayout-react'

export interface ViewerPreferences {
  showPoints: boolean
  showCams: boolean
  pointSize: number
  showGrid: boolean
  showCenter: boolean
  background: string | null
}

export interface ViewerCameraPose {
  position: [number, number, number]
  quaternion: [number, number, number, number]
}

export interface ConsolePreferences {
  info: boolean
  warn: boolean
  error: boolean
  debug: boolean
  search: string
}

export interface WorkspacePreferences {
  theme: 'auto' | 'light' | 'dark'
  lang: 'ja' | 'zh' | 'en' | null
  lastProjectId: string | null
  layout: IJsonModel | null
  compactLayout: IJsonModel | null
  viewer: ViewerPreferences
  console: ConsolePreferences
}

export type WorkspacePreferencesPatch = Partial<Omit<WorkspacePreferences, 'viewer' | 'console'>> & {
  viewer?: Partial<ViewerPreferences>
  console?: Partial<ConsolePreferences>
}

export type PipelineState =
  | 'created'
  | 'inspected'
  | 'extracted'
  | 'prepared'
  | 'rectified'
  | 'masked'
  | 'features_extracted'
  | 'matched'
  | 'reconstructed'
  | 'aligned'
  | 'scale_restored'
  | 'scene_aligned'
  | 'cleaned'
  | 'densified'
  | 'exported'

export type SourceRole = 'primary' | 'supplemental'
export type SourceAdapter = string
export type MediaKind = 'video' | 'images'
export type Projection = 'dual_fisheye' | 'equirectangular' | 'perspective'
export type MaskPurpose = 'feature' | 'training'

export interface ProjectSource {
  id: string
  label: string
  role: SourceRole
  adapter: SourceAdapter
  media_kind: MediaKind
  projection: Projection
  path: string
  ordinal: number
  enabled: boolean
}

export interface SourceCreate {
  label?: string
  role?: SourceRole
  adapter: SourceAdapter
  media_kind: MediaKind
  projection: Projection
  path: string
}

export interface ProjectUiState {
  params?: Record<string, unknown>
  reconMode?: string
  clearOutputStages?: string[]
  selectedStage?: string | null
  selectedCameraId?: number | null
  selectedFrameIndex?: number | null
  cameraPose?: ViewerCameraPose | null
}

export interface Project {
  id: string
  name: string
  created_at: string
  updated_at: string
  sources: ProjectSource[]
  state: PipelineState
  // 工程に保存された UI 設定 (step パラメータ / モード). リロードで復元する.
  ui_state: ProjectUiState | null
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
  metric_scale?: { metric?: boolean; scale_factor?: number }
  scene_alignment?: {
    applied?: boolean
    ground?: { applied?: boolean; ground_y?: number }
    orientation?: { applied?: boolean; yaw_deg?: number }
  }
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
  getPreferences: () => req<WorkspacePreferences>('/api/preferences'),
  patchPreferences: (patch: WorkspacePreferencesPatch, keepalive = false) =>
    req<WorkspacePreferences>('/api/preferences', {
      method: 'PATCH', headers: jsonHeaders, body: JSON.stringify(patch), keepalive,
    }),
  listProjects: () => req<Project[]>('/api/projects'),
  getProject: (id: string) => req<Project>(`/api/projects/${id}`),
  createProject: (name: string) =>
    req<Project>('/api/projects', {
      method: 'POST',
      headers: jsonHeaders,
      body: JSON.stringify({ name }),
    }),
  addSource: (id: string, source: SourceCreate) =>
    req<Project>(`/api/projects/${id}/sources`, {
      method: 'POST',
      headers: jsonHeaders,
      body: JSON.stringify(source),
    }),
  deleteSource: (id: string, sourceId: string) =>
    req<Project>(`/api/projects/${id}/sources/${sourceId}`, { method: 'DELETE' }),
  makePrimarySource: (id: string, sourceId: string) =>
    req<Project>(`/api/projects/${id}/sources/${sourceId}/make-primary`, { method: 'POST' }),
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
  getMasks: (id: string, purpose: MaskPurpose) =>
    req<MasksManifest>(`/api/projects/${id}/masks/${purpose}`),
  getExportInfo: (id: string) => req<ExportInfo>(`/api/projects/${id}/export-info`),
  patchUiState: (id: string, ui: ProjectUiState, keepalive = false) =>
    req<Project>(`/api/projects/${id}/ui-state`, {
      method: 'PATCH', headers: jsonHeaders, body: JSON.stringify({ ui }), keepalive,
    }),
  getSourceRegion: (id: string, sourceId: string) =>
    req<SourceRegion>(`/api/projects/${id}/source-region?source_id=${encodeURIComponent(sourceId)}`),
  putSourceRegion: (id: string, sourceId: string, region: SourceRegion) =>
    req<SourceRegion>(`/api/projects/${id}/source-region?source_id=${encodeURIComponent(sourceId)}`, {
      method: 'PUT',
      headers: jsonHeaders,
      body: JSON.stringify({ views: region.views }),
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
  clearOutputs: (id: string, stages: string[]) =>
    req<{ requested: string[]; cleared: string[]; state: string }>(`/api/projects/${id}/clear-outputs`, {
      method: 'POST',
      headers: jsonHeaders,
      body: JSON.stringify({ stages }),
    }),
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
  duration_sec: number | null
  duration_sec_total?: number
  fps?: number | null
  width?: number | null
  height?: number | null
  nb_frames?: number | null
  sources?: Array<{
    id: string
    duration_sec: number | null
    fps?: number | null
    width?: number | null
    height?: number | null
    nb_frames?: number | null
  }>
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
  active_params: Record<string, unknown> | null
  extra: Record<string, unknown> | null
  activity_event: EventEnvelope | null
  progress_event: EventEnvelope | null
}
export interface StagesStatus {
  project_id: string
  state: PipelineState
  stages: StageStatus[]
}

export type RegionView = {
  kind: 'circle'
  cx: 0.5
  cy: 0.5
  r: number
  operations: RegionOperation[]
} | { kind: 'full'; operations: RegionOperation[] }
export interface RegionOperation {
  mode: 'add' | 'subtract'
  x: number
  y: number
  r: number
  stroke_id: number
}
export interface SourceRegion {
  views: Record<string, RegionView>
  saved?: boolean
  needs_review?: boolean
  invalidated?: string[]
}

export interface FrameSelection {
  mode: string
  selected: number
  candidates?: number
  fallback?: boolean
  reasons?: { blur: number; exposure: number; few_features: number; rolling_shutter?: number }
  continuity_strategy?: 'quality' | 'parallax' | 'balanced'
  maximum_gap_sec?: number
  bridge_frames?: number
  bridge_relaxed_sharpness?: number
  bridge_relaxed_rolling_shutter?: number
  unresolved_gaps?: number
}

export interface FrameInfo {
  index: number
  source_id: string
  source_index: number
  timestamp_sec: number | null
  score?: { sharpness: number; features?: number } | null
}

export interface FramesManifest {
  count: number
  sources: Array<{
    id: string
    label: string
    role: SourceRole
    projection: Projection
    kind: string
    count: number
    width: number | null
    height: number | null
    fps: number | null
    selection: FrameSelection | null
  }>
  frames: FrameInfo[]
}

// Feature / training mask Step が個別に生成する manifest.
export interface MaskImageRecord {
  name: string
  source_id: string
  capture_index: number
  path: string
  coverage: number
  coverage_warning: boolean
}
export interface MasksManifest {
  version: 3
  purpose: MaskPurpose
  revision: string
  complete: boolean
  total_images: number
  generated_images: number
  prompt: string[]
  max_inference_size: number
  dilate_px: number
  images: MaskImageRecord[]
}

// export_dataset の出力ディレクトリ (絶対パス).
export interface ExportInfo {
  dir: string
  gui_integration: {
    train_configs_auto_applied: boolean
    warnings: string[]
    required_settings: {
      strategy: string
      gut: boolean
      undistort: boolean
      mask_mode: string
      ppisp: boolean
      ppisp_controller: boolean
      max_width: number
    }
  } | null
}

// 抽出フレーム (fisheye) の URL.
export const frameImageUrl = (id: string, index: number, lens: number) =>
  `/api/projects/${id}/frames/${index}/image?lens=${lens}`
// 用途別 mask PNG の URL.
export const preparedImageUrl = (id: string, name: string) =>
  `/api/projects/${id}/prepared-image?name=${encodeURIComponent(name)}`
export const preparedMaskUrl = (id: string, name: string, purpose: MaskPurpose, revision?: string) =>
  `/api/projects/${id}/prepared-mask?name=${encodeURIComponent(name)}&purpose=${purpose}`
  + (revision ? `&revision=${encodeURIComponent(revision)}` : '')

export type ParsedImageName = {
  kind: 'native' | 'pinhole' | 'erp' | 'perspective'
  sourceId: string
  view: string
  lens: number
  index: number
}

// COLMAP image name を入力 workspace の 3 layout に分解する.
export const parseImageName = (name: string): ParsedImageName | null => {
  const native = name.match(/^sources\/([^/]+)\/(lens0|lens1)\/frame_(\d+)\.(?:jpg|jpeg|png)$/i)
  if (native) {
    return { kind: 'native', sourceId: native[1], view: native[2], lens: native[2] === 'lens0' ? 0 : 1, index: Number(native[3]) }
  }
  const pinhole = name.match(/^sources\/([^/]+)\/(.+)_lens(\d+)\/frame_(\d+)\.(?:jpg|jpeg|png)$/i)
  if (pinhole) {
    return { kind: 'pinhole', sourceId: pinhole[1], view: pinhole[2], lens: Number(pinhole[3]), index: Number(pinhole[4]) }
  }
  const perspective = name.match(/^sources\/([^/]+)\/camera_\d+\/frame_(\d+)\.(?:jpg|jpeg|png)$/i)
  if (perspective) return { kind: 'perspective', sourceId: perspective[1], view: 'main', lens: 0, index: Number(perspective[2]) }
  const erp = name.match(/^sources\/([^/]+)\/frame_(\d+)\.(?:jpg|jpeg|png)$/i)
  if (erp) return { kind: 'erp', sourceId: erp[1], view: 'erp', lens: 0, index: Number(erp[2]) }
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
export const fetchPoints = async (projectId: string, signal?: AbortSignal): Promise<ParsedPoints> => {
  const res = await fetch(`/api/projects/${projectId}/reconstruction/points`, { signal, cache: 'no-store' })
  if (!res.ok) throw new Error(`points fetch failed: ${res.status}`)
  return parsePoints(await res.arrayBuffer())
}

const srgbToLinear = (byte: number): number => {
  const value = byte / 255
  return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4
}
const POINT_COLORS = Float32Array.from({ length: 256 }, (_, byte) => srgbToLinear(byte))

export const parsePoints = (buf: ArrayBuffer): ParsedPoints => {
  if (buf.byteLength < 8) throw new Error('Invalid point cloud header')
  const dv = new DataView(buf)
  const count = dv.getUint32(0, true)
  const stride = dv.getUint32(4, true)
  if (stride !== 20 || buf.byteLength !== 8 + count * stride)
    throw new Error('Point cloud length or stride mismatch')
  const positions = new Float32Array(count * 3)
  const colors = new Float32Array(count * 3)
  let off = 8
  for (let i = 0; i < count; i++) {
    positions[i * 3] = dv.getFloat32(off, true)
    positions[i * 3 + 1] = dv.getFloat32(off + 4, true)
    positions[i * 3 + 2] = dv.getFloat32(off + 8, true)
    if (![positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]].every(Number.isFinite))
      throw new Error(`Invalid point cloud position at index ${i}`)
    colors[i * 3] = POINT_COLORS[dv.getUint8(off + 12)]
    colors[i * 3 + 1] = POINT_COLORS[dv.getUint8(off + 13)]
    colors[i * 3 + 2] = POINT_COLORS[dv.getUint8(off + 14)]
    off += stride
  }
  return { count, positions, colors }
}

// WebSocket ヘルパ. dev では /api/events が Vite proxy 経由で ws:// にアップグレードされる.
export const openEventStream = (
  opts: { jobId?: string; projectId?: string; since?: number },
  onEvent: (e: EventEnvelope) => void,
  onStatus?: (status: 'disconnected' | 'reconnected' | 'invalid-event') => void,
): { close: () => void } => {
  // バックエンド再起動 / 一時的な切断でも進捗が止まらないよう自動再接続する.
  // 受信した最大 id を覚え, 再接続時に since=lastId で続きから取り (取りこぼし / 重複なし).
  // 初回 since<0 は「今から」tail (過去ログを Console に流さない).
  let ws: WebSocket | null = null
  let closed = false
  let lastId = opts.since ?? -1
  let downNotified = false
  let invalidNotified = false
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
      if (downNotified) onStatus?.('reconnected')
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
            if (!invalidNotified) onStatus?.('invalid-event')
            invalidNotified = true
            return
          }
        }
        invalidNotified = false
        onEvent({ ...raw, msg_args: args })
      } catch {
        if (!invalidNotified) onStatus?.('invalid-event')
        invalidNotified = true
      }
    }
    ws.onclose = () => {
      if (closed) return
      if (!downNotified) { onStatus?.('disconnected'); downNotified = true }
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
