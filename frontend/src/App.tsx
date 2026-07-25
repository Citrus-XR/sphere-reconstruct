import { useEffect, useMemo, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Layout, Model, type TabNode, type IJsonModel } from 'flexlayout-react'
import { api, openEventStream, type EventEnvelope } from './api/client'
import { SystemStatsBar } from './components/SystemStatsBar'
import { SettingsMenu } from './components/SettingsMenu'
import { FileBrowser } from './components/FileBrowser'
import { ProjectManager } from './components/ProjectManager'
import { ErrorBoundary } from './components/ErrorBoundary'
import { StageHierarchy, type HierItem } from './components/StageHierarchy'
import { SceneHierarchy } from './components/SceneHierarchy'
import { StageSettings } from './features/StageSettings'
import { CameraInspector } from './features/CameraInspector'
import { FrameInspector } from './features/FrameInspector'
import { FisheyeRegionEditor } from './features/FisheyeRegionEditor'
import { PointCloudViewer } from './viewers/PointCloudViewer'
import { DEFAULT_PARAMS, allParams, type ReconMode, type StageParams } from './features/stageParams'
import { useSettings } from './ui/settings'
import { translateMsg } from './ui/i18n'

const OPTIONAL = new Set(['generate_masks'])
type Lvl = 'info' | 'warn' | 'error' | 'debug'
const LVL_EMOJI: Record<string, string> = { info: 'ℹ️', warn: '⚠️', error: '⛔', debug: '🔍' }
const LAYOUT_KEY = 'layout.flex.v2'

// ログ行の時刻 (ローカル HH:MM:SS) と stage タグ (snake_case → PascalCase: extract_frames → ExtractFrames).
const fmtTime = (ts: string): string => {
  const d = new Date(ts)
  if (isNaN(d.getTime())) return ''
  const p = (n: number) => String(n).padStart(2, '0')
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
}
const stageTag = (s: string | null): string =>
  s ? s.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join('') : ''

// Unity/VSCode 風のドッキング初期レイアウト. タブのタイトルをドラッグして再配置でき,
// 変更は localStorage に保存される. Console は既定でシーンビューの下.
const DEFAULT_LAYOUT: IJsonModel = {
  global: { tabEnableClose: false, tabEnableRename: false, tabSetEnableMaximize: true },
  borders: [],
  layout: {
    type: 'row',
    children: [
      { type: 'tabset', weight: 16, children: [{ type: 'tab', name: 'Hierarchy', component: 'sceneHier' }] },
      {
        type: 'row', weight: 56, children: [
          { type: 'tabset', weight: 72, children: [{ type: 'tab', name: 'Scene View', component: 'scene' }] },
          { type: 'tabset', weight: 28, children: [{ type: 'tab', name: 'Console', component: 'console' }] },
        ],
      },
      {
        type: 'row', weight: 28, children: [
          { type: 'tabset', weight: 42, children: [{ type: 'tab', name: 'Steps', component: 'steps' }] },
          { type: 'tabset', weight: 58, children: [{ type: 'tab', name: 'Inspector', component: 'inspector' }] },
        ],
      },
    ],
  },
}

export const App = () => {
  const { t, lang } = useSettings()
  const qc = useQueryClient()
  const { data: projects } = useQuery({ queryKey: ['projects'], queryFn: api.listProjects, refetchInterval: 3000 })
  const { data: settingsData } = useQuery({ queryKey: ['settings'], queryFn: api.getSettings, retry: false })
  const [projectId, setProjectId] = useState<string | null>(null)
  const [selectedStage, setSelectedStage] = useState<string>('extract_frames')
  const [reconMode, setReconMode] = useState<ReconMode>('native_fisheye')
  const [params, setParamsState] = useState<StageParams>(DEFAULT_PARAMS)
  const setParams = (patch: Partial<StageParams>) => setParamsState(prev => ({ ...prev, ...patch }))
  const [disabled, setDisabled] = useState<Set<string>>(new Set())
  const [browsing, setBrowsing] = useState(false)
  const [managerOpen, setManagerOpen] = useState(false)
  const [activeJobId, setActiveJobId] = useState<string | null>(null)
  const [events, setEvents] = useState<EventEnvelope[]>([])
  // stage 別の最新「進捗を持つイベント」. 環形の値 (.progress) と現在処理中の項目 (message) 両方に使う.
  const [progressByStage, setProgressByStage] = useState<Record<string, EventEnvelope>>({})
  const [showPoints, setShowPoints] = useState(true)
  const [showCams, setShowCams] = useState(true)
  const [selectedCameraId, setSelectedCameraId] = useState<number | null>(null)
  const [selectedFrameIndex, setSelectedFrameIndex] = useState<number | null>(null)
  const [lvlOn, setLvlOn] = useState<Record<Lvl, boolean>>({ info: true, warn: true, error: true, debug: false })
  const [search, setSearch] = useState('')
  // Console 自動スクロール: ユーザーが最下部にいる時だけ新ログで追従する.
  const conRef = useRef<HTMLDivElement>(null)
  const atBottomRef = useRef(true)

  // ドッキングモデル (localStorage 保存).
  const model = useMemo(() => {
    try {
      const saved = localStorage.getItem(LAYOUT_KEY)
      const json = saved ? JSON.parse(saved) : DEFAULT_LAYOUT
      // 旧レイアウト移行: 'sceneHier' タブ名を Hierarchy に統一 (Scene View と紛らわしいため).
      const walk = (n: { component?: string; name?: string; children?: unknown[] }) => {
        if (n?.component === 'sceneHier') n.name = 'Hierarchy'
        ;(n?.children as typeof n[] | undefined)?.forEach(walk)
      }
      walk(json.layout)
      return Model.fromJson(json)
    } catch { return Model.fromJson(DEFAULT_LAYOUT) }
  }, [])

  const promptInit = useRef(false)
  useEffect(() => {
    if (promptInit.current || !settingsData) return
    const dp = (settingsData as { sam3?: { default_prompt?: string } })?.sam3?.default_prompt
    // 保存済み prompt があればそれを尊重し, 空のときだけ既定を入れる.
    if (dp) setParamsState(prev => prev.prompt ? prev : { ...prev, prompt: dp })
    promptInit.current = true
  }, [settingsData]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { if (!projectId && projects?.length) setProjectId(projects[0].id) }, [projects, projectId])
  const project = projects?.find(p => p.id === projectId) ?? null
  const isFisheye = reconMode === 'native_fisheye' && project?.source_kind === 'insv'
  // ソース種別で無効なモードだけ既定へ戻す (ERP は equirect/pinhole, INSV は native/pinhole が有効).
  useEffect(() => {
    const k = project?.source_kind
    if (!k) return
    if (k !== 'insv' && reconMode === 'native_fisheye') setReconMode('equirectangular')
    if (k === 'insv' && reconMode === 'equirectangular') setReconMode('native_fisheye')
  }, [project?.source_kind]) // eslint-disable-line react-hooks/exhaustive-deps

  // 工程に保存された UI 設定を復元 (工程ごと 1 回). 復元後の変更は debounce して保存する.
  const hydratedRef = useRef<string | null>(null)
  useEffect(() => {
    if (!project || hydratedRef.current === project.id) return
    hydratedRef.current = project.id
    const ui = project.ui_state
    if (ui?.params) setParamsState(prev => ({ ...prev, ...(ui.params as Partial<StageParams>) }))
    if (ui?.reconMode) setReconMode(ui.reconMode as ReconMode)
    if (Array.isArray(ui?.disabled)) setDisabled(new Set(ui.disabled))
  }, [project])

  const saveTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  useEffect(() => {
    if (!projectId || hydratedRef.current !== projectId) return  // 復元前は保存しない (既定で上書きしない).
    clearTimeout(saveTimer.current)
    saveTimer.current = setTimeout(() => {
      api.putUiState(projectId, { params, reconMode, disabled: [...disabled] })
        .catch(e => console.warn('ui-state save failed', e))
    }, 600)
    return () => clearTimeout(saveTimer.current)
  }, [params, reconMode, disabled, projectId])

  const { data: stagesData } = useQuery({
    queryKey: ['stages', projectId], queryFn: () => api.getStages(projectId as string),
    enabled: !!projectId, refetchInterval: 2000, retry: false,
  })
  const { data: sourceInfo } = useQuery({
    queryKey: ['source-info', projectId], queryFn: () => api.getSourceInfo(projectId as string),
    enabled: !!projectId && !!project?.source_path, retry: false,
  })
  const { data: recon } = useQuery({
    queryKey: ['reconstruction', projectId], queryFn: () => api.getReconstruction(projectId as string),
    enabled: !!projectId, retry: false,
  })
  const { data: framesData } = useQuery({
    queryKey: ['frames', projectId], queryFn: () => api.getFrames(projectId as string),
    enabled: !!projectId, retry: false,
  })
  const { data: regionData } = useQuery({
    queryKey: ['fisheye-region', projectId], queryFn: () => api.getFisheyeRegion(projectId as string),
    enabled: !!projectId && isFisheye, retry: false,
  })
  const firstFrame = framesData?.frames?.[0]?.index ?? null
  // 一键全流程が通らない構成を検出 (通らないなら理由を返し, ボタンを無効化する).
  // 魚眼有効領域は既定円で動くため必須ではない (未保存でも run-all は通る).
  const runReason: string | null =
    !project?.source_path ? t('noSource')
    : (project.source_kind !== 'insv' && reconMode === 'native_fisheye') ? t('modeMismatch')
    : (project.source_kind === 'insv' && reconMode === 'equirectangular') ? t('modeMismatch')
    : null

  // 実行中ジョブの追跡 (停止ボタン用).
  const { data: jobData } = useQuery({
    queryKey: ['job', activeJobId], queryFn: () => api.getJob(activeJobId as string),
    enabled: !!activeJobId, refetchInterval: 1200, retry: false,
  })
  useEffect(() => {
    if (jobData && ['succeeded', 'failed', 'cancelled'].includes(jobData.status)) {
      setActiveJobId(null)
      // ジョブ完了で成果物が変わるため, 依存クエリを更新 (写真リスト/再構成/魚眼領域).
      for (const key of ['frames', 'reconstruction', 'fisheye-region', 'masks', 'export-info', 'stages']) {
        qc.invalidateQueries({ queryKey: [key, projectId] })
      }
    }
  }, [jobData]) // eslint-disable-line react-hooks/exhaustive-deps
  const anyStageRunning = !!stagesData?.stages.some(s => s.status === 'running')
  const jobRunning = jobData?.status === 'running' || jobData?.status === 'queued'
  const processing = jobRunning || anyStageRunning
  // 停止対象: UI が開始した job を優先, 無ければ実行中ステージの job_id (別セッション/外部起動でも止められる).
  const runningJobId = stagesData?.stages.find(s => s.status === 'running')?.job_id ?? null
  const stopTarget = activeJobId ?? runningJobId
  const stopJob = () => { if (stopTarget) api.cancelJob(stopTarget).then(() => qc.invalidateQueries({ queryKey: ['stages', projectId] })) }

  const synthId = useRef(-1)
  useEffect(() => {
    if (!projectId) return
    setEvents([]); setProgressByStage({})
    const ws = openEventStream({ projectId, since: -1 }, e => {
      // progress を持つイベントは環形インジケータ用に stage 別で保持. kind==='progress' は
      // Console には出さず (spam 防止), それ以外 (開始/完了/失敗/警告) だけログに積む.
      if (e.progress != null && e.stage) setProgressByStage(prev => ({ ...prev, [e.stage as string]: e }))
      if (e.kind !== 'progress') setEvents(prev => [...prev.slice(-500), e])
    }, status => {
      // 接続断 / 再接続を Console に 1 行ずつ出す (クライアント合成イベント).
      const key = status === 'disconnected' ? 'log.ws_disconnected' : 'log.ws_reconnected'
      setEvents(prev => [...prev.slice(-500), {
        id: synthId.current--, job_id: null, project_id: projectId, stage: null,
        level: status === 'disconnected' ? 'warn' : 'info', message: key,
        msg_key: key, msg_args: null, progress: null, kind: 'log', ts: new Date().toISOString(),
      }])
    })
    return () => ws.close()
  }, [projectId])

  const setSource = useMutation({
    mutationFn: (path: string) =>
      api.setSource(projectId as string, path.toLowerCase().endsWith('.insv') ? 'insv'
        : /\.(mp4|mov|mkv|avi)$/i.test(path) ? 'erp_video' : 'erp_images', path),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['projects'] }); qc.invalidateQueries({ queryKey: ['source-info', projectId] }) },
  })
  const runAll = useMutation({
    mutationFn: () => api.runPipeline(projectId as string, allParams(params, reconMode), [...disabled]),
    onSuccess: r => { setActiveJobId(r.job_id); qc.invalidateQueries({ queryKey: ['stages', projectId] }) },
  })
  const clearOutputs = useMutation({
    mutationFn: () => api.clearOutputs(projectId as string),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['stages', projectId] })
      // frames/reconstruction/fisheye-region は成果物が消えると 404 になり, react-query は
      // エラー時に前回 data を保持する (= 残像). invalidate では消えないため remove で破棄する.
      for (const key of ['reconstruction', 'frames', 'fisheye-region', 'masks', 'export-info']) {
        qc.removeQueries({ queryKey: [key, projectId] })
      }
      setSelectedFrameIndex(null); setSelectedCameraId(null)
      setProgressByStage({})
    },
  })

  const onJob = (jobId: string) => { setActiveJobId(jobId); qc.invalidateQueries({ queryKey: ['stages', projectId] }) }
  const toggleStage = (key: string) =>
    setDisabled(prev => { const n = new Set(prev); n.has(key) ? n.delete(key) : n.add(key); return n })
  const selectStep = (stage: string) => { setSelectedStage(stage); setSelectedCameraId(null); setSelectedFrameIndex(null) }
  const selectCamera = (id: number) => { setSelectedCameraId(id); setSelectedStage(''); setSelectedFrameIndex(null) }
  const selectFrame = (index: number) => { setSelectedFrameIndex(index); setSelectedStage(''); setSelectedCameraId(null) }
  const selectedCamImage = selectedCameraId != null ? recon?.images.find(i => i.id === selectedCameraId) : undefined
  // 現在の再構成結果のカメラモデルからモード判定. EQUIRECTANGULAR を先に見る (FISHEYE を含まない).
  const resultMode: 'native' | 'pinhole' | 'equirect' | null = recon?.cameras?.length
    ? (recon.cameras[0].model.includes('EQUIRECTANGULAR') ? 'equirect'
      : recon.cameras[0].model.includes('FISHEYE') ? 'native'
      : recon.cameras[0].model.includes('PINHOLE') ? 'pinhole' : null)
    : null

  const items: HierItem[] = []
  const nonInsvReady = !!project?.source_path && project?.source_kind !== 'insv'
  for (const s of stagesData?.stages ?? []) {
    // pinhole 再投影 (reproject_views) は pinhole_rig モードだけ必要. 他は「生成」不要で隠す.
    if (s.stage === 'reproject_views' && reconMode !== 'pinhole_rig') continue
    const en = !disabled.has(s.stage)
    // mp4 / ERP 画像は検査不要だが, 「緑扱い」にするのは equirectangular (直接処理) のときだけ.
    // ERP+pinhole は reproject 処理が要るので inspect_source を通常表示にする.
    const forceGreen = s.stage === 'inspect_source' && nonInsvReady && reconMode === 'equirectangular'
    const done = s.has_output || forceGreen
    items.push({
      key: s.stage, label: t(`st_${s.stage}`),
      badgeColor: s.status === 'failed' ? 'var(--error)' : done ? '#4caf50' : 'var(--border)',
      statusLabel: s.status === 'running' ? t('running') : s.status === 'failed' ? t('failed') : done ? t('done') : t('notrun'),
      toggleable: OPTIONAL.has(s.stage), enabled: en, dim: !en, running: s.status === 'running',
      progress: progressByStage[s.stage]?.progress ?? 0,
    })
    // 魚眼有効領域は魚眼ソース (native 魚眼 + INSV) を選んだ時点で pipeline に出す.
    // 抽出前は灰色 pending, クリックすると「先に抽出」ヒントを出す (空プレビューにしない).
    // 既定円で動くため警告色にはしない.
    if (s.stage === 'extract_frames' && isFisheye) {
      items.push({
        key: 'fisheye_region', label: t('st_fisheye_region'),
        badgeColor: regionData?.saved ? '#4caf50' : 'var(--border)',
        statusLabel: regionData?.saved ? t('done') : s.has_output ? t('regionDefault') : t('notrun'),
        toggleable: false, enabled: true,
      })
    }
  }

  const stageStatus = stagesData?.stages.find(s => s.stage === selectedStage)
  // KEY+ARGS を持つイベントは表示言語で翻訳し, 未 key 行は message フォールバックを使う.
  const renderMsg = (e: EventEnvelope) => (e.msg_key ? translateMsg(lang, e.msg_key, e.msg_args) : e.message)
  const lastEv = events.length ? events[events.length - 1] : null
  const lastLog = lastEv ? `[${fmtTime(lastEv.ts)}] ${LVL_EMOJI[lastEv.level] ?? ''} ${renderMsg(lastEv)}` : ''
  const filtered = events.filter(e => lvlOn[e.level as Lvl] && (!search || renderMsg(e).toLowerCase().includes(search.toLowerCase())))

  // 新ログ到着時, 直前まで最下部にいた場合のみ追従スクロール (途中を読んでいる時は動かさない).
  useEffect(() => {
    const el = conRef.current
    if (el && atBottomRef.current) el.scrollTop = el.scrollHeight
  }, [filtered.length])

  const factory = (node: TabNode) => {
    const comp = node.getComponent()
    const content = (() => {
      switch (comp) {
      case 'sceneHier':
        return (
          <div className="dock-content nopad">
            <SceneHierarchy recon={recon} frames={framesData?.frames} showPoints={showPoints} setShowPoints={setShowPoints}
              showCams={showCams} setShowCams={setShowCams} selectedCameraId={selectedCameraId} onSelectCamera={selectCamera}
              selectedFrameIndex={selectedFrameIndex} onSelectFrame={selectFrame} />
          </div>
        )
      case 'scene':
        return (
          <div style={{ position: 'relative', width: '100%', height: '100%', background: 'var(--bg)' }}>
            {recon
              ? <PointCloudViewer projectId={projectId as string} recon={recon}
                  showPoints={showPoints} showCams={showCams} selectedCameraId={selectedCameraId} onPickCamera={selectCamera} />
              : null}
            {recon && (
              <div className="scene-info">
                images {recon.stats.num_images} · points {recon.stats.num_points3D.toLocaleString()}
                {recon.stats.registered_ratio != null && ` · reg ${(recon.stats.registered_ratio * 100).toFixed(0)}%`}
              </div>
            )}
          </div>
        )
      case 'steps':
        return (
          <div className="dock-content nopad">
            <StageHierarchy items={items} selected={selectedCamImage ? '' : selectedStage} onSelect={selectStep} />
          </div>
        )
      case 'inspector':
        return (
          <div className="dock-content">
            {selectedCamImage
              ? <CameraInspector projectId={projectId as string} image={selectedCamImage} />
              : selectedFrameIndex != null
              ? <FrameInspector projectId={projectId as string} frameIndex={selectedFrameIndex}
                  frames={framesData?.frames} recon={recon} sourceKind={project?.source_kind ?? null} />
              : selectedStage === 'fisheye_region'
              ? (firstFrame !== null
                  ? <FisheyeRegionEditor projectId={projectId as string} frameIndex={firstFrame}
                      onSaved={() => qc.invalidateQueries({ queryKey: ['fisheye-region', projectId] })} />
                  : <div className="hint">{t('needExtractFirst')}</div>)
              : selectedStage
              ? <StageSettings projectId={projectId as string} stage={selectedStage} status={stageStatus}
                  sourceInfo={sourceInfo} reconMode={reconMode} setReconMode={setReconMode} params={params} setParams={setParams}
                  onJob={onJob} hasSource={!!project?.source_path} sourcePath={project?.source_path ?? null} resultMode={resultMode}
                  sourceKind={project?.source_kind ?? null} processing={processing} stageIsRunning={stageStatus?.status === 'running'} onStop={stopJob}
                  stageDisabled={disabled.has(selectedStage)} onToggleStage={() => toggleStage(selectedStage)}
                  onSelectSource={() => setBrowsing(true)} frameSelection={framesData?.selection}
                  stageProgress={progressByStage[selectedStage]?.progress ?? 0}
                  stageStartedAt={stageStatus?.started_at ?? null}
                  stageProgressMsg={progressByStage[selectedStage] ? renderMsg(progressByStage[selectedStage]) : ''} />
              : <div className="hint">—</div>}
          </div>
        )
      case 'console':
        return (
          <div style={{ height: '100%', display: 'flex', flexDirection: 'column', minWidth: 0 }}>
            <div className="con-bar">
              {(['info', 'warn', 'error'] as Lvl[]).map(l => (
                <button key={l} className={`con-chip${lvlOn[l] ? ' on' : ''}`} title={l}
                  onClick={() => setLvlOn(p => ({ ...p, [l]: !p[l] }))}>{LVL_EMOJI[l]}</button>
              ))}
              <input className="con-search" placeholder={t('search')} value={search} onChange={e => setSearch(e.target.value)} />
              <button className="con-chip" onClick={() => setEvents([])}>{t('clearLog')}</button>
            </div>
            <div className="mono con-log" ref={conRef}
              onScroll={e => { const el = e.currentTarget; atBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24 }}>
              {filtered.map(e => (
                <div key={e.id}>
                  <span style={{ color: 'var(--fg-mute)' }}>[{fmtTime(e.ts)}]{e.stage ? `[${stageTag(e.stage)}]` : ''}</span>{' '}
                  <span title={e.level}>{LVL_EMOJI[e.level] ?? e.level}</span>{' '}
                  {renderMsg(e)}{e.progress != null ? ` (${Math.round(e.progress * 100)}%)` : ''}
                </div>
              ))}
            </div>
          </div>
        )
      default:
        return null
      }
    })()
    return <ErrorBoundary key={`${comp}:${projectId}`} label={comp}>{content}</ErrorBoundary>
  }

  return (
    <div className="ide">
      <div className="ide-top">
        <button className="btn" onClick={() => setManagerOpen(true)}>☰ {t('projects')}</button>
        <h1 style={{ margin: '0 4px' }}>{project?.name ?? 'sphere-reconstruct'}</h1>
        <span className="mono" style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {project?.source_path ?? t('noSource')}
        </span>
        <button className="btn btn-secondary" disabled={!projectId || clearOutputs.isPending || processing}
          onClick={() => { if (window.confirm(t('clearOutputsConfirm'))) clearOutputs.mutate() }}>{t('clearOutputs')}</button>
        {processing
          ? <button className="btn stop" onClick={stopJob}>■ {t('stop')}</button>
          : <button className="btn" disabled={!projectId || !!runReason || runAll.isPending}
              title={runReason ?? ''} onClick={() => runAll.mutate()}>{t('runAll')}</button>}
        <SettingsMenu />
      </div>

      <div className="ide-center">
        {projectId
          ? <ErrorBoundary key={`layout:${projectId}`} label="layout">
              <Layout model={model} factory={factory} onModelChange={m => localStorage.setItem(LAYOUT_KEY, JSON.stringify(m.toJson()))} />
            </ErrorBoundary>
          : <div style={{ padding: 24, color: 'var(--fg-mute)' }}>{t('noProject')}</div>}
      </div>

      <div className="ide-status">
        <span className="laststatus">{lastLog}</span>
        <SystemStatsBar />
      </div>

      {browsing && projectId && (
        <FileBrowser onClose={() => setBrowsing(false)} onPick={(path) => { setSource.mutate(path); setBrowsing(false) }} />
      )}
      {managerOpen && (
        <ProjectManager projects={projects} currentId={projectId} onSelect={setProjectId} onClose={() => setManagerOpen(false)} />
      )}
    </div>
  )
}
