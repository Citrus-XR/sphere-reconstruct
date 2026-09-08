import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState, type ComponentProps } from 'react'
import { useMutation, useQueries, useQuery, useQueryClient } from '@tanstack/react-query'
import { Layout, TabNode } from 'flexlayout-react'
import { api, openEventStream, type EventEnvelope, type SourceCreate, type ProjectUiState, type ViewerCameraPose } from './api/client'
import { SystemStatsBar } from './components/SystemStatsBar'
import { SettingsMenu } from './components/SettingsMenu'
import { FileBrowser } from './components/FileBrowser'
import { ProjectManager } from './components/ProjectManager'
import { CLEARABLE_STAGE_NAMES, ClearOutputsDialog } from './components/ClearOutputsDialog'
import { ErrorBoundary } from './components/ErrorBoundary'
import { StageHierarchy, type HierItem } from './components/StageHierarchy'
import { SceneHierarchy } from './components/SceneHierarchy'
import { PathText } from './components/PathText'
import { AppIcon, type AppIconName } from './components/AppIcon'
import { StageSettings } from './features/StageSettings'
import { CameraInspector } from './features/CameraInspector'
import { FrameInspector } from './features/FrameInspector'
import { SourceRegionEditor } from './features/SourceRegionEditor'
import {
  DEFAULT_PARAMS,
  defaultReconModeForProjection,
  paramsForStage,
  reconModesForProjection,
  type ReconMode,
  type StageParams,
} from './features/stageParams'
import { recommendSourceResolutions } from './features/resolutionPolicy'
import { useSettings } from './ui/settings'
import { usePersistence } from './ui/usePersistence'
import { useDocking, useTabVisibility } from './ui/docking'
import { translateMsg } from './ui/i18n'

type Lvl = 'info' | 'warn' | 'error' | 'debug'
const LVL_ICON: Record<Lvl, AppIconName> = {
  info: 'info', warn: 'warning', error: 'error', debug: 'search',
}
const PointCloudViewer = lazy(() => import('./viewers/PointCloudViewer').then(module => ({
  default: module.PointCloudViewer,
})))
const SceneView = ({ node, ...props }: Omit<ComponentProps<typeof PointCloudViewer>, 'active'> & { node: TabNode }) => {
  const active = useTabVisibility(node)
  return <PointCloudViewer {...props} active={active} />
}
const TabTitle = ({ id, revision, title }: { id: string; revision: number; title: string }) => {
  const ref = useRef<HTMLSpanElement>(null)
  useEffect(() => {
    if (!revision) return
    for (const animation of ref.current?.getAnimations() ?? []) {
      animation.currentTime = 0
      animation.play()
    }
  }, [revision])
  return <span ref={ref} data-tab={id} className={revision ? 'dock-tab-ping' : undefined}>{title}</span>
}

// ログ行の時刻 (ローカル HH:MM:SS).
const fmtTime = (ts: string): string => {
  const d = new Date(ts)
  if (isNaN(d.getTime())) return ''
  const p = (n: number) => String(n).padStart(2, '0')
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
}
const sameValue = (left: unknown, right: unknown): boolean => JSON.stringify(left) === JSON.stringify(right)

export const App = () => {
  const { t, lang, preferences, updatePreferences } = useSettings()
  const docking = useDocking()
  const { model } = docking
  const qc = useQueryClient()
  const projectsQuery = useQuery({
    queryKey: ['projects'], queryFn: api.listProjects, refetchInterval: 3000, retry: false,
  })
  const projects = projectsQuery.data
  const { data: settingsData } = useQuery({ queryKey: ['settings'], queryFn: api.getSettings, retry: false })
  const { data: doctor } = useQuery({ queryKey: ['doctor'], queryFn: api.getDoctor, staleTime: 30_000 })
  const [projectId, setProjectIdState] = useState<string | null>(preferences.lastProjectId)
  const writeProjectUi = useCallback(async (patch: ProjectUiState) => {
    if (!projectId) throw new Error('Cannot save project settings without a project')
    const saved = await api.patchUiState(projectId, patch, true)
    qc.setQueryData<typeof projects>(['projects'], current => current?.map(p => p.id === saved.id ? saved : p))
  }, [projectId, qc])
  const projectPersistence = usePersistence(writeProjectUi)
  const saveProjectUi = projectPersistence.save
  const setProjectId = useCallback((id: string | null) => {
    void projectPersistence.flush().then(() => {
      setProjectIdState(id)
      updatePreferences({ lastProjectId: id })
    }).catch(() => {})
  }, [projectPersistence.flush, updatePreferences])
  const [hydratedProjectId, setHydratedProjectId] = useState<string | null>(null)
  const [selectedStage, setSelectedStage] = useState<string>('extract_frames')
  const [reconMode, setReconMode] = useState<ReconMode>('native_fisheye')
  const [params, setParamsState] = useState<StageParams>(DEFAULT_PARAMS)
  const setParams = (patch: Partial<StageParams>) => {
    setParamsState(prev => ({ ...prev, ...patch }))
    if (projectId && hydratedProjectId === projectId) saveProjectUi({ params: patch })
  }
  const [browsing, setBrowsing] = useState(false)
  const [pendingSource, setPendingSource] = useState<Omit<SourceCreate, 'path'> | null>(null)
  const [managerOpen, setManagerOpen] = useState(false)
  const [clearOutputsOpen, setClearOutputsOpen] = useState(false)
  const [clearOutputStages, setClearOutputStagesState] = useState<string[]>([])
  const setClearOutputStages = (stages: string[]) => {
    setClearOutputStagesState(stages)
    saveProjectUi({ clearOutputStages: stages })
  }
  const [activeJobId, setActiveJobId] = useState<string | null>(null)
  const [events, setEvents] = useState<EventEnvelope[]>([])
  // 数値進捗と最新 activity は別々に保持する。activity-only 行が count/percentage を
  // 上書きすると、長い native process の最も有用な詳細が消えるため。
  const [progressByStage, setProgressByStage] = useState<Record<string, EventEnvelope>>({})
  const [activityByStage, setActivityByStage] = useState<Record<string, EventEnvelope>>({})
  const { showPoints, showCams } = preferences.viewer
  const setShowPoints = (value: boolean) => updatePreferences({ viewer: { showPoints: value } })
  const setShowCams = (value: boolean) => updatePreferences({ viewer: { showCams: value } })
  const [selectedCameraId, setSelectedCameraId] = useState<number | null>(null)
  const [selectedFrameIndex, setSelectedFrameIndex] = useState<number | null>(null)
  const [cameraPose, setCameraPoseState] = useState<ViewerCameraPose | null>(null)
  const setCameraPose = useCallback((pose: ViewerCameraPose) => {
    setCameraPoseState(pose)
    if (projectId && hydratedProjectId === projectId) saveProjectUi({ cameraPose: pose })
  }, [projectId, hydratedProjectId, saveProjectUi])
  const lvlOn = preferences.console
  const search = preferences.console.search
  const changeReconMode = (mode: ReconMode) => {
    if (mode === reconMode) return
    setReconMode(mode)
    if (!projectId) return
    saveProjectUi({ reconMode: mode })
    api.clearStage(projectId, 'prepare_images').then(async () => {
      await Promise.all(['reconstruction', 'masks', 'export-info'].map(
        key => qc.cancelQueries({ queryKey: [key, projectId] }),
      ))
      qc.setQueryData(['reconstruction', projectId], null)
      qc.setQueriesData({ queryKey: ['masks', projectId] }, null)
      qc.setQueryData(['export-info', projectId], null)
      qc.invalidateQueries({ queryKey: ['stages', projectId] })
    }).catch(error => console.warn('mode invalidation failed', error))
  }
  // Console 自動スクロール: 表示位置が最下部にある時だけ新ログへ追従する.
  const conRef = useRef<HTMLDivElement>(null)
  const atBottomRef = useRef(true)

  useEffect(() => {
    if (!projects) return
    if (projectId && projects.some(project => project.id === projectId)) return
    setProjectId(projects[0]?.id ?? null)
  }, [projects, projectId, setProjectId])
  const project = projects?.find(p => p.id === projectId) ?? null
  const primarySource = project?.sources.find(source => source.role === 'primary') ?? null
  // ソース種別で無効なモードだけ既定へ戻す (普通カメラは pinhole のみ).
  useEffect(() => {
    const projection = primarySource?.projection
    if (!projection) return
    if (!reconModesForProjection(projection).includes(reconMode))
      setReconMode(defaultReconModeForProjection(projection))
  }, [primarySource?.projection, reconMode]) // eslint-disable-line react-hooks/exhaustive-deps

  const hydratedRef = useRef<string | null>(null)
  useEffect(() => {
    if (!project || !settingsData || hydratedRef.current === project.id) return
    hydratedRef.current = project.id
    const ui = project.ui_state
    const sam3 = (settingsData as {
      sam3?: { feature_prompt?: string; training_prompt?: string }
    } | undefined)?.sam3
    const savedParams = (ui?.params ?? {}) as Record<string, unknown>
    const knownParams = Object.fromEntries(
      Object.keys(DEFAULT_PARAMS).filter(key => Object.hasOwn(savedParams, key)).map(key => [key, savedParams[key]]),
    ) as Partial<StageParams>
    setParamsState({
      ...DEFAULT_PARAMS,
      baUseGpu: Boolean(doctor?.checks.colmap?.capabilities?.gpu_bundle_adjustment),
      featureMaskPrompt: sam3?.feature_prompt ?? '',
      trainingMaskPrompt: sam3?.training_prompt ?? '',
      ...knownParams,
    })
    const savedReconMode = ui?.reconMode as ReconMode | undefined
    const sourceModes = reconModesForProjection(primarySource?.projection ?? null)
    setReconMode(savedReconMode && sourceModes.includes(savedReconMode)
      ? savedReconMode
      : defaultReconModeForProjection(primarySource?.projection ?? null))
    const clearable = new Set<string>(CLEARABLE_STAGE_NAMES)
    setClearOutputStagesState(Array.isArray(ui?.clearOutputStages)
      ? ui.clearOutputStages.filter((stage): stage is string => typeof stage === 'string' && clearable.has(stage))
      : [])
    setSelectedStage(ui?.selectedStage ?? (project.sources.length ? 'extract_frames' : 'inspect_source'))
    setSelectedCameraId(ui?.selectedCameraId ?? null)
    setSelectedFrameIndex(ui?.selectedFrameIndex ?? null)
    setCameraPoseState(ui?.cameraPose ?? null)
    setActiveJobId(null)
    setEvents([])
    setProgressByStage({})
    setActivityByStage({})
    stageJobsRef.current = {}
    pendingStageEvents.current = {}
    setHydratedProjectId(project.id)
  }, [project, settingsData, doctor])

  const stagesQuery = useQuery({
    queryKey: ['stages', projectId], queryFn: () => api.getStages(projectId as string),
    enabled: !!projectId, refetchInterval: 2000, retry: false,
  })
  const stagesData = stagesQuery.data
  const { data: sourceInfo } = useQuery({
    queryKey: ['source-info', projectId], queryFn: () => api.getSourceInfo(projectId as string),
    enabled: !!projectId && !!project?.sources.length, retry: false,
  })
  const resolutionRecommendation = useMemo(
    () => sourceInfo
      ? recommendSourceResolutions(project?.sources ?? [], sourceInfo, reconMode)
      : null,
    [project?.sources, reconMode, sourceInfo],
  )
  useEffect(() => {
    if (!resolutionRecommendation || hydratedProjectId !== projectId) return
    const next = {
      featureMaxImageSize: params.featureMaxImageSizeAuto
        ? resolutionRecommendation.featureMaxImageSize : params.featureMaxImageSize,
      featureMaxNumFeatures: params.featureMaxNumFeaturesAuto
        ? resolutionRecommendation.featureMaxNumFeatures : params.featureMaxNumFeatures,
      featureMaskSize: params.featureMaskSizeAuto
        ? resolutionRecommendation.maskMaxInferenceSize : params.featureMaskSize,
      trainingMaskSize: params.trainingMaskSizeAuto
        ? resolutionRecommendation.maskMaxInferenceSize : params.trainingMaskSize,
      size: params.sizeAuto ? resolutionRecommendation.pinholeViewSize : params.size,
    }
    if (Object.entries(next).some(([key, value]) => params[key as keyof StageParams] !== value)) setParams(next)
  }, [
    hydratedProjectId,
    params.featureMaskSizeAuto,
    params.featureMaxImageSizeAuto,
    params.featureMaxNumFeaturesAuto,
    params.sizeAuto,
    params.trainingMaskSizeAuto,
    projectId,
    resolutionRecommendation,
  ])
  const { data: recon, dataUpdatedAt: reconUpdatedAt } = useQuery({
    queryKey: ['reconstruction', projectId], queryFn: () => api.getReconstruction(projectId as string),
    enabled: !!projectId, retry: false,
  })
  useEffect(() => { if (reconUpdatedAt) docking.ping('scene') }, [reconUpdatedAt, docking.ping])
  const { data: framesData } = useQuery({
    queryKey: ['frames', projectId], queryFn: () => api.getFrames(projectId as string),
    enabled: !!projectId, retry: false,
  })
  const sourceRegions = useQueries({
    queries: (project?.sources.filter(source => source.enabled) ?? []).map(source => ({
      queryKey: ['source-region', projectId, source.id],
      queryFn: () => api.getSourceRegion(projectId as string, source.id),
      retry: false,
    })),
  })
  const configuredRegions = sourceRegions.filter(region => region.data?.saved && !region.data.needs_review).length
  const regionsComplete = sourceRegions.length > 0 && configuredRegions === sourceRegions.length
  const regionsFailed = sourceRegions.some(region => region.isError)
  const regionsLoading = sourceRegions.some(region => region.isPending)
  const backendUnavailable = projectsQuery.isError || stagesQuery.isError
  const stageJobsRef = useRef<Record<string, string | null>>({})
  const pendingStageEvents = useRef<Record<string, Map<string | null, {
    activity: EventEnvelope
    progress?: EventEnvelope
  }>>>({})
  useEffect(() => {
    if (!stagesData || hydratedProjectId !== projectId) return
    stageJobsRef.current = Object.fromEntries(
      stagesData.stages.map(stage => [stage.stage, stage.job_id]),
    )
    const pending = pendingStageEvents.current
    pendingStageEvents.current = {}
    const newest = (snapshot: EventEnvelope | null, buffered?: EventEnvelope) => (
      buffered && (!snapshot || buffered.id > snapshot.id) ? buffered : snapshot
    )
    const snapshots = stagesData.stages.map(stage => {
      const buffered = pending[stage.stage]?.get(stage.job_id)
      return {
        ...stage,
        progress_event: newest(stage.progress_event, buffered?.progress),
        activity_event: newest(stage.activity_event, buffered?.activity),
      }
    })
    setProgressByStage(previous => {
      const next = { ...previous }
      for (const stage of snapshots) {
        if (!['running', 'queued'].includes(stage.status ?? '') || !stage.job_id) continue
        const snapshot = stage.progress_event
        const existing = next[stage.stage]
        if (!snapshot) {
          if (existing?.job_id !== stage.job_id) delete next[stage.stage]
          continue
        }
        if (existing?.job_id === stage.job_id && existing.id >= snapshot.id) continue
        next[stage.stage] = snapshot
      }
      return next
    })
    setActivityByStage(previous => {
      const next = { ...previous }
      for (const stage of snapshots) {
        if (!['running', 'queued'].includes(stage.status ?? '') || !stage.job_id) continue
        const snapshot = stage.activity_event
        const existing = next[stage.stage]
        if (!snapshot) {
          if (existing?.job_id !== stage.job_id) delete next[stage.stage]
          continue
        }
        if (existing?.job_id === stage.job_id && existing.id >= snapshot.id) continue
        next[stage.stage] = snapshot
      }
      return next
    })
  }, [stagesData, projectId, hydratedProjectId])

  const previousStageRuns = useRef<{ projectId: string | null; states: Record<string, string | null> }>({
    projectId: null,
    states: {},
  })
  useEffect(() => {
    if (!stagesData || !projectId) return
    const current = Object.fromEntries(stagesData.stages.map(stage => [stage.stage,
      `${stage.status}:${stage.job_id}:${stage.finished_at}:${stage.has_output}`]))
    const previous = previousStageRuns.current
    if (previous.projectId === projectId) {
      const completedExternally = stagesData.stages.some(stage =>
        previous.states[stage.stage] !== current[stage.stage]
        && !['running', 'queued'].includes(stage.status ?? ''))
      if (completedExternally) {
        for (const key of ['frames', 'reconstruction', 'source-region', 'masks', 'export-info'])
          qc.invalidateQueries({ queryKey: [key, projectId] })
      }
    }
    previousStageRuns.current = { projectId, states: current }
  }, [stagesData, projectId, qc])
  const primaryFrames = useMemo(
    () => framesData?.frames.filter(frame => frame.source_id === primarySource?.id) ?? [],
    [framesData, primarySource?.id],
  )
  const primarySharpnessScores = useMemo(
    () => primaryFrames.flatMap(frame => frame.score?.sharpness == null ? [] : [frame.score.sharpness]),
    [primaryFrames],
  )
  const route = [
    'inspect_source',
    'extract_frames',
    'prepare_images',
    'rectify_fisheye',
    ...(params.featureMaskEnabled ? ['generate_feature_masks'] : []),
    ...(params.trainingMaskEnabled ? ['generate_training_masks'] : []),
    'extract_features',
    'match_features',
    'reconstruct',
    'align_reconstruction',
    'restore_metric_scale',
    'scene_alignment',
    'cleanup_sparse',
    'dense_initialization',
    'export_dataset',
  ]
  const stageParamChanges = (stage: NonNullable<typeof stagesData>['stages'][number]) => {
    const appliedParams = ['running', 'queued'].includes(stage.status ?? '')
      ? stage.active_params
      : stage.params
    if (!appliedParams) return []
    const expected = paramsForStage(stage.stage, params, reconMode)
    return Object.entries(expected).flatMap(([key, value]) => (
      sameValue(appliedParams[key], value)
        ? []
        : [{ key, previous: appliedParams[key], next: value }]
    ))
  }
  const stageIsFresh = (stage: NonNullable<typeof stagesData>['stages'][number]): boolean => {
    if (!stage.has_output || !stage.params || stage.status === 'stale') return false
    return stageParamChanges(stage).length === 0
  }
  const outputs = new Map((stagesData?.stages ?? []).map(stage => [stage.stage, stageIsFresh(stage)]))
  const nextStep = route.find(stage => !outputs.get(stage)) ?? null
  // 次工程を開始できない構成を検出し, 理由をボタン tooltip に出す.
  // ソース有効領域は projection ごとの既定領域で動作し、保存を run-all の前提にしない。
  const runReason: string | null =
    backendUnavailable ? t('backendDisconnected')
    : !project?.sources.length ? t('noSource')
    : primarySource && !reconModesForProjection(primarySource.projection).includes(reconMode) ? t('modeMismatch')
    : null

  // 実行中ジョブの追跡 (停止ボタン用).
  const jobQuery = useQuery({
    queryKey: ['job', activeJobId], queryFn: () => api.getJob(activeJobId as string),
    enabled: !!activeJobId, refetchInterval: 1200, retry: false,
  })
  const jobData = jobQuery.data
  useEffect(() => {
    if (jobData && ['succeeded', 'failed', 'cancelled'].includes(jobData.status)) {
      setActiveJobId(null)
      // ジョブ完了で成果物が変わるため, 依存クエリを更新 (写真リスト/再構成/魚眼領域).
      for (const key of ['frames', 'reconstruction', 'source-region', 'masks', 'export-info', 'stages']) {
        qc.invalidateQueries({ queryKey: [key, projectId] })
      }
    }
  }, [jobData]) // eslint-disable-line react-hooks/exhaustive-deps
  const anyStageRunning = !backendUnavailable
    && !!stagesData?.stages.some(s => ['running', 'queued'].includes(s.status ?? ''))
  const jobRunning = !backendUnavailable && (jobData?.status === 'running' || jobData?.status === 'queued')
  const processing = jobRunning || anyStageRunning
  // 停止対象: UI が開始した job を優先, 無ければ実行中ステージの job_id (別セッション/外部起動でも止められる).
  const runningJobId = stagesData?.stages.find(s => ['running', 'queued'].includes(s.status ?? ''))?.job_id ?? null
  const stopTarget = activeJobId ?? runningJobId
  const stopJob = () => {
    if (stopTarget) api.cancelJob(stopTarget).then(() => qc.invalidateQueries({ queryKey: ['stages', projectId] }))
  }

  const synthId = useRef(-1)
  useEffect(() => {
    if (!projectId || hydratedProjectId !== projectId) return
    const ws = openEventStream({ projectId, since: -1 }, e => {
      if (e.stage && stageJobsRef.current[e.stage] !== e.job_id) {
        // A socket can arrive before the job snapshot. Keep jobs separate until confirmed.
        const jobs = pendingStageEvents.current[e.stage] ??= new Map()
        const buffered = jobs.get(e.job_id)
        jobs.set(e.job_id, {
          activity: buffered && buffered.activity.id >= e.id ? buffered.activity : e,
          progress: e.progress != null && (!buffered?.progress || buffered.progress.id < e.id)
            ? e : buffered?.progress,
        })
      } else if (e.stage) {
        setActivityByStage(previous => {
          const existing = previous[e.stage as string]
          if (existing?.job_id === e.job_id && existing.id >= e.id) return previous
          return { ...previous, [e.stage as string]: e }
        })
        if (e.progress != null) {
          setProgressByStage(previous => {
            const existing = previous[e.stage as string]
            if (existing?.job_id === e.job_id && existing.id >= e.id) return previous
            return { ...previous, [e.stage as string]: e }
          })
        }
      }
      if (e.kind !== 'progress') {
        setEvents(prev => [...prev.slice(-500), e])
        docking.ping('console')
      }
    }, status => {
      // 接続断 / 再接続を Console に 1 行ずつ出す (クライアント合成イベント).
      const key = status === 'disconnected' ? 'log.ws_disconnected'
        : status === 'reconnected' ? 'log.ws_reconnected' : 'log.ws_invalid_event'
      setEvents(prev => [...prev.slice(-500), {
        id: synthId.current--, job_id: null, project_id: projectId, stage: null,
        level: status === 'reconnected' ? 'info' : 'warn', message: key,
        msg_key: key, msg_args: null, progress: null, kind: 'log', ts: new Date().toISOString(),
      }])
    })
    return () => ws.close()
  }, [projectId, hydratedProjectId])

  const refreshAfterSourceMutation = () => {
    qc.invalidateQueries({ queryKey: ['projects'] })
    qc.invalidateQueries({ queryKey: ['source-info', projectId] })
    qc.invalidateQueries({ queryKey: ['stages', projectId] })
    for (const key of ['reconstruction', 'frames', 'source-region', 'masks', 'export-info'])
      qc.removeQueries({ queryKey: [key, projectId] })
    setSelectedFrameIndex(null)
    setSelectedCameraId(null)
    saveProjectUi({ selectedFrameIndex: null, selectedCameraId: null })
  }
  const addSource = useMutation({
    mutationFn: (source: SourceCreate) => api.addSource(projectId as string, source),
    onSuccess: () => {
      refreshAfterSourceMutation()
      setPendingSource(null)
    },
  })
  const deleteSource = useMutation({
    mutationFn: (sourceId: string) => api.deleteSource(projectId as string, sourceId),
    onSuccess: refreshAfterSourceMutation,
  })
  const makePrimarySource = useMutation({
    mutationFn: (sourceId: string) => api.makePrimarySource(projectId as string, sourceId),
    onSuccess: refreshAfterSourceMutation,
  })
  const runPipeline = useMutation({
    mutationFn: async () => {
      await projectPersistence.flush()
      return api.runPipeline(projectId as string,
        Object.fromEntries(route.map(stage => [stage, paramsForStage(stage, params, reconMode)])),
        ['generate_feature_masks', 'generate_training_masks'].filter(stage => !route.includes(stage)),
      )
    },
    onSuccess: response => {
      setActiveJobId(response.job_id)
      setProgressByStage({})
      setActivityByStage({})
      qc.invalidateQueries({ queryKey: ['stages', projectId] })
    },
  })
  const clearOutputs = useMutation({
    mutationFn: (stages: string[]) => api.clearOutputs(projectId as string, stages),
    onSuccess: async result => {
      const artifactKeys = ['reconstruction', 'frames', 'source-region', 'masks', 'export-info']
      await Promise.all(artifactKeys.map(key => qc.cancelQueries({ queryKey: [key, projectId] })))
      const cleared = new Set(result.cleared)
      if (['reconstruct', 'align_reconstruction', 'restore_metric_scale', 'scene_alignment',
        'cleanup_sparse', 'dense_initialization'].some(stage => cleared.has(stage))) {
        qc.setQueryData(['reconstruction', projectId], null)
      }
      if (cleared.has('extract_frames')) qc.setQueryData(['frames', projectId], null)
      if (cleared.has('generate_feature_masks') || cleared.has('generate_training_masks'))
        qc.setQueriesData({ queryKey: ['masks', projectId] }, null)
      if (cleared.has('export_dataset')) qc.setQueryData(['export-info', projectId], null)
      qc.invalidateQueries({ queryKey: ['stages', projectId] })
      qc.invalidateQueries({ queryKey: ['projects'] })
      setSelectedFrameIndex(null); setSelectedCameraId(null)
      setProgressByStage({})
      setActivityByStage({})
      setClearOutputsOpen(false)
    },
  })

  const onJob = (jobId: string) => {
    setActiveJobId(jobId)
    setProgressByStage(previous => {
      const next = { ...previous }
      delete next[selectedStage]
      return next
    })
    setActivityByStage(previous => {
      const next = { ...previous }
      delete next[selectedStage]
      return next
    })
    qc.invalidateQueries({ queryKey: ['stages', projectId] })
  }
  const selectStep = (stage: string) => {
    if (hydratedProjectId !== projectId) return
    setSelectedStage(stage); setSelectedCameraId(null); setSelectedFrameIndex(null)
    saveProjectUi({ selectedStage: stage, selectedCameraId: null, selectedFrameIndex: null })
    docking.reveal('inspector')
  }
  const selectCamera = (id: number | null) => {
    setSelectedCameraId(id)
    setSelectedFrameIndex(null)
    if (id !== null) {
      setSelectedStage('')
      docking.ping('inspector')
    }
    saveProjectUi({ selectedCameraId: id, selectedFrameIndex: null, ...(id !== null ? { selectedStage: '' } : {}) })
  }
  const selectFrame = (index: number) => {
    setSelectedFrameIndex(index); setSelectedStage(''); setSelectedCameraId(null)
    saveProjectUi({ selectedFrameIndex: index, selectedStage: '', selectedCameraId: null })
    docking.reveal('inspector')
  }
  const selectedCamImage = selectedCameraId != null ? recon?.images.find(i => i.id === selectedCameraId) : undefined
  const selectedCamModel = selectedCamImage
    ? recon?.cameras.find(camera => camera.id === selectedCamImage.camera_id)?.model
    : undefined
  // 現在の再構成結果のカメラモデルからモード判定. EQUIRECTANGULAR を先に見る (FISHEYE を含まない).
  const resultFamilies = new Set(recon?.cameras.map(camera =>
    camera.model.includes('EQUIRECTANGULAR') ? 'equirect'
      : camera.model.includes('FISHEYE') ? 'native' : 'pinhole') ?? [])
  const resultMode: 'native' | 'pinhole' | 'equirect' | 'mixed' | null = resultFamilies.size > 1
    ? 'mixed'
    : resultFamilies.size === 1 ? [...resultFamilies][0] as 'native' | 'pinhole' | 'equirect'
    : null
  // KEY+ARGS を持つイベントは表示言語で翻訳し, 未 key 行は message フォールバックを使う.
  const renderMsg = (event: EventEnvelope) => (
    event.msg_key ? translateMsg(lang, event.msg_key, event.msg_args) : event.message
  )

  const imageSourcesOnly = !!project?.sources.length
    && project.sources.filter(source => source.enabled).every(source => source.media_kind === 'images')
  const items: HierItem[] = []
  for (const s of stagesData?.stages ?? []) {
    const en = s.stage === 'generate_feature_masks'
      ? params.featureMaskEnabled
      : s.stage === 'generate_training_masks' ? params.trainingMaskEnabled
      : s.stage === 'cleanup_sparse' ? params.cleanupSparseEnabled
      : s.stage === 'dense_initialization' ? params.denseEnabled : true
    const toggleable = s.stage === 'generate_feature_masks'
      || s.stage === 'generate_training_masks'
      || s.stage === 'cleanup_sparse'
      || s.stage === 'dense_initialization'
    const done = stageIsFresh(s)
    const parameterChanges = stageParamChanges(s)
    const stale = s.status === 'stale' || (s.has_output && !done)
    const running = !backendUnavailable && ['running', 'queued'].includes(s.status ?? '')
    const progressEvent = progressByStage[s.stage]
    const activityEvent = activityByStage[s.stage]
    const currentProgress = progressEvent?.job_id === s.job_id ? progressEvent.progress : null
    const progressDetail = progressEvent?.job_id === s.job_id ? renderMsg(progressEvent) : ''
    const activityDetail = activityEvent?.job_id === s.job_id && activityEvent.id !== progressEvent?.id
      ? renderMsg(activityEvent) : ''
    items.push({
      key: s.stage,
      label: s.stage === 'extract_frames' && imageSourcesOnly ? t('st_collect_images') : t(`st_${s.stage}`),
      badgeColor: backendUnavailable ? 'var(--error)' : s.status === 'failed' ? 'var(--error)'
        : s.status === 'cancelled' ? '#888' : stale ? '#d69a2a' : done ? '#4caf50' : 'var(--border)',
      statusLabel: backendUnavailable ? t('backendDisconnected')
        : s.status === 'running' ? t('running') : s.status === 'queued' ? t('queued')
        : s.status === 'cancelled' ? t('cancelled') : s.status === 'failed' ? t('failed')
        : stale ? t('stale') : done ? t('done') : t('notrun'),
      toggleable, enabled: en, dim: !en, running,
      progress: currentProgress,
      detail: [
        stale && parameterChanges.length
          ? t('pendingParameterChanges').replace('{changes}', parameterChanges.map(change => (
              `${change.key}: ${JSON.stringify(change.previous)} → ${JSON.stringify(change.next)}`
            )).join(', '))
          : '',
        progressDetail,
        activityDetail,
      ].filter(Boolean).join(' · '),
    })
    if (s.stage === 'extract_frames' && project?.sources.length) {
      items.push({
        key: 'source_region', label: t('st_source_region'),
        badgeColor: backendUnavailable || regionsFailed ? 'var(--error)'
          : regionsComplete ? '#4caf50' : configuredRegions ? '#d69a2a' : 'var(--border)',
        statusLabel: backendUnavailable ? t('backendDisconnected')
          : regionsFailed ? t('failed') : regionsLoading ? t('working') : regionsComplete ? t('done')
          : configuredRegions ? `${t('regionConfigured')} ${configuredRegions}/${sourceRegions.length}` : t('regionUnset'),
        toggleable: false, enabled: true,
      })
    }
  }

  const stageStatus = stagesData?.stages.find(s => s.stage === selectedStage)
  const featureMaskRunning = !backendUnavailable
    && stagesData?.stages.find(s => s.stage === 'generate_feature_masks')?.status === 'running'
  const trainingMaskRunning = !backendUnavailable
    && stagesData?.stages.find(s => s.stage === 'generate_training_masks')?.status === 'running'
  const selectedProgressCandidate = progressByStage[selectedStage]
  const selectedProgress = selectedProgressCandidate?.job_id === stageStatus?.job_id
    ? selectedProgressCandidate : undefined
  const selectedActivityCandidate = activityByStage[selectedStage]
  const selectedActivity = selectedActivityCandidate?.job_id === stageStatus?.job_id
    ? selectedActivityCandidate : undefined
  const lastEv = events.length ? events[events.length - 1] : null
  const lastLog = lastEv ? `[${fmtTime(lastEv.ts)}] ${lastEv.level.toUpperCase()} ${renderMsg(lastEv)}` : ''
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
            <SceneHierarchy key={projectId} recon={recon} frames={framesData?.frames} sources={framesData?.sources}
              showPoints={showPoints} setShowPoints={setShowPoints}
              showCams={showCams} setShowCams={setShowCams} selectedCameraId={selectedCameraId}
              onSelectCamera={id => { selectCamera(id); docking.reveal('inspector') }}
              selectedFrameIndex={selectedFrameIndex} onSelectFrame={selectFrame} />
          </div>
        )
      case 'scene':
        return (
          <div style={{ position: 'relative', width: '100%', height: '100%', background: 'var(--bg)' }}>
            <Suspense fallback={<div className="hint">{t('loadingViewer')}</div>}>
              <SceneView node={node} projectId={hydratedProjectId === projectId ? projectId : null}
                recon={hydratedProjectId === projectId ? recon : null}
                revision={reconUpdatedAt}
                preferences={preferences.viewer} onPreferencesChange={patch => updatePreferences({ viewer: patch })}
                cameraPose={hydratedProjectId === projectId ? cameraPose : null} onCameraPoseChange={setCameraPose}
                selectedCameraId={selectedCameraId} onPickCamera={selectCamera} />
            </Suspense>
            {recon && (
              <div className="scene-info">
                <span className="scene-metric"><AppIcon name="image" size={14} />
                  {t('sceneImages')} {recon.stats.num_images}</span>
                <span className="scene-metric"><AppIcon name="points" size={14} />
                  {t('scenePoints')} {recon.stats.num_points3D.toLocaleString()}</span>
                {recon.stats.registered_ratio != null && <span className="scene-metric">
                  <AppIcon name="checkmark" size={14} />
                  {t('sceneRegistered')} {(recon.stats.registered_ratio * 100).toFixed(0)}%
                </span>}
                {recon.stats.camera_trajectory_diameter != null && <span className="scene-metric">
                  <AppIcon name="path" size={14} />
                  {t('scenePathSpan')} {recon.stats.camera_trajectory_diameter.toFixed(3)}{' '}
                  {recon.metric_scale?.metric ? t('sceneMeters') : t('sceneUnits')}
                </span>}
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
            {!projectId || hydratedProjectId !== projectId
              ? <div className="hint">{projectId ? t('loadingSettings') : t('noProject')}</div>
              : selectedCamImage
              ? <CameraInspector projectId={projectId as string} image={selectedCamImage}
                  cameraModel={selectedCamModel ?? ''}
                  sources={project?.sources ?? []} featureMaskRunning={featureMaskRunning}
                  trainingMaskRunning={trainingMaskRunning} />
              : selectedFrameIndex != null
              ? <FrameInspector projectId={projectId as string} frameIndex={selectedFrameIndex}
                  frames={framesData?.frames} recon={recon} sources={project?.sources ?? []}
                  featureMaskRunning={featureMaskRunning} trainingMaskRunning={trainingMaskRunning} />
              : selectedStage === 'source_region'
              ? <SourceRegionEditor key={projectId} projectId={projectId as string}
                  sources={project?.sources ?? []} frames={framesData?.frames ?? []} />
              : selectedStage
              ? <StageSettings projectId={projectId as string} stage={selectedStage} status={stageStatus}
                  sourceInfo={sourceInfo} reconMode={reconMode} setReconMode={changeReconMode} params={params} setParams={setParams}
                  onJob={onJob} hasSource={!!project?.sources.length} sources={project?.sources ?? []} resultMode={resultMode}
                  primaryProjection={primarySource?.projection ?? null} processing={processing}
                  stageIsRunning={!backendUnavailable && ['running', 'queued'].includes(stageStatus?.status ?? '')}
                  onStop={stopJob}
                  onSelectSource={source => { addSource.reset(); setPendingSource(source); setBrowsing(true) }}
                  onDeleteSource={sourceId => deleteSource.mutate(sourceId)}
                  onMakePrimarySource={sourceId => makePrimarySource.mutate(sourceId)}
                  sourceMutationError={addSource.error ?? deleteSource.error ?? makePrimarySource.error}
                  frameSelection={primarySource
                    ? framesData?.sources.find(source => source.id === primarySource.id)?.selection : null}
                  frameSharpnessScores={primarySharpnessScores}
                  stageProgress={selectedProgress?.progress ?? null}
                  stageStartedAt={backendUnavailable ? null : stageStatus?.started_at ?? null}
                  stageProgressMsg={selectedProgress ? renderMsg(selectedProgress) : ''}
                  stageActivityMsg={selectedActivity && selectedActivity.id !== selectedProgress?.id
                    ? renderMsg(selectedActivity) : ''}
                  blockedReason={null} />
              : <div className="hint">—</div>}
          </div>
        )
      case 'console':
        return (
          <div style={{ height: '100%', display: 'flex', flexDirection: 'column', minWidth: 0 }}>
            <div className="con-bar">
              {(['info', 'warn', 'error'] as Lvl[]).map(l => (
                <button key={l} className={`con-chip${lvlOn[l] ? ' on' : ''}`} title={l}
                  aria-label={l} onClick={() => updatePreferences({ console: { [l]: !lvlOn[l] } })}>
                  <AppIcon name={LVL_ICON[l]} size={15} />
                </button>
              ))}
              <span className="con-search-wrap"><AppIcon name="search" size={14} />
                <input className="con-search" placeholder={t('search')} value={search}
                  onChange={e => updatePreferences({ console: { search: e.target.value } })} />
              </span>
              <button className="con-chip icon-label" onClick={() => setEvents([])}>
                <AppIcon name="broom" size={14} /> {t('clearLog')}
              </button>
            </div>
            <div className="mono con-log" ref={conRef}
              onScroll={e => { const el = e.currentTarget; atBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24 }}>
              {filtered.map(e => (
                <div key={e.id}>
                  <span style={{ color: 'var(--fg-mute)' }}>
                    [{fmtTime(e.ts)}]{e.stage ? `[${t(`st_${e.stage}`)}]` : ''}
                  </span>{' '}
                  <span className={`log-level log-level-${e.level}`} title={e.level}>
                    <AppIcon name={LVL_ICON[e.level as Lvl] ?? 'info'} size={11} />
                  </span>{' '}
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
    return <ErrorBoundary key={comp === 'scene' ? comp : `${comp}:${projectId}`} label={comp}>{content}</ErrorBoundary>
  }

  return (
    <div className="ide">
      <div className="ide-top">
        <button className="btn top-glass-button icon-label" onClick={() => setManagerOpen(true)}>
          <AppIcon name="navigation" /> {t('projects')}
        </button>
        <h1 style={{ margin: '0 4px' }}>{project?.name ?? 'sphere-reconstruct'}</h1>
        {primarySource
          ? <PathText path={primarySource.path} compact className="ide-source-path" />
          : <span className="mono ide-source-path">{t('noSource')}</span>}
        <div className="top-actions">
          <button className="btn btn-secondary top-glass-button icon-label"
            title={t('clearOutputs')}
            disabled={!projectId || clearOutputs.isPending || processing || backendUnavailable}
            onClick={() => { clearOutputs.reset(); setClearOutputsOpen(true) }}>
            <AppIcon name="broom" /> {t('clearOutputs')}
          </button>
          {processing
            ? <button className="btn stop top-glass-button icon-label" onClick={stopJob}>
                <AppIcon name="stop" /> {t('stop')}
              </button>
            : <button className="btn top-glass-button icon-label" disabled={!projectId || hydratedProjectId !== projectId || !!runReason || !nextStep || runPipeline.isPending}
                title={runReason ?? t('runAll')} onClick={() => runPipeline.mutate()}>
                <AppIcon name="play" /> {t('runAll')}
              </button>}
          <SettingsMenu />
        </div>
      </div>

      <div className="ide-center">
        <ErrorBoundary label="layout">
          <Layout model={model} factory={factory} onModelChange={docking.onModelChange}
            onRenderTab={(node, values) => {
              const id = node.getId()
              values.content = <TabTitle id={id} revision={docking.pings[id] ?? 0}
                title={t(docking.titles[id as keyof typeof docking.titles])} />
            }} />
        </ErrorBoundary>
      </div>

      {(projectPersistence.error || runPipeline.error) && <div className="persistence-error" role="alert">
        {projectPersistence.error ? `${t('settingsSaveFailed')}: ${projectPersistence.error.message}` : runPipeline.error?.message}
        {projectPersistence.error && <button className="btn"
          onClick={() => { void projectPersistence.flush().catch(() => {}) }}>{t('retry')}</button>}
      </div>}

      <div className="ide-status">
        <span className={`laststatus${backendUnavailable ? ' backend-offline' : ''}`}>
          {backendUnavailable ? <><AppIcon name="error" size={13} /> {t('backendDisconnectedHint')}</> : lastLog}
        </span>
        <SystemStatsBar />
      </div>

      {browsing && projectId && (
        <FileBrowser selectionError={addSource.error} onClose={() => { setBrowsing(false); setPendingSource(null) }}
          selectionKind={pendingSource?.media_kind === 'images' ? 'directory' : 'file'}
          onPick={path => {
            if (!pendingSource) return
            addSource.mutate({ ...pendingSource, path }, { onSuccess: () => setBrowsing(false) })
          }} />
      )}
      {managerOpen && (
        <ProjectManager projects={projects} currentId={projectId} onSelect={setProjectId}
          onBeforeMutation={projectPersistence.flush} onClose={() => setManagerOpen(false)} />
      )}
      {clearOutputsOpen && projectId && (
        <ClearOutputsDialog stages={stagesData?.stages ?? []} selectedStages={clearOutputStages}
          onSelectedStagesChange={setClearOutputStages} pending={clearOutputs.isPending}
          error={clearOutputs.error} onClose={() => setClearOutputsOpen(false)}
          onClear={stages => clearOutputs.mutate(stages)} />
      )}
    </div>
  )
}
