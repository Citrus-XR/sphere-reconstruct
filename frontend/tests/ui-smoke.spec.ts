import { expect, test, type Page } from '@playwright/test'
import path from 'node:path'
import { COLMAP_TRIANGULATION_PRESETS, DEFAULT_PARAMS, paramsForStage, type ReconMode } from '../src/features/stageParams'
import { type ProjectSource, type SourceRegion } from '../src/api/client'
import { mergeUiPatch } from '../src/ui/persistence'
import type { RootState } from '@react-three/fiber'

const MOCK_SOURCE = 'D:\\VID 2026\\clip.insv'
const MOCK_EXPORT = 'D:\\LFStudio\\export_dataset'
const MOCK_ENV_PATH = 'D:\\very-long-workspace-directory\\nested-runtime\\models\\and-tools\\current-environment'
const MOCK_PHONE = 'D:\\mixed-inputs\\Phone photos'
const TRAINING_MASK_PROMPT = "person,camera operator,person's shadow"
const FEATURE_MASK_PROMPT = `${TRAINING_MASK_PROMPT},animal,sky,vehicle,water`

const expandPhotos = async (page: Page) => {
  const toggle = page.getByRole('button', { name: /Photos/ })
  if (await toggle.getAttribute('aria-expanded') === 'false') await toggle.click()
}

interface MockOptions {
  regionSources?: ProjectSource[]
  regionSaved?: boolean
  sourceKind?: 'insv' | 'erp_video' | 'perspective_images'
  imageName?: string
  cameraModel?: string
  reconMode?: ReconMode
  secondProject?: boolean
  multipleFrameSources?: boolean
  runningStage?: string
  runningProgress?: number | null
  runningMessageKey?: string
  runningMessageArgs?: Record<string, unknown>
  runningActivityMessageKey?: string
  runningActivityMessageArgs?: Record<string, unknown>
  runningHasOutput?: boolean
  socketEvents?: Array<Record<string, unknown>>
  incrementalFeatureMask?: boolean
  frameCount?: number
  gpuBundleAdjustment?: boolean
  pendingStages?: string[]
  lastProjectId?: string
  disconnectStagesAfter?: number
  sourceWidth?: number
  sourceHeight?: number
  stageParams?: Record<string, Record<string, unknown>>
  runningActiveParams?: Record<string, unknown>
  emptyScene?: boolean
  initialStageSnapshotDelayMs?: number
  savedUiStates?: Record<string, Record<string, unknown>>
}

const installUiMock = async (page: Page, options: MockOptions = {}) => {
  const sourceKind = options.sourceKind ?? 'insv'
  const isInsv = sourceKind === 'insv'
  const isPerspectiveImages = sourceKind === 'perspective_images'
  const reconMode = options.reconMode ?? (isInsv ? 'native_fisheye'
    : isPerspectiveImages ? 'pinhole_rig' : 'equirectangular')
  const imageName = options.imageName ?? (isInsv
    ? 'sources/s1/lens0/frame_000000.jpg'
    : isPerspectiveImages ? 'sources/s1/camera_00/frame_000000.jpg' : 'sources/s1/frame_000000.jpg')
  const unexpectedRequests: string[] = []
  const reruns: Array<{ stage: string; body: Record<string, Record<string, unknown>> }> = []
  const pipelines: Array<{ params_by_stage: Record<string, Record<string, unknown>>; skip: string[] }> = []
  let preferences: Record<string, unknown> = {
    theme: 'auto', lang: 'en', lastProjectId: options.lastProjectId ?? null, layout: null, compactLayout: null,
    viewer: { showPoints: true, showCams: true, pointSize: 2.5, showGrid: true, showCenter: true, background: null },
    console: { info: true, warn: true, error: true, debug: false, search: '' },
  }
  const sourceAdds: Array<Record<string, unknown>> = []
  const clearRequests: string[][] = []
  const uiStateUpdates: Array<{ projectId: string; ui: Record<string, unknown> }> = []
  const deletedProjects = new Set<string>()
  const createdProjects: Array<Record<string, unknown>> = []
  const maskRequests = { feature: 0, training: 0 }
  const pendingStages = new Set(options.pendingStages ?? [])
  let stageRequestCount = 0
  const uiStates: Record<string, Record<string, unknown> | null> = {
    p1: options.reconMode ? {
      reconMode: options.reconMode,
      params: {},
      disabled: [],
    } : null,
    p2: null,
    ...options.savedUiStates,
  }
  let reconstructionAvailable = !options.emptyScene
  const stageParamOverrides: Record<string, Record<string, unknown>> = {}
  const sourcesByProject: Record<string, Array<Record<string, unknown>>> = {
    p1: [{
      id: 's1', label: isInsv ? 'Primary 360' : isPerspectiveImages ? 'Primary camera' : 'Primary ERP', role: 'primary',
      adapter: isInsv ? 'insta360' : isPerspectiveImages ? 'generic_images' : 'generic_video',
      media_kind: isPerspectiveImages ? 'images' : 'video',
      projection: isInsv ? 'dual_fisheye' : isPerspectiveImages ? 'perspective' : 'equirectangular',
      path: isPerspectiveImages ? MOCK_PHONE : MOCK_SOURCE,
      ordinal: 0, enabled: true,
    }],
    p2: [],
  }
  if (options.regionSources) sourcesByProject.p1 = options.regionSources.map(source => ({ ...source }))
  const regions: Record<string, SourceRegion> = {}
  await page.routeWebSocket('**/api/events**', socket => {
    if (!options.socketEvents?.length) return
    setTimeout(() => {
      for (const event of options.socketEvents ?? []) {
        const { msg_args: messageArgs, ...rest } = event
        socket.send(JSON.stringify({
          level: 'info', project_id: 'p1', kind: 'progress', ts: '2026-01-01T00:00:02Z',
          ...rest,
          msg_args: messageArgs ? JSON.stringify(messageArgs) : null,
        }))
      }
    }, 200)
  })
  await page.route('**/api/**', async route => {
    const url = new URL(route.request().url())
    const path = url.pathname
    if (!path.startsWith('/api/')) {
      await route.continue()
      return
    }
    if (path === '/api/preferences') {
      if (route.request().method() === 'PATCH') preferences = mergeUiPatch(preferences, route.request().postDataJSON())
      await route.fulfill({ json: preferences })
      return
    }
    if (path === '/api/projects' && route.request().method() === 'POST') {
      const { name } = route.request().postDataJSON() as { name: string }
      const id = `p${createdProjects.length + 3}`
      const project = {
        id, name, created_at: '2026-01-02T00:00:00Z', updated_at: '2026-01-02T00:00:00Z',
        sources: [], state: 'created', ui_state: null,
      }
      createdProjects.push(project)
      uiStates[id] = null
      sourcesByProject[id] = []
      await route.fulfill({ json: project })
      return
    }
    if (path === '/api/projects') {
      const projects = [{
        id: 'p1', name: 'Mock project', created_at: '2026-01-01T00:00:00Z', updated_at: '2026-01-01T00:00:00Z',
        sources: sourcesByProject.p1, state: 'exported',
        ui_state: uiStates.p1,
      }]
      if (options.secondProject) projects.push({
        id: 'p2', name: 'Second project', created_at: '2025-01-01T00:00:00Z', updated_at: '2025-01-01T00:00:00Z',
        sources: sourcesByProject.p2, state: 'created', ui_state: uiStates.p2,
      })
      await route.fulfill({ json: [...projects, ...createdProjects].filter(project => !deletedProjects.has(project.id)) })
      return
    }
    if (path === '/api/settings') {
      await route.fulfill({ json: { sam3: {
        training_prompt: TRAINING_MASK_PROMPT,
        feature_prompt: FEATURE_MASK_PROMPT,
      } } })
      return
    }
    if (path === '/api/system/doctor') {
      await route.fulfill({ json: { ready: true, platform: {}, checks: {
        workspace: { ok: true, message: 'ok', path: MOCK_ENV_PATH },
        colmap: { ok: true, message: 'ok', capabilities: {
          gpu_bundle_adjustment: options.gpuBundleAdjustment ?? false,
        } },
      } } })
      return
    }
    if (path === '/api/system/stats') {
      await route.fulfill({ json: { cpu_percent: 1, ram: null, gpus: [] } })
      return
    }
    if (path === '/api/fs/drives') {
      await route.fulfill({ json: { drives: ['D:\\'] } })
      return
    }
    if (path === '/api/fs/browse') {
      await route.fulfill({ json: { path: 'D:\\', parent: null,
        dirs: [{ name: 'Phone photos', path: MOCK_PHONE, is_dir: true }], files: [] } })
      return
    }
    const projectMatch = path.match(/^\/api\/projects\/([^/]+)/)
    const projectId = projectMatch?.[1]
    if (projectId && path === `/api/projects/${projectId}/run`) {
      pipelines.push(route.request().postDataJSON())
      await route.fulfill({ json: { job_id: 'pipeline-job' } })
      return
    }
    if (path === '/api/jobs/pipeline-job') {
      await route.fulfill({ json: { id: 'pipeline-job', project_id: 'p1', status: 'running' } })
      return
    }
    if (projectId && path === `/api/projects/${projectId}/stages`) {
      stageRequestCount += 1
      if (stageRequestCount === 1 && options.initialStageSnapshotDelayMs)
        await new Promise(resolve => setTimeout(resolve, options.initialStageSnapshotDelayMs))
      if (options.disconnectStagesAfter != null && stageRequestCount > options.disconnectStagesAfter) {
        await route.fulfill({ status: 503, json: { detail: 'backend unavailable' } })
        return
      }
      const names = ['inspect_source', 'extract_frames', 'prepare_images', 'rectify_fisheye', 'generate_feature_masks', 'generate_training_masks', 'extract_features', 'match_features', 'reconstruct', 'align_reconstruction', 'restore_metric_scale', 'scene_alignment', 'cleanup_sparse', 'dense_initialization', 'export_dataset']
      const extras: Record<string, Record<string, unknown>> = {
        inspect_source: { kind: 'insv', file_size: 1024, gravity_samples: 42 },
        extract_frames: { frames: 1, selection_mode: 'interval', selected: 1 },
        prepare_images: { images: 2, sources: 1, camera_groups: 1 },
        rectify_fisheye: { rectified_images: 2, rectified_groups: 1, image_format: 'png' },
        generate_feature_masks: { purpose: 'feature', images: 2, average_dynamic_coverage: 0.2, coverage_warnings: 0 },
        generate_training_masks: { purpose: 'training', images: 2, average_dynamic_coverage: 0.1, coverage_warnings: 0 },
        extract_features: { images: 2, minimum_keypoints: 100, average_keypoints: 200, maximum_keypoints: 300, descriptor_images: 2 },
        match_features: { raw_pairs: 1, verified_pairs: 1, minimum_inliers: 20, average_inliers: 20, maximum_inliers: 20, total_inliers: 20 },
        reconstruct: {
          input_images: 2, num_images: 2, num_points3D: 42, registered_ratio: 1,
          mean_reprojection_error: 0.5,
          primary_trajectory: {
            available: true, passed: false, maximum_to_p95_ratio: 56.6,
            largest_steps: [{ from_capture: 261, to_capture: 262, distance: 62.9 }],
          },
        },
        align_reconstruction: { applied: true, spread_deg: 0.4, preview_points: 42 },
        restore_metric_scale: { applied: true, metric: true, scale_factor: 445, baseline_pairs: 2 },
        scene_alignment: {
          applied: true,
          ground: {
            applied: true, ground_y: 5, support_points: 1200,
            plane_selection: 'nearest_dominant_horizontal_plane', metric_scale_available: false,
            camera_height_median_model_units: 1.36,
          },
          orientation: {
            applied: true, method: 'orthogonal_vertical_planes', yaw_deg: -21.56,
            confidence: 0.985, orthogonality_residual_deg: 4.99,
          },
        },
        cleanup_sparse: { enabled: true, input_points: 42, removed_points: 2, output_points: 40 },
        dense_initialization: { enabled: false, method: 'passthrough', base_points: 42, new_points: 0, total_points: 42 },
        export_dataset: { images: 2, total_points: 42, validation: { loadable: true, training_ready: true }, lfstudio_training_metrics: 'external' },
      }
      await route.fulfill({ json: { project_id: 'p1', state: 'exported', stages: names.map(stage => ({
        stage, has_output: stage === options.runningStage
          ? options.runningHasOutput ?? false
          : !pendingStages.has(stage),
        status: stage === options.runningStage ? 'running' : pendingStages.has(stage) ? null : 'succeeded', error_text: null,
        job_id: stage === options.runningStage ? 'running-job' : null,
        started_at: stage === options.runningStage ? '2026-01-01T00:00:00Z' : null,
        finished_at: null,
        params: stageParamOverrides[stage] ?? options.stageParams?.[stage]
          ?? paramsForStage(stage, DEFAULT_PARAMS, reconMode),
        active_params: stage === options.runningStage ? options.runningActiveParams ?? null : null,
        extra: extras[stage],
        activity_event: stage === options.runningStage ? {
          id: options.runningActivityMessageKey ? 1000 : 999,
          job_id: 'running-job', project_id: 'p1', stage, level: 'info',
          message: 'working', msg_key: options.runningActivityMessageKey ?? options.runningMessageKey ?? null,
          msg_args: options.runningActivityMessageKey
            ? options.runningActivityMessageArgs ?? { source: 'Primary 360', cur: 43, tot: 100 }
            : options.runningMessageKey
              ? options.runningMessageArgs ?? { source: 'Primary 360', cur: 42, tot: 100 }
              : null,
          progress: options.runningActivityMessageKey ? null : options.runningProgress ?? null,
          kind: 'progress', ts: '2026-01-01T00:00:02Z',
        } : null,
        progress_event: stage === options.runningStage && options.runningProgress != null ? {
          id: 999, job_id: 'running-job', project_id: 'p1', stage, level: 'info',
          message: 'working', msg_key: options.runningMessageKey ?? null,
          msg_args: options.runningMessageKey
            ? options.runningMessageArgs ?? { source: 'Primary 360', cur: 42, tot: 100 }
            : null,
          progress: options.runningProgress, kind: 'progress', ts: '2026-01-01T00:00:01Z',
        } : null,
      })) } })
      return
    }
    if (projectId && path === `/api/projects/${projectId}/source-info`) {
      const width = options.sourceWidth ?? 100
      const height = options.sourceHeight ?? 100
      await route.fulfill({ json: { id: 's1', duration_sec: 1, duration_sec_total: 1, fps: 30, width, height,
        sources: [{ id: 's1', duration_sec: 1, fps: 30, width, height }] } })
      return
    }
    if (projectId && path === `/api/projects/${projectId}/reconstruction`) {
      if (!reconstructionAvailable) {
        await route.fulfill({ status: 404, json: { detail: 'reconstruction preview not available' } })
        return
      }
      await route.fulfill({ json: {
        cameras: [{ id: 1, model: options.cameraModel ?? (isInsv ? 'OPENCV_FISHEYE'
          : isPerspectiveImages ? 'SIMPLE_RADIAL' : 'EQUIRECTANGULAR'), width: 100, height: 100, params: [] }],
        images: [{ id: 1, name: imageName, camera_id: 1,
          qvec: [1, 0, 0, 0], tvec: [-1, -2, -3], position: [1, 2, 3], num_points: 42 }],
        stats: { num_cameras: 1, num_images: 1, num_points3D: 42, mean_reprojection_error: 0.5,
          mean_track_length: 2, registered_ratio: 1, camera_trajectory_diameter: 1.25 },
        points_file: 'points.bin', points_stride: 20,
      } })
      return
    }
    if (projectId && path === `/api/projects/${projectId}/reconstruction/points`) {
      const body = Buffer.alloc(8)
      body.writeUInt32LE(0, 0)
      body.writeUInt32LE(20, 4)
      await route.fulfill({ status: 200, contentType: 'application/octet-stream', body })
      return
    }
    if (projectId && path === `/api/projects/${projectId}/frames`) {
      const frameSources = [{
        id: 's1', label: 'Primary 360', role: 'primary',
        projection: isInsv ? 'dual_fisheye' : isPerspectiveImages ? 'perspective' : 'equirectangular',
        kind: isInsv ? 'insv_dual' : isPerspectiveImages ? 'perspective_images' : 'equirectangular_video',
        count: 1, width: 100, height: 100, fps: 30,
        selection: { mode: 'interval', selected: 1 },
      }]
      const frames = Array.from({ length: options.frameCount ?? 1 }, (_, index) => ({
        index, source_id: 's1', source_index: index, timestamp_sec: index / 30,
        score: { sharpness: 12 + index },
      }))
      frameSources[0].count = frames.length
      if (options.multipleFrameSources) {
        frameSources.push({
          id: 's2', label: 'Phone photos', role: 'supplemental', projection: 'perspective',
          kind: 'perspective_images', count: 1, width: 100, height: 100, fps: 0,
          selection: { mode: 'all', selected: 1 },
        })
        frames.push({ index: 1, source_id: 's2', source_index: 0, timestamp_sec: 0, score: { sharpness: 24 } })
      }
      for (const source of options.regionSources?.slice(1) ?? []) {
        const index = frames.length
        frameSources.push({ id: source.id, label: source.label, role: source.role,
          projection: source.projection, kind: source.projection === 'dual_fisheye' ? 'insv_dual' : 'perspective_images',
          count: 1, width: 100, height: 100, fps: 0, selection: { mode: 'all', selected: 1 } })
        frames.push({ index, source_id: source.id, source_index: 0, timestamp_sec: 0, score: { sharpness: 24 } })
      }
      await route.fulfill({ json: { count: frames.length, sources: frameSources, frames } })
      return
    }
    const masksMatch = projectId && path.match(new RegExp(`^/api/projects/${projectId}/masks/(feature|training)$`))
    if (masksMatch) {
      const purpose = masksMatch[1] as 'feature' | 'training'
      maskRequests[purpose] += 1
      if (options.incrementalFeatureMask && purpose === 'training') {
        await route.fulfill({ status: 404, json: { detail: 'training masks not run yet' } })
        return
      }
      const maskNames = isInsv && reconMode !== 'pinhole_rig'
        ? ['sources/s1/lens0/frame_000000.jpg', 'sources/s1/lens1/frame_000000.jpg']
        : [imageName]
      const incrementalPending = options.incrementalFeatureMask
        && purpose === 'feature' && maskRequests.feature === 1
      await route.fulfill({ json: {
        version: 3, purpose, revision: options.incrementalFeatureMask ? 'live-run' : `${purpose}-run`,
        complete: !options.incrementalFeatureMask,
        total_images: maskNames.length,
        generated_images: incrementalPending ? 0 : maskNames.length,
        prompt: (purpose === 'feature' ? FEATURE_MASK_PROMPT : TRAINING_MASK_PROMPT).split(','),
        max_inference_size: 2048, dilate_px: 8,
        images: (incrementalPending ? [] : maskNames).map((name, index) => ({
        name, source_id: 's1', capture_index: 0, path: `${purpose}-mask.png`,
        coverage: (purpose === 'feature' ? 0.2 : 0.1) + index * 0.1,
        coverage_warning: false,
      })) } })
      return
    }
    if (projectId && path === `/api/projects/${projectId}/source-region`) {
      const radius = options.frameCount === 0 ? 0.5 : 0.48
      const sourceId = url.searchParams.get('source_id')!
      const key = `${projectId}:${sourceId}`
      const source = sourcesByProject[projectId].find(item => item.id === sourceId)
      if (route.request().method() === 'PUT') {
        regions[key] = { ...route.request().postDataJSON(), saved: true, needs_review: false }
      }
      await route.fulfill({ json: regions[key] ?? {
        views: source?.projection === 'dual_fisheye' ? {
          lens0: { kind: 'circle', cx: 0.5, cy: 0.5, r: radius, operations: [] },
          lens1: { kind: 'circle', cx: 0.5, cy: 0.5, r: radius, operations: [] },
        } : { main: { kind: 'full', operations: [] } },
        saved: options.regionSaved ?? true, needs_review: source?.projection === 'dual_fisheye',
      } })
      return
    }
    if (projectId && path === `/api/projects/${projectId}/export-info`) {
      await route.fulfill({ json: {
        dir: MOCK_EXPORT,
        gui_integration: {
          train_configs_auto_applied: false,
          warnings: ['lfstudio_gui_does_not_auto_apply_train_configs'],
          required_settings: {
            strategy: 'mrnf', gut: true, undistort: false, mask_mode: 'segment',
            ppisp: false, ppisp_controller: false,
          },
        },
      } })
      return
    }
    if (projectId && path === `/api/projects/${projectId}/clear-outputs`
      && route.request().method() === 'POST') {
      const requested = (route.request().postDataJSON() as { stages: string[] }).stages
      clearRequests.push(requested)
      const names = ['inspect_source', 'extract_frames', 'prepare_images', 'rectify_fisheye', 'generate_feature_masks', 'generate_training_masks', 'extract_features', 'match_features', 'reconstruct', 'align_reconstruction', 'restore_metric_scale', 'scene_alignment', 'cleanup_sparse', 'dense_initialization', 'export_dataset']
      const first = Math.min(...requested.map(stage => names.indexOf(stage)).filter(index => index >= 0))
      const cleared = names.slice(first)
      cleared.forEach(stage => pendingStages.add(stage))
      if (cleared.includes('reconstruct')) reconstructionAvailable = false
      await route.fulfill({ json: { requested, cleared, state: first <= 2 ? 'extracted' : 'reconstructed' } })
      return
    }
    if (path.includes('/image') || path.includes('/prepared-image') || path.includes('/prepared-mask')) {
      const pixel = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M/wHwAF/gL+AvJkGQAAAABJRU5ErkJggg==', 'base64')
      await route.fulfill({ status: 200, contentType: 'image/png', body: pixel })
      return
    }
    if (projectId && path === `/api/projects/${projectId}/ui-state` && route.request().method() === 'PATCH') {
      const ui = (route.request().postDataJSON() as { ui: Record<string, unknown> }).ui
      uiStates[projectId] = mergeUiPatch(uiStates[projectId] ?? {}, ui)
      uiStateUpdates.push({ projectId, ui })
      await route.fulfill({ json: {
        id: projectId, name: projectId === 'p1' ? 'Mock project'
          : projectId === 'p2' ? 'Second project' : createdProjects.find(project => project.id === projectId)?.name,
        sources: sourcesByProject[projectId], state: 'reconstructed',
        created_at: '2026-01-01T00:00:00Z', updated_at: '2026-01-01T00:00:00Z', ui_state: uiStates[projectId],
      } })
      return
    }
    if (projectId && path === `/api/projects/${projectId}/sources` && route.request().method() === 'POST') {
      const body = route.request().postDataJSON() as Record<string, unknown>
      sourceAdds.push(body)
      sourcesByProject[projectId].push({
        id: `s${sourcesByProject[projectId].length + 1}`,
        label: body.label || 'Supplemental source',
        role: sourcesByProject[projectId].length ? 'supplemental' : 'primary',
        adapter: body.adapter,
        media_kind: body.media_kind,
        projection: body.projection,
        path: body.path,
        ordinal: sourcesByProject[projectId].length,
        enabled: true,
      })
      await route.fulfill({ json: { id: projectId, sources: sourcesByProject[projectId] } })
      return
    }
    const sourceMutation = path.match(/^\/api\/projects\/([^/]+)\/sources\/([^/]+)(\/make-primary)?$/)
    if (sourceMutation && route.request().method() === 'DELETE') {
      sourcesByProject[sourceMutation[1]] = sourcesByProject[sourceMutation[1]].filter(source => source.id !== sourceMutation[2])
      await route.fulfill({ json: { id: sourceMutation[1], sources: sourcesByProject[sourceMutation[1]] } })
      return
    }
    if (sourceMutation?.[3] && route.request().method() === 'POST') {
      for (const source of sourcesByProject[sourceMutation[1]])
        source.role = source.id === sourceMutation[2] ? 'primary' : 'supplemental'
      await route.fulfill({ json: { id: sourceMutation[1], sources: sourcesByProject[sourceMutation[1]] } })
      return
    }
    if (projectId && path === `/api/projects/${projectId}` && route.request().method() === 'DELETE') {
      deletedProjects.add(projectId)
      await route.fulfill({ json: { deleted: projectId } })
      return
    }
    const rerun = path.match(/^\/api\/projects\/([^/]+)\/rerun\/([^/]+)$/)
    if (rerun && route.request().method() === 'POST') {
      const body = route.request().postDataJSON() as Record<string, Record<string, unknown>>
      reruns.push({ stage: rerun[2], body })
      stageParamOverrides[rerun[2]] = body.params_by_stage[rerun[2]]
      pendingStages.delete(rerun[2])
      await route.fulfill({ json: { job_id: 'j1' } })
      return
    }
    if (path === '/api/jobs/j1') {
      await route.fulfill({ json: { id: 'j1', project_id: 'p1', kind: 'rerun_stage', stage: reruns.at(-1)?.stage ?? null,
        status: 'succeeded', created_at: '2026-01-01T00:00:00Z', started_at: '2026-01-01T00:00:00Z',
        finished_at: '2026-01-01T00:00:01Z', error_text: null, pid: null } })
      return
    }
    unexpectedRequests.push(`${route.request().method()} ${path}`)
    await route.fulfill({ status: 501, json: { detail: 'unexpected mocked request' } })
  })
  return { unexpectedRequests, reruns, pipelines, sourceAdds, clearRequests, maskRequests, uiStateUpdates,
    getPreferences: () => preferences, uiStates }
}

const inspectScene = async (page: Page) => page.evaluate(async () => {
  const modulePath = '/node_modules/.vite/deps/@react-three_fiber.js'
  const { _roots } = await import(/* @vite-ignore */ modulePath)
  const state = _roots.get(document.querySelector('canvas'))?.store.getState() as RootState | undefined
  if (!state) return null
  state.gl.render(state.scene, state.camera)
  const gl = state.gl.getContext()
  const pixels = new Uint8Array(gl.drawingBufferWidth * gl.drawingBufferHeight * 4)
  gl.readPixels(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight, gl.RGBA, gl.UNSIGNED_BYTE, pixels)
  const colors = new Set<number>()
  for (let index = 0; index < pixels.length; index += 16)
    colors.add((pixels[index] << 16) | (pixels[index + 1] << 8) | pixels[index + 2])
  return {
    position: state.camera.position.toArray(), quaternion: state.camera.quaternion.toArray(),
    colors: colors.size, geometries: state.gl.info.memory.geometries,
  }
})

test('server preferences restore layout, display and project state without browser storage', async ({ page }) => {
  const mock = await installUiMock(page)
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  await page.addInitScript(() => {
    Storage.prototype.getItem = () => { throw new Error('browser storage must not be read') }
    Storage.prototype.setItem = () => { throw new Error('browser storage must not be written') }
  })
  await page.goto('/')
  await expect(page.getByRole('tab', { name: 'Scene View' })).toHaveAttribute('aria-selected', 'true')
  await page.getByRole('checkbox', { name: 'Show cameras' }).uncheck()
  await page.getByRole('button', { name: /^Frame extraction/ }).click()
  await expect(page.getByRole('tab', { name: 'Inspector' }).locator('[data-tab]')).toHaveClass('dock-tab-ping')
  await page.getByRole('slider', { name: /Sharpness threshold/ }).fill('240')
  await expect.poll(() => (mock.uiStates.p1?.params as Record<string, unknown>)?.minSharpness).toBe(240)
  await expect.poll(() => (mock.getPreferences().layout as object | null) !== null).toBe(true)
  await page.reload()
  await expect(page.getByRole('tab', { name: 'Inspector' })).toHaveAttribute('aria-selected', 'true')
  await expect(page.getByRole('slider', { name: /Sharpness threshold/ })).toHaveValue('240')
  await expect(page.getByRole('checkbox', { name: 'Show cameras' })).not.toBeChecked()
  expect(errors).toEqual([])
  expect(mock.unexpectedRequests).toEqual([])
})

for (const viewport of [{ width: 1440, height: 900 }, { width: 390, height: 844 }]) {
  test(`empty scene is rendered and interactive at ${viewport.width}px`, async ({ page }, testInfo) => {
    await page.setViewportSize(viewport)
    const mock = await installUiMock(page, { emptyScene: true })
    await page.goto('/')
    await expect(page.locator('canvas')).toBeVisible()
    await expect.poll(async () => (await inspectScene(page))?.colors ?? 0).toBeGreaterThan(20)
    const canvas = page.locator('canvas')
    const box = (await canvas.boundingBox())!
    expect(box.width).toBeGreaterThan(300)
    expect(box.height).toBeGreaterThan(300)
    const before = (await inspectScene(page))!
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2)
    await page.mouse.wheel(0, 100)
    await expect.poll(async () => (await inspectScene(page))?.position).not.toEqual(before.position)
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
    await page.screenshot({ path: testInfo.outputPath(`empty-${viewport.width}.png`) })
    expect(mock.unexpectedRequests).toEqual([])
  })
}

test('point refresh keeps the Canvas and camera even when reconstruction metadata is unchanged', async ({ page }) => {
  const mock = await installUiMock(page)
  let revision = 0
  let pointFetches = 0
  await page.route('**/reconstruction/points', async route => {
    pointFetches++
    const body = Buffer.alloc(8 + 3 * 20)
    body.writeUInt32LE(3, 0)
    body.writeUInt32LE(20, 4)
    for (let point = 0; point < 3; point++) {
      body.writeFloatLE(point + revision * 100, 8 + point * 20)
      body.writeFloatLE(point % 2, 12 + point * 20)
      body.writeFloatLE(point, 16 + point * 20)
      body.fill(180, 20 + point * 20, 23 + point * 20)
    }
    await route.fulfill({ contentType: 'application/octet-stream', body })
  })
  await page.goto('/')
  await expect.poll(() => pointFetches).toBeGreaterThan(0)
  await expect.poll(async () => (await inspectScene(page))?.colors ?? 0).toBeGreaterThan(20)
  const canvasHandle = await page.locator('canvas').elementHandle()
  const box = (await page.locator('canvas').boundingBox())!
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2)
  await page.mouse.wheel(0, 200)
  await expect.poll(() => mock.uiStates.p1?.cameraPose).toBeTruthy()
  const pose = (await inspectScene(page))!
  const initialFetches = pointFetches
  revision++
  await page.getByRole('button', { name: /^Sparse noise cleanup/ }).click()
  await page.getByRole('button', { name: 'Regenerate', exact: true }).click()
  await expect.poll(() => mock.reruns.length).toBe(1)
  await page.getByRole('tab', { name: 'Scene View' }).click()
  await expect.poll(() => pointFetches, { timeout: 10_000 }).toBeGreaterThan(initialFetches)
  await expect(page.getByRole('tab', { name: 'Scene View' }).locator('[data-tab]')).toHaveClass('dock-tab-ping')
  expect(await canvasHandle!.evaluate(canvas => canvas === document.querySelector('canvas'))).toBe(true)
  const refreshed = (await inspectScene(page))!
  expect(refreshed.position).toEqual(pose.position)
  expect(refreshed.quaternion).toEqual(pose.quaternion)
  await page.getByRole('tab', { name: 'Inspector' }).click()
  await page.keyboard.press('KeyF')
  await page.keyboard.down('KeyW')
  await page.waitForTimeout(150)
  await page.keyboard.up('KeyW')
  await page.getByRole('tab', { name: 'Scene View' }).click()
  expect((await inspectScene(page))!.position).toEqual(pose.position)
  expect(mock.unexpectedRequests).toEqual([])
})

test('project switches restore independent saved camera poses without cross-project writes', async ({ page }) => {
  const poseA = { position: [11, 12, 13], quaternion: [0, 0, 0, 1] }
  const poseB = { position: [21, 22, 23], quaternion: [0, 0, 0, 1] }
  const mock = await installUiMock(page, { secondProject: true,
    savedUiStates: { p1: { cameraPose: poseA }, p2: { cameraPose: poseB } } })
  await page.goto('/')
  await expect.poll(async () => (await inspectScene(page))?.position).toEqual(poseA.position)
  await page.getByRole('button', { name: /Projects/ }).click()
  await page.getByText('Second project', { exact: true }).click()
  await expect.poll(async () => (await inspectScene(page))?.position).toEqual(poseB.position)
  await page.getByRole('button', { name: /Projects/ }).click()
  await page.getByText('Mock project', { exact: true }).click()
  await expect.poll(async () => (await inspectScene(page))?.position).toEqual(poseA.position)
  expect(mock.uiStates.p1?.cameraPose).toEqual(poseA)
  expect(mock.uiStates.p2?.cameraPose).toEqual(poseB)
  expect(mock.unexpectedRequests).toEqual([])
})

test('IDE loads the split pipeline and environment diagnostics', async ({ page }) => {
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  const mock = await installUiMock(page)
  await page.goto('/')
  await expect(page.locator('h1')).toBeVisible()
  await expect(page.getByText('Extract features', { exact: true })).toBeVisible()
  await expect(page.getByText('Match features', { exact: true })).toBeVisible()
  await expect(page.getByText('Gravity alignment', { exact: true })).toBeVisible()
  await expect(page.getByText('Dense initialization', { exact: true })).toBeVisible()
  await expect(page.getByText('Export', { exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: /Internal fisheye normalization/ })).toBeVisible()

  await page.getByTitle('Settings', { exact: true }).click()
  await expect(page.getByText(/Environment [✓⚠]/)).toBeVisible()
  const popBox = await page.locator('.pop').boundingBox()
  const environmentPath = page.locator('.environment-check .path-text')
  const pathBox = await environmentPath.boundingBox()
  expect(popBox).not.toBeNull()
  expect(pathBox).not.toBeNull()
  expect(pathBox!.x + pathBox!.width).toBeLessThanOrEqual(popBox!.x + popBox!.width + 0.5)
  expect(await environmentPath.locator('.path-value').evaluate(
    element => element.scrollWidth > element.clientWidth,
  )).toBe(true)
  expect(errors).toEqual([])
  expect(mock.unexpectedRequests).toEqual([])
})

test('unchecked and checked checkboxes remain visibly distinct', async ({ page }) => {
  const mock = await installUiMock(page)
  await page.goto('/')

  const checkbox = page.getByRole('checkbox', { name: 'Show point cloud' })
  await checkbox.uncheck()
  const unchecked = await checkbox.evaluate(element => {
    const style = getComputedStyle(element)
    return {
      borderColor: style.borderTopColor,
      borderStyle: style.borderTopStyle,
      borderWidth: style.borderTopWidth,
      backgroundImage: style.backgroundImage,
    }
  })
  expect(unchecked.borderStyle).toBe('solid')
  expect(unchecked.borderWidth).toBe('1px')
  expect(unchecked.borderColor).not.toBe('rgba(0, 0, 0, 0)')
  expect(unchecked.backgroundImage).toBe('none')

  await checkbox.check()
  await expect(checkbox).toBeChecked()
  expect(await checkbox.evaluate(element => getComputedStyle(element).backgroundImage)).toContain('svg')
  expect(mock.unexpectedRequests).toEqual([])
})

test('running stage restores numeric progress and activity from the stage snapshot', async ({ page }) => {
  const mock = await installUiMock(page, {
    runningStage: 'extract_frames',
    runningProgress: 0.42,
    runningMessageKey: 'log.extract_candidates_progress',
    runningActivityMessageKey: 'log.extract_scoring_progress',
  })
  await page.goto('/')

  const stage = page.getByRole('button', { name: /Frame extraction 42%/ })
  await expect(stage).toBeVisible()
  await expect(stage.getByRole('progressbar', { name: 'Frame extraction' }))
    .toHaveAttribute('aria-valuenow', '42')
  await stage.click()
  await expect(page.getByText('Primary 360: candidate decode 42/100', { exact: true })).toBeVisible()
  await expect(page.getByText('Primary 360: candidate scoring 43/100', { exact: true })).toBeVisible()
  expect(mock.unexpectedRequests).toEqual([])
})

test('running stage compares settings with the active request instead of the previous artifact', async ({ page }) => {
  const activeParams = paramsForStage('extract_frames', DEFAULT_PARAMS, 'native_fisheye')
  const mock = await installUiMock(page, {
    runningStage: 'extract_frames',
    runningHasOutput: true,
    runningActiveParams: activeParams,
    stageParams: {
      extract_frames: { ...activeParams, target_motion: 8 },
    },
  })
  await page.goto('/')

  const stage = page.getByRole('button', { name: /^Frame extraction/ })
  await expect(stage).not.toContainText('target_motion')
  await stage.click()

  await page.getByRole('slider', { name: 'Motion spacing' }).fill('3')
  await expect(stage).toContainText('Pending settings: target_motion: 2 → 3')
  expect(mock.unexpectedRequests).toEqual([])
})

test('fisheye normalization is a visible step with detailed progress', async ({ page }) => {
  const mock = await installUiMock(page, {
    runningStage: 'rectify_fisheye',
    runningProgress: 0.156,
    runningMessageKey: 'log.rectify_image_progress',
    runningMessageArgs: { cur: 351, tot: 2976 },
  })
  await page.goto('/')

  const stage = page.getByRole('button', { name: /Internal fisheye normalization 16%/ })
  await expect(stage).toBeVisible()
  await expect(stage).toContainText('rectifying fisheye PNG 351/2976')
  await stage.click()
  await expect(page.locator('.mono').getByText('rectifying fisheye PNG 351/2976', { exact: true })).toBeVisible()
  expect(mock.unexpectedRequests).toEqual([])
})

test('backend disconnect expires a stale running snapshot instead of extending elapsed time', async ({ page }) => {
  const mock = await installUiMock(page, {
    runningStage: 'extract_frames',
    runningProgress: 0.42,
    runningMessageKey: 'log.extract_candidates_progress',
    disconnectStagesAfter: 1,
  })
  await page.goto('/')
  await expect(page.getByRole('button', { name: /Frame extraction 42%/ })).toBeVisible()

  await expect(page.getByText(
    'The backend is unreachable. Displayed progress is a stale snapshot and is not still running.',
    { exact: true },
  )).toBeVisible({ timeout: 12_000 })
  await expect(page.getByRole('button', { name: /Frame extraction Backend disconnected/ })).toBeVisible()
  await expect(page.getByRole('progressbar', { name: 'Frame extraction' })).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Stop' })).toHaveCount(0)
  expect(mock.unexpectedRequests).toEqual([])
})

test('running stage shows indeterminate progress when only activity is known', async ({ page }) => {
  const mock = await installUiMock(page, {
    runningStage: 'extract_frames',
    runningProgress: null,
  })
  await page.goto('/')

  const stage = page.getByRole('button', { name: /Frame extraction working/ })
  const progress = stage.getByRole('progressbar', { name: 'Frame extraction' })
  await expect(progress).toBeVisible()
  await expect(progress).not.toHaveAttribute('aria-valuenow')
  const spinner = progress.locator('g.progress-ring-indeterminate')
  await expect(spinner).toHaveCount(1)
  await expect(spinner.locator('circle')).toHaveAttribute('transform', /rotate\(-90/)
  const animation = await spinner.evaluate(element => {
    const style = getComputedStyle(element)
    return { name: style.animationName, transformBox: style.transformBox, origin: style.transformOrigin }
  })
  expect(animation.name).toBe('progress-ring-spin')
  expect(animation.transformBox).toBe('view-box')
  expect(mock.unexpectedRequests).toEqual([])
})

test('incremental mapper shows rig frame count and global refinement at the same time', async ({ page }) => {
  const mock = await installUiMock(page, {
    runningStage: 'reconstruct',
    runningProgress: 0.899,
    runningMessageKey: 'log.recon_mapper_progress',
    runningMessageArgs: { done: 1487, total: 1488 },
    runningActivityMessageKey: 'log.recon_global_refinement',
    runningActivityMessageArgs: { pass: 9, done: 1487, total: 1488 },
  })
  await page.goto('/')

  const stage = page.getByRole('button', { name: /Sparse reconstruction 90%/ })
  await expect(stage).toContainText('mapper: registered frames 1487/1488')
  await expect(stage).toContainText('global refinement 9: registered frames 1487/1488')
  await stage.click()
  await expect(page.getByText('mapper: registered frames 1487/1488', { exact: true })).toBeVisible()
  await expect(page.getByText('global refinement 9: registered frames 1487/1488', { exact: true })).toBeVisible()
  expect(mock.unexpectedRequests).toEqual([])
})

for (const initialStageSnapshotDelayMs of [0, 800]) {
  test(`activity-only events retain percentage and late events from an old job are ignored${
    initialStageSnapshotDelayMs ? ' before the initial snapshot' : ''
  }`, async ({ page }) => {
    const mock = await installUiMock(page, {
      initialStageSnapshotDelayMs,
      runningStage: 'extract_frames',
      runningProgress: 0.42,
      runningMessageKey: 'log.extract_candidates_progress',
      socketEvents: [
        {
          id: 1000, job_id: 'running-job', stage: 'extract_frames', progress: null,
          message: 'scoring', msg_key: 'log.extract_scoring_progress',
          msg_args: { source: 'Primary 360', cur: 43, tot: 100 },
        },
        {
          id: 1001, job_id: 'old-job', stage: 'extract_frames', progress: 0.9,
          message: 'old', msg_key: null, msg_args: null,
        },
      ],
    })
    await page.goto('/')

    await expect(page.getByRole('button', { name: /Frame extraction 42%/ })).toBeVisible()
    await page.getByRole('button', { name: /Frame extraction 42%/ }).click()
    await expect(page.getByText('Primary 360: candidate scoring 43/100', { exact: true })).toBeVisible()
    await expect(page.getByText('old', { exact: true })).toHaveCount(0)
    expect(mock.unexpectedRequests).toEqual([])
  })
}

test('console warning icon follows the log text size', async ({ page }) => {
  const mock = await installUiMock(page, {
    socketEvents: [{
      id: 1000, job_id: null, stage: 'extract_frames', level: 'warn', kind: 'log',
      message: 'warning icon size check', msg_key: null, msg_args: null,
    }],
  })
  await page.goto('/')
  await page.getByText('Console', { exact: true }).first().click()

  const row = page.locator('.con-log > div').filter({ hasText: 'warning icon size check' })
  await expect(row).toBeVisible()
  const sizes = await row.evaluate(element => {
    const icon = element.querySelector('.log-level svg') as SVGElement
    return {
      text: Number.parseFloat(getComputedStyle(element).fontSize),
      icon: icon.getBoundingClientRect().height,
    }
  })
  expect(sizes.icon).toBeCloseTo(sizes.text, 1)
  expect(mock.unexpectedRequests).toEqual([])
})

test('generate all submits one server-owned pipeline including fisheye normalization', async ({ page }) => {
  const mock = await installUiMock(page, {
    pendingStages: ['prepare_images', 'rectify_fisheye', 'generate_feature_masks'],
  })
  await page.goto('/')

  await page.getByRole('button', { name: 'Generate all pending steps' }).click()
  await expect.poll(() => mock.pipelines.length).toBe(1)
  expect(mock.pipelines[0].params_by_stage).toHaveProperty('rectify_fisheye')
  expect(mock.pipelines[0].params_by_stage).toHaveProperty('export_dataset')
  expect(mock.reruns).toEqual([])
  await expect(page.getByRole('button', { name: /Internal fisheye normalization/ })).toBeVisible()
  expect(mock.unexpectedRequests).toEqual([])
})

test('scene copy follows language and paths remain copyable native values', async ({ page, context }) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write'])
  const mock = await installUiMock(page)
  await page.goto('/')

  await expect(page.locator('.scene-toolbar')).not.toContainText('Right-drag')
  const sceneInfo = page.locator('.scene-info')
  await expect(sceneInfo).toContainText('Images 1')
  await expect(sceneInfo).toContainText('Points 42')
  await expect(sceneInfo).toContainText('Registered 100%')
  await expect(sceneInfo).toContainText('Path span 1.250 units')

  const sourcePath = page.locator('.ide-source-path')
  await expect(sourcePath.locator('.path-value')).toHaveText('D:/VID 2026/clip.insv')
  await expect(sourcePath).toHaveAttribute('data-copy-value', MOCK_SOURCE)
  await sourcePath.locator('.path-copy').click()
  await expect(sourcePath.locator('.path-copy')).toHaveClass(/copied/)
  await expect(sourcePath.getByRole('status')).toHaveText('Path copied')
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(MOCK_SOURCE)

  await page.getByText('Export', { exact: true }).click()
  await expect(page.getByText('D:/LFStudio/export_dataset', { exact: true })).toHaveCount(1)
  await expect(page.getByText(/MRNF · GUT=true · mask=segment · PPISP=false · controller=false/)).toBeVisible()
  const statistics = page.getByRole('button', { name: 'Stage statistics', exact: true })
  await expect(statistics).toHaveAttribute('aria-expanded', 'false')
  await expect(page.getByText('LFStudio training loss / PSNR / SSIM', { exact: true })).toHaveCount(0)
  await statistics.click()
  await expect(statistics).toHaveAttribute('aria-expanded', 'true')
  await expect(page.getByText('LFStudio training loss / PSNR / SSIM', { exact: true })).toBeVisible()
  await expect(page.getByText(/unavailable \(LFStudio training and its output directory are externally managed\)/)).toBeVisible()
  const displayedPaths = (await page.locator('.path-value').allTextContents()).join('\n')
  expect(displayedPaths).not.toContain('¥')
  expect(displayedPaths).not.toContain('\\')

  await page.getByTitle('Settings', { exact: true }).click()
  await page.locator('.pop select').nth(1).selectOption('zh')
  await page.mouse.click(10, 200)
  await page.getByRole('tab', { name: '场景视图' }).click()
  await expect(sceneInfo).toContainText('图像 1')
  await expect(sceneInfo).toContainText('点 42')
  await expect(sceneInfo).toContainText('注册 100%')
  await expect(sceneInfo).toContainText('轨迹范围 1.250 单位')
  await expect(page.getByText('场景视图', { exact: true }).first()).toBeVisible()
  expect(mock.unexpectedRequests).toEqual([])
})

test('photo source groups collapse independently', async ({ page }) => {
  const mock = await installUiMock(page, { multipleFrameSources: true })
  await page.goto('/')

  const photosToggle = page.getByRole('button', { name: /Photos/ })
  await expect(photosToggle).toHaveAttribute('aria-expanded', 'false')
  await expect(page.locator('[data-source-id]')).toHaveCount(0)
  await photosToggle.click()

  const primaryGroup = page.locator('[data-source-id="s1"]')
  const phoneGroup = page.locator('[data-source-id="s2"]')
  const primaryToggle = primaryGroup.getByRole('button', { name: /Primary 360/ })
  const phoneToggle = phoneGroup.getByRole('button', { name: /Phone photos/ })

  await expect(primaryToggle).toHaveAttribute('aria-expanded', 'true')
  await expect(phoneToggle).toHaveAttribute('aria-expanded', 'true')
  await expect(primaryGroup.getByRole('button', { name: /Frame 0/ })).toBeVisible()
  await expect(phoneGroup.getByRole('button', { name: /Frame 0/ })).toBeVisible()

  await primaryToggle.click()
  await expect(primaryToggle).toHaveAttribute('aria-expanded', 'false')
  await expect(primaryGroup.getByRole('button', { name: /Frame 0/ })).toHaveCount(0)
  await expect(phoneGroup.getByRole('button', { name: /Frame 0/ })).toBeVisible()

  await phoneToggle.click()
  await expect(phoneToggle).toHaveAttribute('aria-expanded', 'false')
  await primaryToggle.click()
  await expect(primaryGroup.getByRole('button', { name: /Frame 0/ })).toBeVisible()
  await expect(phoneGroup.getByRole('button', { name: /Frame 0/ })).toHaveCount(0)
  expect(mock.unexpectedRequests).toEqual([])
})

test('clear outputs dialog supports custom dependency-aware selection and removes Scene View residue', async ({ page }) => {
  const mock = await installUiMock(page)
  await page.goto('/')
  await expect(page.locator('.scene-info')).toBeVisible()

  await page.getByRole('button', { name: 'Clear outputs' }).click()
  const dialog = page.getByRole('dialog', { name: 'Select outputs to clear' })
  await expect(dialog).toBeVisible()
  await expect(dialog.getByTestId('clear-all-outputs')).toBeVisible()
  await expect(dialog.getByTestId('clear-after-frames')).toBeVisible()

  await dialog.getByRole('checkbox', { name: /Sparse reconstruction/ }).check()
  await expect(dialog.getByRole('checkbox', { name: /Export/ })).toBeChecked()
  await dialog.getByRole('button', { name: 'Cancel' }).click()
  await expect.poll(() => mock.uiStateUpdates.at(-1)?.ui.clearOutputStages).toEqual(['reconstruct'])

  await page.reload()
  await page.getByRole('button', { name: 'Clear outputs' }).click()
  const restoredDialog = page.getByRole('dialog', { name: 'Select outputs to clear' })
  await expect(restoredDialog.getByRole('checkbox', { name: /Sparse reconstruction/ })).toBeChecked()
  await expect(restoredDialog.getByRole('checkbox', { name: /Export/ })).toBeChecked()
  await restoredDialog.getByTestId('clear-selected-outputs').click()

  await expect.poll(() => mock.clearRequests).toEqual([['reconstruct']])
  await expect(dialog).toHaveCount(0)
  await expect(page.locator('.scene-info')).toHaveCount(0)
  expect(mock.unexpectedRequests).toEqual([])
})

test('clear outputs dialog offers all-output and keep-frames shortcuts', async ({ page }) => {
  const mock = await installUiMock(page)
  await page.goto('/')

  await page.getByRole('button', { name: 'Clear outputs' }).click()
  await page.getByTestId('clear-after-frames').click()
  await expect.poll(() => mock.clearRequests).toEqual([['prepare_images']])

  await page.getByRole('button', { name: 'Clear outputs' }).click()
  await page.getByTestId('clear-all-outputs').click()
  await expect.poll(() => mock.clearRequests).toEqual([['prepare_images'], ['inspect_source']])
  expect(mock.unexpectedRequests).toEqual([])
})

test('middle mouse drag activates Scene View panning', async ({ page }) => {
  const mock = await installUiMock(page)
  await page.goto('/')

  const canvas = page.locator('canvas').first()
  await expect(canvas).toBeVisible()
  const bounds = await canvas.boundingBox()
  expect(bounds).not.toBeNull()
  await page.mouse.move(bounds!.x + bounds!.width / 2, bounds!.y + bounds!.height / 2)
  await page.mouse.down({ button: 'middle' })
  await expect(canvas).toHaveCSS('cursor', 'grabbing')
  await page.mouse.move(bounds!.x + bounds!.width / 2 + 80, bounds!.y + bounds!.height / 2 + 40, { steps: 4 })
  await page.mouse.up({ button: 'middle' })
  await expect(canvas).not.toHaveCSS('cursor', 'grabbing')
  expect(mock.unexpectedRequests).toEqual([])
})

test('mouse wheel adjusts Scene View movement speed with centered feedback', async ({ page }) => {
  const mock = await installUiMock(page)
  await page.goto('/')

  const canvas = page.locator('canvas').first()
  const bounds = await canvas.boundingBox()
  expect(bounds).not.toBeNull()
  await page.mouse.move(bounds!.x + bounds!.width / 2, bounds!.y + bounds!.height / 2)

  await page.mouse.wheel(0, 100)
  await expect(page.locator('.scene-speed-feedback')).toHaveCount(0)
  await page.mouse.down({ button: 'right' })
  await page.mouse.wheel(0, 100)
  const feedback = page.locator('.scene-speed-feedback')
  await expect(feedback).toHaveText('0.5x')
  await expect(async () => {
    const canvasBounds = await canvas.boundingBox()
    const feedbackBounds = await feedback.boundingBox()
    expect(canvasBounds).not.toBeNull()
    expect(feedbackBounds).not.toBeNull()
    expect(Math.abs(feedbackBounds!.x + feedbackBounds!.width / 2 - (canvasBounds!.x + canvasBounds!.width / 2))).toBeLessThan(3)
    expect(Math.abs(feedbackBounds!.y + feedbackBounds!.height / 2 - (canvasBounds!.y + canvasBounds!.height / 2))).toBeLessThan(3)
  }).toPass()

  await page.mouse.wheel(0, -100)
  await expect(feedback).toHaveText('1x')
  await page.mouse.wheel(0, -100)
  await expect(feedback).toHaveText('2x')
  await page.mouse.up({ button: 'right' })
  expect(mock.unexpectedRequests).toEqual([])
})

test('stop buttons keep white labels and icons in dark theme', async ({ page }) => {
  const mock = await installUiMock(page, { runningStage: 'extract_frames', runningProgress: 0.42 })
  await page.goto('/')

  await page.getByTitle('Settings', { exact: true }).click()
  await page.locator('.pop select').first().selectOption('dark')
  const stopButtons = page.getByRole('button', { name: 'Stop' })
  await expect(stopButtons).toHaveCount(1)
  for (const stopButton of await stopButtons.all()) {
    await expect(stopButton).toHaveCSS('color', 'rgb(255, 255, 255)')
    const icon = stopButton.locator('svg')
    if (await icon.count()) await expect(icon).toHaveCSS('color', 'rgb(255, 255, 255)')
  }
  expect(mock.unexpectedRequests).toEqual([])
})

test('Scene View background follows the applied theme without one-step lag', async ({ page }) => {
  const mock = await installUiMock(page)
  await page.goto('/')

  const background = page.locator('.scene-toolbar input[type="color"]')
  await page.getByTitle('Settings', { exact: true }).click()
  const theme = page.locator('.pop select').first()

  await theme.selectOption('light')
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light')
  await expect(background).toHaveValue('#fdfdfd')
  await theme.selectOption('dark')
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark')
  await expect(background).toHaveValue('#111821')
  await theme.selectOption('light')
  await expect(background).toHaveValue('#fdfdfd')
  expect(mock.unexpectedRequests).toEqual([])
})

test('source region keeps a fixed fisheye center and paints sensor-local custom regions', async ({ page }) => {
  const mock = await installUiMock(page, { frameCount: 3 })
  await page.goto('/')

  await page.getByRole('button', { name: /Source valid region/ }).click()
  await expect(page.getByText('Coordinate update: review and save the circle')).toBeVisible()
  const previewFrame = page.getByRole('slider', { name: 'Preview frame' })
  await expect(previewFrame).toHaveAttribute('max', '2')
  await previewFrame.fill('2')
  await expect(page.locator('img[alt="lens0"]')).toHaveAttribute('src', /frames\/2\/image/)
  const radius = page.getByRole('slider', { name: 'Valid-circle radius' })
  await expect(radius).toHaveValue('0.48')
  await expect(radius).toHaveAttribute('max', '0.5')
  await radius.fill('0.4')
  await page.getByRole('button', { name: 'Discard staged changes', exact: true }).click()
  await expect(radius).toHaveValue('0.48')
  const editor = page.locator('svg').filter({ has: page.locator('mask') }).first()
  const circleBounds = (await editor.boundingBox())!
  await editor.click({ position: { x: circleBounds.width * 0.94, y: circleBounds.height * 0.94 } })
  await expect(radius).toHaveValue('0.5')
  await page.getByRole('button', { name: 'Discard staged changes', exact: true }).click()
  await expect(radius).toHaveValue('0.48')
  await page.getByRole('button', { name: 'Keep brush (+)', exact: true }).click()
  await editor.evaluate(element => element.scrollIntoView({ block: 'center' }))
  const bounds = await editor.boundingBox()
  expect(bounds).not.toBeNull()
  await page.mouse.move(bounds!.x + bounds!.width * 0.7, bounds!.y + bounds!.height * 0.5)
  const preview = page.getByTestId('source-region-brush-preview')
  await expect(preview).toBeVisible()
  expect(Number(await preview.getAttribute('cx'))).toBeCloseTo(0.7, 5)
  await page.mouse.down()
  await page.mouse.move(bounds!.x + bounds!.width * 0.8, bounds!.y + bounds!.height * 0.5)
  await page.mouse.up()
  await expect(page.getByText('Operations: 2', { exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Undo last stroke', exact: true }).click()
  await expect(page.getByText('Operations: 0', { exact: true })).toBeVisible()
  await editor.evaluate(element => element.scrollIntoView({ block: 'center' }))
  const nextBounds = await editor.boundingBox()
  expect(nextBounds).not.toBeNull()
  await page.mouse.move(nextBounds!.x + nextBounds!.width * 0.75, nextBounds!.y + nextBounds!.height * 0.5)
  await page.mouse.down()
  await page.mouse.move(nextBounds!.x + nextBounds!.width * 0.85, nextBounds!.y + nextBounds!.height * 0.5)
  await page.mouse.up()
  await expect(page.getByText('Operations: 2', { exact: true })).toBeVisible()
  const saveRequest = page.waitForRequest(request => (
    request.method() === 'PUT' && request.url().includes('/source-region')
  ))
  await page.getByRole('button', { name: 'Save', exact: true }).click()
  const payload = (await saveRequest).postDataJSON() as {
    views: { lens0: { cx: number; cy: number; operations: Array<{ mode: string; stroke_id: number }> } }
  }
  expect(payload.views.lens0.cx).toBe(0.5)
  expect(payload.views.lens0.cy).toBe(0.5)
  expect(payload.views.lens0.operations).toHaveLength(2)
  expect(payload.views.lens0.operations[0].mode).toBe('add')
  expect(payload.views.lens0.operations[0].stroke_id).toBe(payload.views.lens0.operations[1].stroke_id)
  expect(mock.unexpectedRequests).toEqual([])
})

test('source region exposes the default fisheye radius before frame extraction', async ({ page }) => {
  const mock = await installUiMock(page, { frameCount: 0 })
  await page.goto('/')

  await page.getByRole('button', { name: /Source valid region/ }).click()
  await expect(page.getByRole('slider', { name: 'Valid-circle radius' })).toHaveValue('0.5')
  await expect(page.getByTestId('source-region-no-preview')).toContainText('Run frame extraction')
  await expect(page.locator('img[alt="lens0"]')).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Save', exact: true })).toBeVisible()
  expect(mock.unexpectedRequests).toEqual([])
})

test('source region switches independent fisheye and phone drafts and restores saved strokes', async ({ page }, testInfo) => {
  const source = (id: string, label: string, projection: ProjectSource['projection']): ProjectSource => ({
    id, label, projection, role: id === 's1' ? 'primary' : 'supplemental',
    adapter: projection === 'dual_fisheye' ? 'insta360' : 'generic_video',
    media_kind: 'video', path: `${id}.mov`, ordinal: Number(id.slice(1)), enabled: true,
  })
  const mock = await installUiMock(page, { regionSources: [
    source('s1', 'Primary 360', 'dual_fisheye'),
    source('s2', 'Second 360', 'dual_fisheye'),
    source('s3', 'Phone video', 'perspective'),
  ] })
  await page.goto('/')
  const phoneImage = await page.evaluate(() => {
    const canvas = document.createElement('canvas')
    canvas.width = 320
    canvas.height = 180
    const context = canvas.getContext('2d')!
    context.fillStyle = '#90b5bc'
    context.fillRect(0, 0, 320, 90)
    context.fillStyle = '#51645b'
    context.fillRect(0, 90, 320, 90)
    return canvas.toDataURL('image/png').split(',')[1]
  })
  await page.route('**/frames/2/image?*', route => route.fulfill({
    contentType: 'image/png', body: Buffer.from(phoneImage, 'base64'),
  }))
  await page.getByRole('button', { name: /Source valid region/ }).click()
  const selector = page.getByRole('combobox', { name: 'Source', exact: true })
  const radius = page.getByRole('slider', { name: 'Valid-circle radius' })
  await radius.fill('0.4')
  await selector.selectOption('s2')
  await expect(radius).toHaveValue('0.48')
  await expect(page.locator('img[alt="lens0"]')).toHaveAttribute('src', /frames\/1\/image/)
  await page.getByRole('combobox', { name: 'Sensor' }).selectOption('lens1')
  await radius.fill('0.42')
  await selector.selectOption('s3')
  await expect(radius).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Base circle', exact: true })).toHaveCount(0)
  await expect(page.getByRole('combobox', { name: 'Sensor' })).toHaveCount(0)
  await expect(page.locator('img[alt="main"]')).toHaveAttribute('src', /frames\/2\/image/)
  const editor = page.getByTestId('source-region-canvas')
  await expect(editor).toHaveAttribute('viewBox', '0 0 1 0.5625')
  const stroke = async () => {
    await editor.evaluate(element => element.scrollIntoView({ block: 'center' }))
    const bounds = (await editor.boundingBox())!
    expect(bounds.width / bounds.height).toBeCloseTo(16 / 9, 2)
    await page.mouse.move(bounds.x + bounds.width * 0.3, bounds.y + bounds.height * 0.5)
    const preview = page.getByTestId('source-region-brush-preview')
    await expect(preview).toBeVisible()
    const brush = (await preview.boundingBox())!
    expect(brush.width).toBeCloseTo(brush.height, 2)
    await page.mouse.down()
    await page.mouse.move(bounds.x + bounds.width * 0.4, bounds.y + bounds.height * 0.5)
    await page.mouse.up()
  }
  await stroke()
  await expect(page.getByText('Operations: 2', { exact: true })).toBeVisible()
  await selector.selectOption('s1')
  await expect(radius).toHaveValue('0.4')
  await selector.selectOption('s2')
  await page.getByRole('combobox', { name: 'Sensor' }).selectOption('lens1')
  await expect(radius).toHaveValue('0.42')
  await selector.selectOption('s3')
  await expect(page.getByText('Operations: 2', { exact: true })).toBeVisible()
  const saveRequest = page.waitForRequest(request => request.method() === 'PUT' && request.url().includes('/source-region'))
  await page.getByRole('button', { name: 'Save', exact: true }).click()
  const request = await saveRequest
  expect(new URL(request.url()).searchParams.get('source_id')).toBe('s3')
  const saved = request.postDataJSON() as SourceRegion
  expect(Object.keys(saved.views)).toEqual(['main'])
  expect(saved.views.main.kind).toBe('full')
  expect(saved.views.main.operations.every(operation => operation.mode === 'subtract')).toBe(true)
  await selector.selectOption('s1')
  await selector.selectOption('s3')
  await expect(page.getByText('Operations: 2', { exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Undo last stroke' }).click()
  await expect(page.getByText('Operations: 0', { exact: true })).toBeVisible()
  await stroke()
  await page.screenshot({ path: testInfo.outputPath('source-region-desktop.png') })
  await page.locator('.flexlayout__tab_button').filter({ hasText: 'Inspector' }).dblclick()
  await page.setViewportSize({ width: 390, height: 844 })
  await page.getByRole('tab', { name: 'Inspector' }).click()
  await expect(editor).toBeVisible()
  await editor.scrollIntoViewIfNeeded()
  const mobileBounds = (await editor.boundingBox())!
  expect(mobileBounds.x).toBeGreaterThanOrEqual(0)
  expect(mobileBounds.x + mobileBounds.width).toBeLessThanOrEqual(390)
  await page.mouse.move(mobileBounds.x + mobileBounds.width * 0.5, mobileBounds.y + mobileBounds.height * 0.5)
  const mobileBrush = (await page.getByTestId('source-region-brush-preview').boundingBox())!
  expect(mobileBrush.width).toBeCloseTo(mobileBrush.height, 2)
  await page.screenshot({ path: testInfo.outputPath('source-region-mobile.png') })
  expect(mock.unexpectedRequests).toEqual([])
})

test('source region is available for a project with only ordinary photos', async ({ page }) => {
  const mock = await installUiMock(page, { sourceKind: 'perspective_images' })
  await page.goto('/')
  await page.getByRole('button', { name: /Source valid region/ }).click()
  await expect(page.getByRole('slider', { name: 'Valid-circle radius' })).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Exclude brush (-)', exact: true })).toHaveAttribute('aria-pressed', 'true')
  await expect(page.getByTestId('source-region-canvas')).toBeVisible()
  expect(mock.unexpectedRequests).toEqual([])
})

test('source region step tracks saved enabled sources and updates immediately', async ({ page }) => {
  const source = (id: string, projection: ProjectSource['projection'], enabled = true): ProjectSource => ({
    id, label: id, projection, enabled, role: id === 's1' ? 'primary' : 'supplemental',
    adapter: projection === 'dual_fisheye' ? 'insta360' : 'generic_video', media_kind: 'video',
    path: `${id}.mov`, ordinal: Number(id.slice(1)),
  })
  const mock = await installUiMock(page, { frameCount: 0, regionSaved: false, regionSources: [
    source('s1', 'dual_fisheye'), source('s2', 'perspective'), source('s3', 'dual_fisheye', false),
  ] })
  await page.goto('/')
  const step = page.getByRole('button', { name: /^Source valid region / })
  await expect(step).toHaveAttribute('aria-label', 'Source valid region Not set')
  await step.click()
  await page.getByRole('slider', { name: 'Valid-circle radius' }).fill('0.45')
  await expect(step).toHaveAttribute('aria-label', 'Source valid region Not set')
  await page.getByRole('button', { name: 'Save', exact: true }).click()
  await expect(step).toHaveAttribute('aria-label', 'Source valid region Configured 1/2')
  await page.getByRole('combobox', { name: 'Source', exact: true }).selectOption('s2')
  await page.getByRole('button', { name: 'Save', exact: true }).click()
  await expect(step).toHaveAttribute('aria-label', 'Source valid region done')
  await expect(step.locator('.hier-badge')).toHaveCSS('background-color', 'rgb(76, 175, 80)')
  await page.reload()
  await expect(step).toHaveAttribute('aria-label', 'Source valid region done')
  expect(mock.unexpectedRequests).toEqual([])
})

for (const sourceKind of ['insv', 'perspective_images'] as const) {
  test(`source region exclusion stays visible on black and white images (${sourceKind})`, async ({ page }, testInfo) => {
    const mock = await installUiMock(page, { sourceKind })
    const operations = [
      { mode: 'subtract' as const, x: 0.35, y: 0.5, r: 0.13, stroke_id: 1 },
      { mode: 'subtract' as const, x: 0.4, y: 0.5, r: 0.1, stroke_id: 1 },
      { mode: 'subtract' as const, x: 0.65, y: 0.5, r: 0.13, stroke_id: 2 },
      { mode: 'add' as const, x: 0.35, y: 0.5, r: 0.04, stroke_id: 3 },
    ]
    const region: SourceRegion = { saved: true, views: sourceKind === 'insv' ? {
      lens0: { kind: 'circle', cx: 0.5, cy: 0.5, r: 0.48, operations },
      lens1: { kind: 'circle', cx: 0.5, cy: 0.5, r: 0.48, operations: [] },
    } : { main: { kind: 'full', operations } } }
    await page.route('**/source-region?*', route => route.fulfill({ json: region }))
    await page.goto('/')
    const image = await page.evaluate(() => {
      const canvas = document.createElement('canvas')
      canvas.width = canvas.height = 400
      const context = canvas.getContext('2d')!
      context.fillStyle = 'black'
      context.fillRect(0, 0, 200, 400)
      context.fillStyle = 'white'
      context.fillRect(200, 0, 200, 400)
      return canvas.toDataURL('image/png').split(',')[1]
    })
    await page.route('**/frames/*/image?*', route => route.fulfill({
      contentType: 'image/png', body: Buffer.from(image, 'base64'),
    }))
    await page.getByRole('button', { name: /Source valid region/ }).click()
    const editor = page.getByTestId('source-region-canvas')
    await expect(editor).toBeVisible()
    const pixels = await editor.evaluate(async element => {
      const svg = element.cloneNode(true) as SVGSVGElement
      svg.removeAttribute('style')
      svg.setAttribute('width', '400')
      svg.setAttribute('height', '400')
      const url = URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(svg)], { type: 'image/svg+xml' }))
      try {
        const image = new Image()
        image.src = url
        await image.decode()
        const canvas = document.createElement('canvas')
        canvas.width = canvas.height = 400
        const context = canvas.getContext('2d')!
        context.fillStyle = 'black'
        context.fillRect(0, 0, 200, 400)
        context.fillStyle = 'white'
        context.fillRect(200, 0, 200, 400)
        context.drawImage(image, 0, 0)
        const pixel = (x: number, y: number) => [...context.getImageData(x, y, 1, 1).data].slice(0, 3)
        const highlighted = (left: number, right: number) => {
          let count = 0
          for (let x = left; x < right; x++) for (let y = 180; y < 220; y++) {
            const [r, g, b] = pixel(x, y)
            if (r > 90 && r > g * 1.2 && b > g * 1.1) count++
          }
          return count
        }
        return { black: highlighted(90, 130), white: highlighted(220, 300),
          validBlack: pixel(40, 200), validWhite: pixel(360, 200), restored: pixel(140, 200),
          overlap: pixel(191, 200), boundary: pixel(311, 200) }
      } finally { URL.revokeObjectURL(url) }
    })
    expect(pixels.black).toBeGreaterThan(40)
    expect(pixels.white).toBeGreaterThan(40)
    expect(pixels.validBlack).toEqual([0, 0, 0])
    expect(pixels.validWhite).toEqual([255, 255, 255])
    expect(pixels.restored).toEqual([0, 0, 0])
    expect(pixels.overlap[0]).toBeLessThan(220)
    expect(pixels.boundary[0]).toBeGreaterThan(220)
    await editor.scrollIntoViewIfNeeded()
    await page.screenshot({ path: testInfo.outputPath('region-contrast-desktop.png') })
    await page.locator('.flexlayout__tab_button').filter({ hasText: 'Inspector' }).dblclick()
    await page.setViewportSize({ width: 390, height: 844 })
    await page.getByRole('tab', { name: 'Inspector' }).click()
    await expect(editor).toBeVisible()
    await editor.scrollIntoViewIfNeeded()
    await page.screenshot({ path: testInfo.outputPath('region-contrast-mobile.png') })
    expect(mock.unexpectedRequests).toEqual([])
  })
}

test('photo and dataset camera inspectors share capture summary fields', async ({ page }) => {
  test.setTimeout(60_000)
  const mock = await installUiMock(page)
  await page.goto('/')

  await expandPhotos(page)
  const photo = page.getByText('Frame 0', { exact: true })
  await expect(photo).toBeVisible()
  await photo.click()
  let summary = page.getByTestId('capture-summary')
  await expect(summary).toBeVisible()
  await expect(summary.getByText('Frame', { exact: true })).toBeVisible()
  await expect(summary.getByText('Lens', { exact: true })).toBeVisible()
  await expect(summary.getByText('Reconstruction registration', { exact: true })).toBeVisible()
  await expect(summary.getByText('Frame 3D points', { exact: true })).toBeVisible()

  const cameras = page.getByText(/Dataset · Cameras/)
  await expect(cameras).toBeVisible()
  await cameras.click()
  const camera = page.getByText('sources/s1/lens0/frame_000000.jpg', { exact: true })
  await expect(camera).toBeVisible()
  await camera.click()
  summary = page.getByTestId('capture-summary')
  await expect(summary).toBeVisible()
  await expect(summary.getByText('Frame', { exact: true })).toBeVisible()
  await expect(summary.getByText('Lens', { exact: true })).toBeVisible()
  await expect(summary.getByText('Reconstruction registration', { exact: true })).toBeVisible()
  await expect(summary.getByText('Image 3D points', { exact: true })).toBeVisible()
  await expect(page.getByText('Camera position', { exact: true })).toBeVisible()
  await expect(page.locator('.inspector-preview.circular')).toBeVisible()
  expect(mock.unexpectedRequests).toEqual([])
})

test('clicking empty Scene View space clears the selected dataset camera', async ({ page }) => {
  const mock = await installUiMock(page)
  await page.goto('/')
  await page.getByRole('button', { name: /Dataset · Cameras/ }).click()
  await page.getByRole('button', { name: 'sources/s1/lens0/frame_000000.jpg' }).click()
  await expect(page.getByText('Camera position', { exact: true })).toBeVisible()

  await page.getByRole('tab', { name: 'Scene View' }).click()
  const canvas = page.locator('canvas')
  const bounds = await canvas.boundingBox()
  expect(bounds).not.toBeNull()
  await canvas.click({ position: { x: 20, y: bounds!.height - 20 } })

  await page.getByRole('tab', { name: 'Inspector' }).click()
  await expect(page.getByText('Camera position', { exact: true })).toHaveCount(0)
  expect(mock.unexpectedRequests).toEqual([])
})

test('feature and training masks are independent steps with distinct defaults', async ({ page }) => {
  const mock = await installUiMock(page)
  await page.goto('/')

  await page.getByRole('button', { name: /SAM3 feature masks/ }).click()
  const featureEnabled = page.getByRole('checkbox', { name: 'Enable feature-matching masks' })
  await expect(featureEnabled).toBeChecked()
  await expect(page.getByText('Feature-mask prompt', { exact: true }).locator('..').locator('input'))
    .toHaveValue(FEATURE_MASK_PROMPT)
  await expect(page.getByText('Automatic resolution: 2048px', { exact: true })).toBeVisible()
  await featureEnabled.uncheck()
  await expect(page.getByRole('button', { name: /SAM3 feature masks skip/ })).toBeVisible()

  await page.getByRole('button', { name: /SAM3 training masks/ }).click()
  await expect(page.getByRole('checkbox', { name: 'Enable training masks' })).toBeChecked()
  await expect(page.getByText('Training-mask prompt', { exact: true }).locator('..').locator('input'))
    .toHaveValue(TRAINING_MASK_PROMPT)
  await expect(page.getByText('Automatic resolution: 2048px', { exact: true })).toBeVisible()
  expect(mock.unexpectedRequests).toEqual([])
})

test('sparse cleanup exposes full-track defaults and persists percent edits as ratios', async ({ page }) => {
  const mock = await installUiMock(page)
  await page.goto('/')
  await page.getByRole('button', { name: /Sparse noise cleanup/ }).click()

  await expect(page.getByRole('checkbox', { name: 'Enable full-track sparse cleanup' })).toBeChecked()
  const relativeError = page.getByRole('spinbutton', { name: 'Relative position error limit % (default 2)' })
  await expect(relativeError).toHaveValue('2')
  await expect(page.getByRole('spinbutton', { name: 'Pixel noise σ px (default 1)' })).toHaveValue('1')
  await expect(page.getByRole('spinbutton', { name: 'Reprojection error P95 px (default 2)' }))
    .toHaveValue('2')
  await expect(page.getByText('Both lenses from the same source and capture form one group.', { exact: false })).toBeVisible()
  await expect(page.getByText('Uncertainty assumes correct camera poses and calibration;', { exact: false })).toBeVisible()

  await relativeError.fill('1.5')
  await relativeError.press('Tab')
  await expect.poll(() => (mock.uiStateUpdates.at(-1)?.ui.params as Record<string, unknown>)?.cleanupRelativeError)
    .toBe(0.015)
  await relativeError.fill('0')
  await relativeError.press('Tab')
  await expect(relativeError).toHaveValue('1.5')

  await page.getByRole('button', { name: 'Regenerate', exact: true }).click()
  await expect.poll(() => mock.reruns.length).toBe(1)
  expect(mock.reruns[0].body.params_by_stage.cleanup_sparse).toEqual({
    enabled: true, relative_error: 0.015, pixel_sigma: 1, max_cross_error: 2,
  })
  await page.reload()
  await page.getByRole('button', { name: /Sparse noise cleanup/ }).click()
  await expect(relativeError).toHaveValue('1.5')
  expect(mock.unexpectedRequests).toEqual([])
})

test('new reconstruction settings use strict while an explicit COLMAP preset survives reload', async ({ page }) => {
  const mock = await installUiMock(page)
  await page.goto('/')
  await page.getByRole('button', { name: /Sparse reconstruction/ }).click()
  await page.getByText('Advanced COLMAP', { exact: false }).click()

  const preset = page.getByRole('combobox', { name: 'Triangulation preset' })
  await expect(preset).toHaveValue('strict')
  await expect(page.getByRole('spinbutton', { name: 'filter_max_reproj_error (app default 1.0 px)' }))
    .toHaveValue('1')
  await preset.selectOption('default')
  await expect.poll(() => (mock.uiStateUpdates.at(-1)?.ui.params as Record<string, unknown>)?.filterMaxReprojError)
    .toBe(4)
  await page.reload()
  await page.getByRole('button', { name: /Sparse reconstruction/ }).click()
  await page.getByText('Advanced COLMAP', { exact: false }).click()
  await expect(preset).toHaveValue('default')
  await expect(page.getByRole('spinbutton', { name: 'filter_max_reproj_error (app default 1.0 px)' }))
    .toHaveValue('4')
  await page.getByRole('button', { name: 'Regenerate', exact: true }).click()
  await expect.poll(() => mock.reruns.length).toBe(1)
  expect(mock.reruns[0].body.params_by_stage.reconstruct).toMatchObject({
    filter_max_reproj_error: 4, filter_min_tri_angle: 1.5, tri_min_angle: 1.5,
  })
  expect(mock.unexpectedRequests).toEqual([])
})

test('existing custom reconstruction settings retain their values when defaults change', async ({ page }) => {
  const mock = await installUiMock(page, {
    savedUiStates: {
      p1: { params: {
        ...COLMAP_TRIANGULATION_PRESETS.standard,
        colmapTriangulationPreset: 'custom', filterMaxReprojError: 1.75,
      } },
    },
  })
  await page.goto('/')
  await page.getByRole('button', { name: /Sparse reconstruction/ }).click()
  await page.getByText('Advanced COLMAP', { exact: false }).click()
  await expect(page.getByRole('combobox', { name: 'Triangulation preset' })).toHaveValue('custom')
  await expect(page.getByRole('spinbutton', { name: 'filter_max_reproj_error (app default 1.0 px)' }))
    .toHaveValue('1.75')
  await expect(page.getByRole('spinbutton', { name: 'tri_min_angle (app default 5.0°)' })).toHaveValue('3')
  expect(mock.unexpectedRequests).toEqual([])
})

test('dense initialization is an independent opt-in step with bounded defaults', async ({ page }) => {
  const mock = await installUiMock(page)
  await page.goto('/')

  await page.getByText('Dense initialization', { exact: true }).click()
  const enabled = page.getByRole('checkbox', { name: 'Enable RoMaV2 dense initialization' })
  await expect(enabled).not.toBeChecked()
  await expect(page.getByRole('button', { name: /Dense initialization skip/ })).toBeVisible()
  await enabled.check()
  await expect(page.getByRole('combobox')).toHaveValue('turbo')
  await expect(page.getByText('25%', { exact: true })).toBeVisible()
  await expect(page.getByRole('spinbutton', { name: 'Hard cap for added points' })).toHaveValue('200000')
  await expect(page.getByText('Maximum points written into the sparse model after all filters and voxel deduplication; not a Gaussian cap.', { exact: true })).toBeVisible()
  await expect(page.getByText('These geometry filters reject mismatches and unstable triangulation.', { exact: false })).toBeVisible()
  await expect(page.getByText('Maximum pixel error after projecting a triangulated point back into both cameras.', { exact: false })).toBeVisible()
  const confidence = page.getByRole('spinbutton', { name: 'Certainty threshold' })
  await confidence.fill('.5')
  await confidence.press('Tab')
  await expect(confidence).toHaveValue('0.5')
  expect(mock.unexpectedRequests).toEqual([])
})

test('frame extraction exposes and submits rolling-shutter motion limit', async ({ page }) => {
  const mock = await installUiMock(page)
  await page.goto('/')

  await page.getByText('Frame extraction', { exact: true }).click()
  const limit = page.getByRole('slider', { name: 'Rolling-shutter rotation limit' })
  await expect(limit).toHaveValue('0.8')
  await limit.fill('0.6')
  await page.getByRole('button', { name: 'Regenerate', exact: true }).click()
  await expect.poll(() => mock.reruns.length).toBe(1)
  expect(mock.reruns[0].body.params_by_stage.extract_frames.max_rolling_shutter_motion_deg).toBe(0.6)
  expect(mock.unexpectedRequests).toEqual([])
})

test('perspective image collections hide video frame controls and use pinhole reconstruction', async ({ page }) => {
  const mock = await installUiMock(page, { sourceKind: 'perspective_images' })
  await page.goto('/')

  await expect(page.getByRole('button', { name: /Collect images done/ })).toBeVisible()
  await page.getByRole('button', { name: /Collect images done/ }).click()
  await expect(page.getByTestId('image-collection-hint')).toContainText('collected unchanged')
  await expect(page.getByText('Method', { exact: true })).toHaveCount(0)
  await expect(page.getByRole('slider', { name: 'Rolling-shutter rotation limit' })).toHaveCount(0)

  await page.getByRole('button', { name: /Source done/ }).click()
  const mode = page.getByText('Reconstruction mode', { exact: true }).locator('..').locator('select')
  await expect(mode).toHaveValue('pinhole_rig')
  await expect(mode.locator('option')).toHaveCount(1)
  expect(mock.unexpectedRequests).toEqual([])
})

test('frame sharpness scores show a recommendation without changing the applied threshold', async ({ page }) => {
  const mock = await installUiMock(page, { frameCount: 5 })
  await page.goto('/')

  await page.getByText('Frame extraction', { exact: true }).click()
  await page.getByText('Method', { exact: true }).locator('..').locator('select').selectOption('spatial')
  const threshold = page.getByRole('slider', { name: /Sharpness threshold/ })
  await expect(threshold).toHaveValue('0')
  await expect(threshold).toHaveAttribute('max', '2000')
  await expect(page.getByTestId('sharpness-recommendation')).toContainText('Recommended threshold 10')
  await expect(page.getByText(/Pending settings: min_sharpness/)).toHaveCount(0)
  await expandPhotos(page)
  await expect(page.locator('[title^="Sharpness:"]').first()).toHaveAttribute('title', 'Sharpness: 12.0')
  expect(mock.unexpectedRequests).toEqual([])
})

test('nonzero spatial sharpness threshold becomes fresh after regeneration', async ({ page }) => {
  const mock = await installUiMock(page, { frameCount: 5 })
  await page.goto('/')
  await page.getByText('Frame extraction', { exact: true }).click()
  const threshold = page.getByRole('slider', { name: /Sharpness threshold/ })
  await threshold.fill('100')
  const pendingStage = page.getByRole('button', { name: /Frame extraction regenerate/ })
  await expect(pendingStage).toBeVisible()
  await expect(pendingStage).toContainText('Pending settings: min_sharpness: 0 → 100')

  await page.getByRole('button', { name: 'Regenerate', exact: true }).click()

  await expect.poll(() => mock.reruns.at(-1)?.body.params_by_stage.extract_frames.min_sharpness)
    .toBe(100)
  await expect(page.getByRole('button', { name: /Frame extraction done/ })).toBeVisible()
  await expect(page.getByText(/Pending settings: min_sharpness/)).toHaveCount(0)
  expect(mock.unexpectedRequests).toEqual([])
})

test('photo and dataset camera previews switch between both mask steps', async ({ page }) => {
  const mock = await installUiMock(page)
  await page.goto('/')

  await expandPhotos(page)
  await page.getByText('Frame 0', { exact: true }).click()
  let purpose = page.getByRole('group', { name: 'Mask purpose' })
  await expect(purpose.getByRole('button', { name: 'Training' })).toHaveAttribute('aria-pressed', 'true')
  await page.getByRole('tab', { name: 'Mask' }).click()
  await expect(page.locator('img[alt="Mask"]')).toHaveAttribute('src', /purpose=training/)
  await purpose.getByRole('button', { name: 'Feature' }).click()
  await expect(page.locator('img[alt="Mask"]')).toHaveAttribute('src', /purpose=feature/)

  await page.getByText(/Dataset · Cameras/).click()
  await page.getByText('sources/s1/lens0/frame_000000.jpg', { exact: true }).click()
  purpose = page.getByRole('group', { name: 'Mask purpose' })
  await page.getByRole('tab', { name: 'Mask' }).click()
  await purpose.getByRole('button', { name: 'Feature' }).click()
  await expect(page.locator('img[alt="Mask"]')).toHaveAttribute('src', /purpose=feature/)
  expect(mock.unexpectedRequests).toEqual([])
})

test('a completed SAM3 image becomes previewable before the mask step finishes', async ({ page }) => {
  const mock = await installUiMock(page, {
    runningStage: 'generate_feature_masks',
    incrementalFeatureMask: true,
  })
  await page.goto('/')

  await expandPhotos(page)
  await page.getByText('Frame 0', { exact: true }).click()
  await expect(page.getByText('Feature · Dynamic coverage', { exact: true })).toBeVisible({ timeout: 5_000 })
  await page.getByRole('tab', { name: 'Mask' }).click()
  await expect(page.locator('img[alt="Mask"]')).toHaveAttribute('src', /revision=live-run/)
  expect(mock.maskRequests.feature).toBeGreaterThanOrEqual(2)
  expect(mock.unexpectedRequests).toEqual([])
})

test('feature, matching, and mapper controls have localized names and explanations', async ({ page }) => {
  const mock = await installUiMock(page)
  await page.goto('/')

  await page.getByRole('button', { name: /^Frame extraction/ }).click()
  await expect(page.getByText('Method', { exact: true }).locator('..').locator('select')).toHaveValue('spatial')
  await page.getByText('Extract features', { exact: true }).click()
  await expect(page.getByText('Feature image-size limit (max_image_size)', { exact: true })).toBeVisible()
  await expect(page.getByTestId('primary-image-size')).toHaveText('Current primary image size: 100 × 100 px')
  await expect(page.getByText('Features per image (max_num_features)', { exact: true })).toBeVisible()
  await expect(page.getByText('SIFT peak threshold (peak_threshold)', { exact: true })).toBeVisible()
  await expect(page.getByText('SIFT edge threshold (edge_threshold)', { exact: true })).toBeVisible()
  await expect(page.getByText('Affine shape + DSP (affine_shape + DSP)', { exact: true })).toBeVisible()
  const featureStatistics = page.getByRole('button', { name: 'Stage statistics', exact: true })
  await expect(featureStatistics).toHaveAttribute('aria-expanded', 'false')
  await featureStatistics.click()
  await expect(page.getByText('Minimum keypoints', { exact: true })).toBeVisible()
  await expect(page.getByText('Maximum keypoints', { exact: true })).toBeVisible()

  await page.getByText('Match features', { exact: true }).click()
  await expect(page.getByText('Feature matcher (matcher_type)', { exact: true })).toBeVisible()
  await expect(page.getByText('Image-pair strategy (pairing)', { exact: true })).toBeVisible()
  await page.getByText('Image-pair strategy (pairing)', { exact: true }).locator('..').locator('select').selectOption('sequential')
  await expect(page.getByText('Loop closure (loop_closure)', { exact: true })).toBeVisible()
  await expect(page.getByText('Transitive matching (transitive_matching)', { exact: true })).toBeVisible()
  await expect(page.getByText('Matches per image pair (max_num_matches)', { exact: true })).toBeVisible()
  await expect(page.getByText('Two-view minimum inliers (two-view min_num_inliers)', { exact: true })).toBeVisible()
  await expect(page.getByText('Geometry-guided matching (guided_matching)', { exact: true })).toBeVisible()
  await expect(page.getByText('Generalized rig verification', { exact: true })).toBeVisible()

  await page.getByText('Sparse reconstruction', { exact: true }).click()
  await expect(page.getByText('Reconstruction solver (mapper)', { exact: true })).toBeVisible()
  const mapper = page.getByText('Reconstruction solver (mapper)', { exact: true }).locator('..').locator('select')
  const fisheyeWarning = page.getByTestId('glomap-fisheye-warning')
  await expect(mapper).toHaveValue('incremental')
  await expect(fisheyeWarning).toHaveCount(0)
  await mapper.selectOption('global')
  await expect(fisheyeWarning).toContainText('GLOMAP does not support fisheye rays beyond 180°')
  await expect(page.getByText('View-graph calibration (view_graph_calibration)', { exact: true })).toBeVisible()
  await expect(page.getByText('GPU bundle adjustment (ba_use_gpu)', { exact: true })).toBeVisible()
  const reconstructionStatistics = page.getByRole('button', { name: 'Stage statistics', exact: true })
  await reconstructionStatistics.click()
  await expect(page.getByText('Maximum / P95 ratio', { exact: true })).toBeVisible()
  await expect(page.getByText('56.6×', { exact: true })).toBeVisible()
  await expect(page.getByText('Largest jumps #1 · From capture', { exact: true })).toBeVisible()
  await expect(page.getByText('Largest jumps #1 · To capture', { exact: true })).toBeVisible()
  await page.getByText('Restore metric scale', { exact: true }).click()
  await expect(page.getByText('Metric-scale restoration', { exact: true })).toBeVisible()
  await page.getByText('Scene coordinate alignment', { exact: true }).click()
  await expect(page.getByText('Does not require metric scale. Aligns horizontal axes from dominant orthogonal walls and moves the nearest dominant plane below the camera path to Y=0. Yaw is unchanged without reliable walls.', { exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Stage statistics', exact: true }).click()
  await expect(page.getByText('Nearest dominant horizontal plane', { exact: true })).toBeVisible()
  await expect(page.getByText('1.360 units', { exact: true })).toBeVisible()
  await expect(page.getByText('-21.560°', { exact: true })).toBeVisible()
  await page.getByText('Export', { exact: true }).click()
  await expect(page.getByText('Optimize fisheye training images', { exact: true })).toBeVisible()
  await page.getByText('Sparse reconstruction', { exact: true }).click()

  await page.getByTitle('Settings', { exact: true }).click()
  await page.locator('.pop select').nth(1).selectOption('zh')
  await expect(page.getByText('重建器 (mapper)', { exact: true })).toBeVisible()
  await expect(page.getByTestId('glomap-fisheye-warning')).toContainText('GLOMAP 不支持超过 180° 的鱼眼射线')
  await expect(page.getByText('视图图校准 (view_graph_calibration)', { exact: true })).toBeVisible()
  await expect(page.getByText('GPU 光束平差 (ba_use_gpu)', { exact: true })).toBeVisible()
  await page.mouse.click(10, 200)
  await page.getByText('恢复真实大小', { exact: true }).click()
  await expect(page.getByText('真实大小恢复方式', { exact: true })).toBeVisible()
  await page.getByText('场景坐标对齐', { exact: true }).first().click()
  await expect(page.getByText('不要求真实比例；根据主要正交墙面对齐水平主轴，并把相机路径下方最近的主要平面移动到 Y=0。没有可靠墙面时不会强行旋转。', { exact: true })).toBeVisible()
  await page.getByText('特征匹配', { exact: true }).click()
  await expect(page.getByText('特征匹配器 (matcher_type)', { exact: true })).toBeVisible()
  await expect(page.getByText('图像配对策略 (pairing)', { exact: true })).toBeVisible()
  await expect(page.getByText('双视图最少内点数 (two-view min_num_inliers)', { exact: true })).toBeVisible()
  await expect(page.getByText('Generalized rig 验证', { exact: true })).toBeVisible()
  await page.getByText('特征抽取', { exact: true }).click()
  await expect(page.getByText('特征抽取最大图像尺寸 (max_image_size)', { exact: true })).toBeVisible()
  await expect(page.getByText('每张图像最大特征数 (max_num_features)', { exact: true })).toBeVisible()
  await expect(page.getByText('SIFT 峰值阈值 (peak_threshold)', { exact: true })).toBeVisible()
  await expect(page.getByText('SIFT 边缘阈值 (edge_threshold)', { exact: true })).toBeVisible()
  await expect(page.getByText('仿射形状 + DSP (affine_shape + DSP)', { exact: true })).toBeVisible()
  await page.getByText('导出', { exact: true }).click()
  await expect(page.getByText('优化鱼眼训练图像', { exact: true })).toBeVisible()
  expect(mock.unexpectedRequests).toEqual([])
})

test('resolution limits follow fisheye source geometry automatically', async ({ page }) => {
  const fisheye = await installUiMock(page, { sourceWidth: 5376, sourceHeight: 5376 })
  await page.goto('/')
  await page.getByText('Extract features', { exact: true }).click()
  const featureLimit = page.getByRole('spinbutton', { name: 'Feature image-size limit (max_image_size)' })
  await expect(featureLimit).toBeDisabled()
  await expect(featureLimit).toHaveValue('5376')
  await expect(page.getByRole('spinbutton', { name: 'Features per image (max_num_features)' }))
    .toBeDisabled()
  await expect(page.getByRole('spinbutton', { name: 'Features per image (max_num_features)' }))
    .toHaveValue('16384')
  await page.getByText('SAM3 feature masks', { exact: true }).click()
  await expect(page.getByRole('checkbox', { name: 'Auto from sources' })).toBeChecked()
  await expect(page.getByText('Automatic resolution: 3072px', { exact: true })).toBeVisible()
  expect(fisheye.unexpectedRequests).toEqual([])
})

test('resolution limits preserve ERP horizontal sampling automatically', async ({ page }) => {
  const erp = await installUiMock(page, {
    sourceKind: 'erp_video', sourceWidth: 7680, sourceHeight: 3840,
  })
  await page.goto('/')
  await page.getByText('Extract features', { exact: true }).click()
  await expect(page.getByRole('spinbutton', { name: 'Feature image-size limit (max_image_size)' }))
    .toHaveValue('7680')
  await expect(page.getByRole('spinbutton', { name: 'Features per image (max_num_features)' }))
    .toHaveValue('32768')
  await page.getByText('SAM3 feature masks', { exact: true }).click()
  await expect(page.getByText('Automatic resolution: 4096px', { exact: true })).toBeVisible()
  expect(erp.unexpectedRequests).toEqual([])
})

test('a capable pinned runtime enables GPU bundle adjustment by default', async ({ page }) => {
  const mock = await installUiMock(page, { gpuBundleAdjustment: true })
  await page.goto('/')

  await page.getByText('Sparse reconstruction', { exact: true }).click()
  const row = page.getByText('GPU bundle adjustment (ba_use_gpu)', { exact: true }).locator('..')
  await expect(row.locator('input[type="checkbox"]')).toBeChecked()
  await expect(row.locator('input[type="checkbox"]')).toBeEnabled()
  expect(mock.unexpectedRequests).toEqual([])
})

test('phone image folders can be added as supplemental perspective sources', async ({ page }) => {
  const mock = await installUiMock(page)
  await page.goto('/')

  await page.getByRole('button', { name: /Source done/ }).click()
  const sourceType = page.getByText('Add data source', { exact: true }).locator('..').locator('select')
  await sourceType.selectOption('perspective_images')
  await page.getByRole('button', { name: 'Choose path…' }).click()
  const phoneRow = page.locator('.fs-row').filter({ hasText: 'Phone photos' })
  await phoneRow.getByRole('button', { name: 'Select this folder' }).click()

  await expect.poll(() => mock.sourceAdds.length).toBe(1)
  expect(mock.sourceAdds[0]).toMatchObject({
    role: 'supplemental',
    adapter: 'generic_images',
    media_kind: 'images',
    projection: 'perspective',
    path: MOCK_PHONE,
  })
  await expect(page.getByText('Supplemental source', { exact: true })).toBeVisible()
  await expect(page.getByText('Detail', { exact: true })).toBeVisible()
  expect(mock.unexpectedRequests).toEqual([])
})

test('switching projects resets inspector selection state', async ({ page }) => {
  const mock = await installUiMock(page, { secondProject: true })
  await page.goto('/')

  await expandPhotos(page)
  await page.getByRole('button', { name: /Frame 0/ }).click()
  await expect(page.getByTestId('capture-summary')).toBeVisible()

  await page.getByRole('button', { name: /Projects/ }).click()
  await page.getByText('Second project', { exact: true }).click()
  await expect(page.locator('h1')).toHaveText('Second project')
  await expect(page.getByTestId('capture-summary')).toHaveCount(0)
  expect(mock.unexpectedRequests).toEqual([])
})

test('deleting the current project selects the next available project', async ({ page }) => {
  const mock = await installUiMock(page, { secondProject: true })
  await page.goto('/')
  await page.getByRole('button', { name: /Projects/ }).click()
  const row = page.locator('.fs-row').filter({ hasText: 'Mock project' })
  await row.getByRole('button', { name: 'Delete', exact: true }).click()
  await row.getByRole('button', { name: 'Delete', exact: true }).click()
  await expect(page.locator('h1')).toHaveText('Second project')
  expect(mock.unexpectedRequests).toEqual([])
})

test('browser refresh restores the last opened project', async ({ page }) => {
  const mock = await installUiMock(page, { secondProject: true })
  await page.goto('/')
  await page.getByRole('button', { name: /Projects/ }).click()
  const row = page.locator('.fs-row').filter({ hasText: 'Second project' })
  await row.getByRole('button', { name: 'Open', exact: true }).click()
  await expect(page.locator('h1')).toHaveText('Second project')

  await expect.poll(() => mock.getPreferences().lastProjectId).toBe('p2')
  await page.reload()

  await expect(page.locator('h1')).toHaveText('Second project')
  expect(mock.unexpectedRequests).toEqual([])
})

test('creating a project opens it immediately', async ({ page }) => {
  const mock = await installUiMock(page)
  await page.goto('/')
  await page.getByRole('button', { name: /Projects/ }).click()
  await page.getByPlaceholder('Project name').fill('New active project')
  await page.getByRole('button', { name: 'Create', exact: true }).click()

  await expect(page.locator('h1')).toHaveText('New active project')
  expect(mock.unexpectedRequests).toEqual([])
})

test('dataset camera preview resolves ERP and pinhole image layouts', async ({ page }) => {
  const erpMock = await installUiMock(page, {
    sourceKind: 'erp_video',
    imageName: 'sources/s1/frame_000000.jpg',
    cameraModel: 'EQUIRECTANGULAR',
  })
  await page.goto('/')
  await page.getByRole('button', { name: /Dataset · Cameras/ }).click()
  await page.getByRole('button', { name: 'sources/s1/frame_000000.jpg' }).click()
  await expect(page.locator('.inspector-preview-image')).toHaveAttribute('src', /\/prepared-image\?name=/)
  await expect(page.locator('.inspector-preview.circular')).toHaveCount(0)
  expect(erpMock.unexpectedRequests).toEqual([])
})

test('pinhole dataset camera uses the reprojected view endpoint', async ({ page }) => {
  const mock = await installUiMock(page, {
    imageName: 'sources/s1/front_lens0/frame_000000.jpg',
    cameraModel: 'PINHOLE',
    reconMode: 'pinhole_rig',
  })
  await page.goto('/')
  await page.getByRole('button', { name: /Dataset · Cameras/ }).click()
  await page.getByRole('button', { name: 'sources/s1/front_lens0/frame_000000.jpg' }).click()
  await expect(page.locator('.inspector-preview-image')).toHaveAttribute('src', /\/prepared-image\?name=/)
  await expect(page.locator('.inspector-preview.circular')).toHaveCount(0)
  expect(mock.unexpectedRequests).toEqual([])
})

test('stage rows and source dialog support keyboard navigation', async ({ page }) => {
  const mock = await installUiMock(page)
  await page.goto('/')

  const sourceStep = page.getByRole('button', { name: /Source done/ })
  await sourceStep.focus()
  await page.keyboard.press('Enter')
  await page.getByRole('button', { name: 'Choose path…' }).click()
  const dialog = page.getByRole('dialog', { name: 'Select source…' })
  await expect(dialog).toBeVisible()
  await page.keyboard.press('Escape')
  await expect(dialog).toHaveCount(0)
  expect(mock.unexpectedRequests).toEqual([])
})

test('complete INSV flow can be driven from UI', async ({ page, request }) => {
  const source = process.env.SPHERE_E2E_SOURCE
  test.skip(!source, 'SPHERE_E2E_SOURCE is required for the real UI pipeline test')
  const projectName = `e2e-${Date.now()}`
  await page.goto('/')
  await page.getByRole('button', { name: /Projects/ }).click()
  await page.getByPlaceholder('Project name').fill(projectName)
  await page.getByRole('button', { name: 'Create', exact: true }).click()
  await expect(page.locator('h1')).toHaveText(projectName)

  const projects = await (await request.get('/api/projects')).json() as Array<{ id: string; name: string }>
  const project = projects.find(item => item.name === projectName)
  expect(project).toBeTruthy()

  const filesystem = /^[A-Za-z]:[\\/]/.test(source) ? path.win32 : path.posix
  const drives = (await (await request.get('/api/fs/drives')).json() as { drives: string[] }).drives
  const sourceKey = filesystem.resolve(source).toLowerCase()
  const root = drives
    .filter(candidate => {
      const key = filesystem.resolve(candidate).toLowerCase()
      return sourceKey === key || sourceKey.startsWith(key.endsWith(filesystem.sep) ? key : key + filesystem.sep)
    })
    .sort((left, right) => right.length - left.length)[0]
  expect(root, `No file-browser root contains ${source}`).toBeTruthy()

  await page.getByRole('button', { name: /Source/ }).click()
  await page.getByRole('button', { name: 'Choose path…' }).click()
  const dialog = page.getByRole('dialog', { name: 'Select source…' })
  await dialog.getByRole('button', { name: root.replaceAll('\\', '/') }).click()
  const relativeDirectory = filesystem.relative(root, filesystem.dirname(source))
  for (const directory of relativeDirectory.split(/[\\/]/).filter(Boolean)) {
    await dialog.locator('.fs-main').filter({ hasText: directory }).click()
  }
  const sourceSet = page.waitForResponse(response => (
    response.request().method() === 'POST' && response.url().endsWith(`/api/projects/${project!.id}/sources`)
  ))
  await dialog.locator('.fs-row-button').filter({ hasText: filesystem.basename(source) }).click()
  await sourceSet
  await expect(dialog).toHaveCount(0)

  const generateAll = page.getByRole('button', { name: 'Generate all pending steps' })
  await expect(generateAll).toBeEnabled()
  await generateAll.click()
  await expect.poll(async () => {
    const statuses = await (await request.get(`/api/projects/${project!.id}/stages`)).json() as {
      stages: Array<{ stage: string; has_output: boolean; status: string | null; error_text?: string | null }>
    }
    const failed = statuses.stages.find(item => item.status === 'failed')
    expect(failed?.error_text ?? null).toBeNull()
    return statuses.stages.find(item => item.stage === 'export_dataset')?.has_output ?? false
  }, { timeout: 30 * 60_000 }).toBe(true)

  const finalStatuses = await (await request.get(`/api/projects/${project!.id}/stages`)).json() as {
    stages: Array<{ stage: string; has_output: boolean }>
  }
  expect(finalStatuses.stages.find(item => item.stage === 'export_dataset')?.has_output).toBe(true)
  const exportInfo = await (await request.get(`/api/projects/${project!.id}/export-info`)).json() as { dir: string }
  expect(exportInfo.dir).toBeTruthy()
  if (process.env.SPHERE_E2E_KEEP !== '1') {
    await request.delete(`/api/projects/${project!.id}`)
  } else {
    console.log(`E2E_PROJECT_ID=${project!.id}`)
  }
})
