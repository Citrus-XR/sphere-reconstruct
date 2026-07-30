import { expect, test, type Page } from '@playwright/test'
import path from 'node:path'
import { DEFAULT_PARAMS, paramsForStage, type ReconMode } from '../src/features/stageParams'

const MOCK_SOURCE = 'D:\\VID 2026\\clip.insv'
const MOCK_EXPORT = 'D:\\LFStudio\\export_dataset'
const MOCK_ENV_PATH = 'D:\\very-long-workspace-directory\\nested-runtime\\models\\and-tools\\current-environment'
const MOCK_PHONE = 'D:\\mixed-inputs\\Phone photos'
const TRAINING_MASK_PROMPT = "person,camera operator,person's shadow"
const FEATURE_MASK_PROMPT = `${TRAINING_MASK_PROMPT},animal,sky,tree,vehicle,airplane,water`

interface MockOptions {
  sourceKind?: 'insv' | 'erp_video'
  imageName?: string
  cameraModel?: string
  reconMode?: ReconMode
  secondProject?: boolean
  multipleFrameSources?: boolean
  runningStage?: string
  runningProgress?: number | null
  runningMessageKey?: string
  socketEvents?: Array<Record<string, unknown>>
  incrementalFeatureMask?: boolean
  frameCount?: number
  gpuBundleAdjustment?: boolean
}

const installUiMock = async (page: Page, options: MockOptions = {}) => {
  const sourceKind = options.sourceKind ?? 'insv'
  const reconMode = options.reconMode ?? (sourceKind === 'insv' ? 'native_fisheye' : 'equirectangular')
  const imageName = options.imageName ?? (sourceKind === 'insv'
    ? 'sources/s1/front/frame_000000.jpg' : 'sources/s1/frame_000000.jpg')
  const unexpectedRequests: string[] = []
  const reruns: Array<{ stage: string; body: Record<string, Record<string, unknown>> }> = []
  const sourceAdds: Array<Record<string, unknown>> = []
  const deletedProjects = new Set<string>()
  const maskRequests = { feature: 0, training: 0 }
  const sourcesByProject: Record<string, Array<Record<string, unknown>>> = {
    p1: [{
      id: 's1', label: sourceKind === 'insv' ? 'Primary 360' : 'Primary ERP', role: 'primary',
      adapter: sourceKind === 'insv' ? 'insta360_insv' : 'generic_video', media_kind: 'video',
      projection: sourceKind === 'insv' ? 'dual_fisheye' : 'equirectangular', path: MOCK_SOURCE,
      ordinal: 0, enabled: true,
    }],
    p2: [],
  }
  await page.addInitScript(() => {
    localStorage.clear()
    localStorage.setItem('lang', 'en')
  })
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
    if (path === '/api/projects') {
      const projects = [{
        id: 'p1', name: 'Mock project', created_at: '2026-01-01T00:00:00Z', updated_at: '2026-01-01T00:00:00Z',
        sources: sourcesByProject.p1, state: 'exported',
        ui_state: options.reconMode ? {
          reconMode: options.reconMode,
          params: {},
          disabled: [],
        } : null,
      }]
      if (options.secondProject) projects.push({
        id: 'p2', name: 'Second project', created_at: '2025-01-01T00:00:00Z', updated_at: '2025-01-01T00:00:00Z',
        sources: sourcesByProject.p2, state: 'created', ui_state: null,
      })
      await route.fulfill({ json: projects.filter(project => !deletedProjects.has(project.id)) })
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
    const projectMatch = path.match(/^\/api\/projects\/(p1|p2)/)
    const projectId = projectMatch?.[1]
    if (projectId && path === `/api/projects/${projectId}/stages`) {
      const names = ['inspect_source', 'extract_frames', 'prepare_images', 'generate_feature_masks', 'generate_training_masks', 'extract_features', 'match_features', 'reconstruct', 'align_reconstruction', 'restore_metric_scale', 'position_ground', 'export_dataset']
      const extras: Record<string, Record<string, unknown>> = {
        inspect_source: { kind: 'insv', file_size: 1024, gravity_samples: 42 },
        extract_frames: { frames: 1, selection_mode: 'interval', selected: 1 },
        prepare_images: { images: 2, sources: 1, camera_groups: 1 },
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
        position_ground: { applied: true, ground_y: 5, support_points: 1200 },
        export_dataset: { images: 2, total_points: 42, validation: { loadable: true, training_ready: true }, lfstudio_training_metrics: 'external' },
      }
      await route.fulfill({ json: { project_id: 'p1', state: 'exported', stages: names.map(stage => ({
        stage, has_output: stage !== options.runningStage,
        status: stage === options.runningStage ? 'running' : 'succeeded', error_text: null,
        job_id: stage === options.runningStage ? 'running-job' : null,
        started_at: stage === options.runningStage ? '2026-01-01T00:00:00Z' : null,
        finished_at: null,
        params: paramsForStage(stage, DEFAULT_PARAMS, reconMode),
        extra: extras[stage],
        progress_event: stage === options.runningStage ? {
          id: 999, job_id: 'running-job', project_id: 'p1', stage, level: 'info',
          message: 'working', msg_key: options.runningMessageKey ?? null,
          msg_args: options.runningMessageKey ? { source: 'Primary 360', cur: 42, tot: 100 } : null,
          progress: options.runningProgress ?? null, kind: 'progress', ts: '2026-01-01T00:00:01Z',
        } : null,
      })) } })
      return
    }
    if (projectId && path === `/api/projects/${projectId}/source-info`) {
      await route.fulfill({ json: { id: 's1', duration_sec: 1, duration_sec_total: 1, fps: 30, width: 100, height: 100,
        sources: [{ id: 's1', duration_sec: 1, fps: 30, width: 100, height: 100 }] } })
      return
    }
    if (projectId && path === `/api/projects/${projectId}/reconstruction`) {
      await route.fulfill({ json: {
        cameras: [{ id: 1, model: options.cameraModel ?? (sourceKind === 'insv' ? 'OPENCV_FISHEYE' : 'EQUIRECTANGULAR'), width: 100, height: 100, params: [] }],
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
        projection: sourceKind === 'insv' ? 'dual_fisheye' : 'equirectangular',
        kind: sourceKind === 'insv' ? 'insv_dual' : 'equirectangular_video',
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
      const maskNames = sourceKind === 'insv' && reconMode !== 'pinhole_rig'
        ? ['sources/s1/front/frame_000000.jpg', 'sources/s1/back/frame_000000.jpg']
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
    if (projectId && path === `/api/projects/${projectId}/fisheye-region`) {
      await route.fulfill({ json: { lens0: { cx: 0.5, cy: 0.5, r: 0.48 }, lens1: { cx: 0.5, cy: 0.5, r: 0.48 }, saved: true, needs_review: true } })
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
    if (path.includes('/image') || path.includes('/prepared-image') || path.includes('/prepared-mask')) {
      const pixel = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M/wHwAF/gL+AvJkGQAAAABJRU5ErkJggg==', 'base64')
      await route.fulfill({ status: 200, contentType: 'image/png', body: pixel })
      return
    }
    if (projectId && path === `/api/projects/${projectId}/ui-state` && route.request().method() === 'PUT') {
      await route.fulfill({ json: {} })
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
    const sourceMutation = path.match(/^\/api\/projects\/(p1|p2)\/sources\/([^/]+)(\/make-primary)?$/)
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
    const rerun = path.match(/^\/api\/projects\/(p1|p2)\/rerun\/([^/]+)$/)
    if (rerun && route.request().method() === 'POST') {
      reruns.push({ stage: rerun[2], body: route.request().postDataJSON() })
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
  return { unexpectedRequests, reruns, sourceAdds, maskRequests }
}

test('IDE loads the split pipeline and environment diagnostics', async ({ page }) => {
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  const mock = await installUiMock(page)
  await page.goto('/')
  await expect(page.locator('h1')).toBeVisible()
  await expect(page.getByText('Extract features', { exact: true })).toBeVisible()
  await expect(page.getByText('Match features', { exact: true })).toBeVisible()
  await expect(page.getByText('Gravity alignment', { exact: true })).toBeVisible()
  await expect(page.getByText('Export', { exact: true })).toBeVisible()

  await page.getByTitle('Settings').click()
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

test('running stage restores numeric progress and activity from the stage snapshot', async ({ page }) => {
  const mock = await installUiMock(page, {
    runningStage: 'extract_frames',
    runningProgress: 0.42,
    runningMessageKey: 'log.extract_candidates_progress',
  })
  await page.goto('/')

  const stage = page.getByRole('button', { name: /Frame extraction 42%/ })
  await expect(stage).toBeVisible()
  await expect(stage.getByRole('progressbar', { name: 'Frame extraction' }))
    .toHaveAttribute('aria-valuenow', '42')
  await expect(page.getByText('Primary 360: candidate decode 42/100', { exact: true })).toBeVisible()
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
  expect(mock.unexpectedRequests).toEqual([])
})

test('activity-only events retain percentage and late events from an old job are ignored', async ({ page }) => {
  const mock = await installUiMock(page, {
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
  await expect(page.getByText('Primary 360: candidate scoring 43/100', { exact: true })).toBeVisible()
  await expect(page.getByText('old', { exact: true })).toHaveCount(0)
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

  await page.getByTitle('Settings').click()
  await page.locator('.pop select').nth(1).selectOption('zh')
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
  const feedback = page.locator('.scene-speed-feedback')
  await expect(feedback).toHaveText('0.5x')
  let feedbackBounds = await feedback.boundingBox()
  expect(feedbackBounds).not.toBeNull()
  expect(Math.abs(feedbackBounds!.x + feedbackBounds!.width / 2 - (bounds!.x + bounds!.width / 2))).toBeLessThan(3)
  expect(Math.abs(feedbackBounds!.y + feedbackBounds!.height / 2 - (bounds!.y + bounds!.height / 2))).toBeLessThan(3)

  await page.mouse.wheel(0, -100)
  await expect(feedback).toHaveText('1x')
  await page.mouse.wheel(0, -100)
  await expect(feedback).toHaveText('2x')
  expect(mock.unexpectedRequests).toEqual([])
})

test('fisheye valid-region editor can switch its preview frame', async ({ page }) => {
  const mock = await installUiMock(page, { frameCount: 3 })
  await page.goto('/')

  await page.getByRole('button', { name: /Fisheye region/ }).click()
  await expect(page.getByText('Coordinate update: review and save the circle')).toBeVisible()
  const previewFrame = page.getByRole('slider', { name: 'Preview frame' })
  await expect(previewFrame).toHaveAttribute('max', '2')
  await previewFrame.fill('2')
  await expect(page.locator('img[alt="lens0"]')).toHaveAttribute('src', /frames\/2\/image/)
  expect(mock.unexpectedRequests).toEqual([])
})

test('photo and dataset camera inspectors share capture summary fields', async ({ page }) => {
  test.setTimeout(60_000)
  const mock = await installUiMock(page)
  await page.goto('/')

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
  const camera = page.getByText('sources/s1/front/frame_000000.jpg', { exact: true })
  await expect(camera).toBeVisible()
  await camera.click()
  summary = page.getByTestId('capture-summary')
  await expect(summary).toBeVisible()
  await expect(summary.getByText('Frame', { exact: true })).toBeVisible()
  await expect(summary.getByText('Lens', { exact: true })).toBeVisible()
  await expect(summary.getByText('Reconstruction registration', { exact: true })).toBeVisible()
  await expect(summary.getByText('Image 3D points', { exact: true })).toBeVisible()
  await expect(page.getByText('Camera position', { exact: true })).toBeVisible()
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
  await expect(page.getByText('2048px', { exact: true })).toBeVisible()
  await featureEnabled.uncheck()
  await expect(page.getByRole('button', { name: /SAM3 feature masks skip/ })).toBeVisible()

  await page.getByRole('button', { name: /SAM3 training masks/ }).click()
  await expect(page.getByRole('checkbox', { name: 'Enable training masks' })).toBeChecked()
  await expect(page.getByText('Training-mask prompt', { exact: true }).locator('..').locator('input'))
    .toHaveValue(TRAINING_MASK_PROMPT)
  await expect(page.getByText('2048px', { exact: true })).toBeVisible()
  expect(mock.unexpectedRequests).toEqual([])
})

test('photo and dataset camera previews switch between both mask steps', async ({ page }) => {
  const mock = await installUiMock(page)
  await page.goto('/')

  await page.getByText('Frame 0', { exact: true }).click()
  let purpose = page.getByRole('group', { name: 'Mask purpose' })
  await expect(purpose.getByRole('button', { name: 'Training' })).toHaveAttribute('aria-pressed', 'true')
  await page.getByRole('tab', { name: 'Mask' }).click()
  await expect(page.locator('img[alt="Mask"]')).toHaveAttribute('src', /purpose=training/)
  await purpose.getByRole('button', { name: 'Feature' }).click()
  await expect(page.locator('img[alt="Mask"]')).toHaveAttribute('src', /purpose=feature/)

  await page.getByText(/Dataset · Cameras/).click()
  await page.getByText('sources/s1/front/frame_000000.jpg', { exact: true }).click()
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

  await page.getByText('Extract features', { exact: true }).click()
  await expect(page.getByText('Feature image-size limit (max_image_size)', { exact: true })).toBeVisible()
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

  await page.getByText('Sparse reconstruction', { exact: true }).click()
  await expect(page.getByText('Reconstruction solver (mapper)', { exact: true })).toBeVisible()
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
  await page.getByText('Position from predicted ground', { exact: true }).click()
  await expect(page.getByText('Predicts local ground near the primary camera path with equal weight per camera sample, then moves its median height to dataset Y=0.', { exact: true })).toBeVisible()
  await page.getByText('Export', { exact: true }).click()
  await expect(page.getByText('Optimize fisheye training images', { exact: true })).toBeVisible()
  await page.getByText('Sparse reconstruction', { exact: true }).click()

  await page.getByTitle('Settings').click()
  await page.locator('.pop select').nth(1).selectOption('zh')
  await expect(page.getByText('重建器 (mapper)', { exact: true })).toBeVisible()
  await expect(page.getByText('视图图校准 (view_graph_calibration)', { exact: true })).toBeVisible()
  await expect(page.getByText('GPU 光束平差 (ba_use_gpu)', { exact: true })).toBeVisible()
  await page.mouse.click(10, 200)
  await page.getByText('恢复真实大小', { exact: true }).click()
  await expect(page.getByText('真实大小恢复方式', { exact: true })).toBeVisible()
  await page.getByText('根据地面预测矫正位置', { exact: true }).first().click()
  await expect(page.getByText('在主相机路径附近按每个相机样本等权预测局部地面，并把其中位高度移动到数据集 Y=0。', { exact: true })).toBeVisible()
  await page.getByText('特征匹配', { exact: true }).click()
  await expect(page.getByText('特征匹配器 (matcher_type)', { exact: true })).toBeVisible()
  await expect(page.getByText('图像配对策略 (pairing)', { exact: true })).toBeVisible()
  await expect(page.getByText('双视图最少内点数 (two-view min_num_inliers)', { exact: true })).toBeVisible()
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

  const pipelineDeadline = Date.now() + 30 * 60_000
  while (true) {
    const statuses = await (await request.get(`/api/projects/${project!.id}/stages`)).json() as {
      stages: Array<{ stage: string; has_output: boolean }>
    }
    if (statuses.stages.find(item => item.stage === 'export_dataset')?.has_output) break
    expect(Date.now(), 'UI pipeline did not reach export before the deadline').toBeLessThan(pipelineDeadline)
    const nextButton = page.getByRole('button', { name: 'Generate next' })
    await expect(nextButton).toBeEnabled()
    const jobResponse = page.waitForResponse(response => (
      response.request().method() === 'POST' && response.url().includes('/rerun/')
    ), { timeout: 5_000 }).catch(() => null)
    await nextButton.click()
    const response = await jobResponse
    if (response) {
      const { job_id: jobId } = await response.json() as { job_id: string }
      let job: { status: string; error_text: string | null } = { status: 'queued', error_text: null }
      await expect.poll(async () => {
        job = await (await request.get(`/api/jobs/${jobId}`)).json() as typeof job
        return job.status
      }, { timeout: 15 * 60_000 }).toMatch(/succeeded|failed|cancelled/)
      expect(job.status, job.error_text ?? `job ${jobId} failed`).toBe('succeeded')
      await page.waitForTimeout(300)
      continue
    }
    const save = page.getByRole('button', { name: 'Save', exact: true })
    const manualStep = await save.waitFor({ state: 'visible', timeout: 3_000 }).then(() => true).catch(() => false)
    if (manualStep) {
      await save.click()
      continue
    }
  }

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
