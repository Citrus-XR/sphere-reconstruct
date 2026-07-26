import { expect, test, type Page } from '@playwright/test'
import path from 'node:path'
import { DEFAULT_PARAMS, paramsForStage, type ReconMode } from '../src/features/stageParams'

const MOCK_SOURCE = 'D:\\VID 2026\\clip.insv'
const MOCK_EXPORT = 'D:\\LFStudio\\export_dataset'

interface MockOptions {
  sourceKind?: 'insv' | 'erp_video'
  imageName?: string
  cameraModel?: string
  reconMode?: ReconMode
  denoiseMethod?: 'off' | 'fastdvdnet' | 'ffmpeg_adaptive'
  secondProject?: boolean
}

const installUiMock = async (page: Page, options: MockOptions = {}) => {
  const sourceKind = options.sourceKind ?? 'insv'
  const reconMode = options.reconMode ?? (sourceKind === 'insv' ? 'native_fisheye' : 'equirectangular')
  const imageName = options.imageName ?? (sourceKind === 'insv' ? 'front/frame_000000.jpg' : 'frame_000000.jpg')
  const unexpectedRequests: string[] = []
  const reruns: Array<{ stage: string; body: Record<string, Record<string, unknown>> }> = []
  const deletedProjects = new Set<string>()
  await page.addInitScript(() => {
    localStorage.clear()
    localStorage.setItem('lang', 'en')
  })
  await page.routeWebSocket('**/api/events**', () => {})
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
        source_kind: sourceKind, source_path: MOCK_SOURCE, state: 'exported',
        ui_state: options.reconMode || options.denoiseMethod ? {
          reconMode: options.reconMode,
          params: options.denoiseMethod ? { denoiseMethod: options.denoiseMethod } : {},
          disabled: [],
        } : null,
      }]
      if (options.secondProject) projects.push({
        id: 'p2', name: 'Second project', created_at: '2025-01-01T00:00:00Z', updated_at: '2025-01-01T00:00:00Z',
        source_kind: 'insv', source_path: 'D:\\second.insv', state: 'created', ui_state: null,
      })
      await route.fulfill({ json: projects.filter(project => !deletedProjects.has(project.id)) })
      return
    }
    if (path === '/api/settings') {
      await route.fulfill({ json: { sam3: { default_prompt: '' } } })
      return
    }
    if (path === '/api/system/doctor') {
      await route.fulfill({ json: { ready: true, platform: {}, checks: {
        workspace: { ok: true, message: 'ok', path: 'D:\\workspace' },
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
      await route.fulfill({ json: { path: 'D:\\', parent: null, dirs: [], files: [] } })
      return
    }
    const projectMatch = path.match(/^\/api\/projects\/(p1|p2)/)
    const projectId = projectMatch?.[1]
    if (projectId && path === `/api/projects/${projectId}/stages`) {
      const names = ['inspect_source', 'extract_frames', 'generate_masks', 'extract_features', 'match_features', 'reconstruct', 'align_reconstruction', 'denoise_frames', 'export_dataset']
      await route.fulfill({ json: { project_id: 'p1', state: 'exported', stages: names.map(stage => ({
        stage, has_output: true, status: 'succeeded', error_text: null, job_id: null,
        started_at: null, finished_at: null,
        params: stage === 'denoise_frames' ? { method: 'fastdvdnet' } : paramsForStage(stage, DEFAULT_PARAMS, reconMode),
        extra: null,
      })) } })
      return
    }
    if (projectId && path === `/api/projects/${projectId}/source-info`) {
      await route.fulfill({ json: { kind: sourceKind, duration_sec: 1, fps: 30, width: 100, height: 100 } })
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
      await route.fulfill({ json: { kind: sourceKind === 'insv' ? 'insv_dual' : 'erp_video', count: 1, width: 100, height: 100, fps: 30,
        selection: { mode: 'interval', selected: 1 }, frames: [{ index: 0, timestamp_sec: 0, score: { sharpness: 12 } }] } })
      return
    }
    if (projectId && path === `/api/projects/${projectId}/masks`) {
      await route.fulfill({ json: reconMode === 'pinhole_rig'
        ? { kind: 'sam3_pinhole_masks', frames: [{ index: 0, views: [
            { view: 'front', lens: 0, path: 'mask.png', coverage: 0.1 },
          ] }] }
        : sourceKind === 'insv'
        ? { kind: 'sam3_fisheye_masks', frames: [{ index: 0, lenses: [
            { lens: 0, path: 'mask.png', coverage: 0.1 }, { lens: 1, path: 'mask.png', coverage: 0.2 },
          ] }] }
        : { kind: 'sam3_erp_masks', frames: [{ index: 0, path: 'mask.png', coverage: 0.1 }] } })
      return
    }
    if (projectId && path === `/api/projects/${projectId}/denoise`) {
      await route.fulfill({ json: { method: 'fastdvdnet', device: 'cuda', frames: [
        sourceKind === 'insv'
          ? { index: 0, lens0: 'denoise/front/frame_000000.jpg', lens1: 'denoise/back/frame_000000.jpg' }
          : { index: 0, erp: 'denoise/frame_000000.jpg' },
      ] } })
      return
    }
    if (projectId && path === `/api/projects/${projectId}/fisheye-region`) {
      await route.fulfill({ json: { lens0: { cx: 0.5, cy: 0.5, r: 0.48 }, lens1: { cx: 0.5, cy: 0.5, r: 0.48 }, saved: true } })
      return
    }
    if (projectId && path === `/api/projects/${projectId}/export-info`) {
      await route.fulfill({ json: { dir: MOCK_EXPORT, dataset_dir: MOCK_EXPORT,
        preview_dir: `${MOCK_EXPORT}\\preview`, train_configs_dir: `${MOCK_EXPORT}\\train_configs`,
        training_output_dir: 'D:\\LFStudio\\training_outputs' } })
      return
    }
    if (path.includes('/image') || path.includes('/fisheye-mask/') || path.includes('/pinhole/')) {
      const pixel = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M/wHwAF/gL+AvJkGQAAAABJRU5ErkJggg==', 'base64')
      await route.fulfill({ status: 200, contentType: 'image/png', body: pixel })
      return
    }
    if (projectId && path === `/api/projects/${projectId}/ui-state` && route.request().method() === 'PUT') {
      await route.fulfill({ json: {} })
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
  return { unexpectedRequests, reruns }
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
  await expect(page.getByText('Training image denoise', { exact: true })).toBeVisible()

  await page.getByTitle('Settings').click()
  await expect(page.getByText(/Environment [✓⚠]/)).toBeVisible()
  expect(errors).toEqual([])
  expect(mock.unexpectedRequests).toEqual([])
})

test('scene copy follows language and paths remain copyable native values', async ({ page, context }) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write'])
  const mock = await installUiMock(page)
  await page.goto('/')

  await expect(page.getByText('Right-drag: look · WASD/arrows: move · Z/X: roll · F: focus')).toBeVisible()
  await expect(page.getByText(/Images 1 · Points 42 · Registered 100% · Path span 1\.250 units/)).toBeVisible()

  const sourcePath = page.locator('.ide-source-path')
  await expect(sourcePath.locator('.path-value')).toHaveText('D:/VID 2026/clip.insv')
  await expect(sourcePath).toHaveAttribute('data-copy-value', MOCK_SOURCE)
  await sourcePath.locator('.path-copy').click()
  await expect(sourcePath.locator('.path-copy')).toHaveClass(/copied/)
  await expect(sourcePath.getByRole('status')).toHaveText('Path copied')
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(MOCK_SOURCE)

  await page.getByText('Export', { exact: true }).click()
  await expect(page.getByText('D:/LFStudio/export_dataset', { exact: true }).first()).toBeVisible()
  const displayedPaths = (await page.locator('.path-value').allTextContents()).join('\n')
  expect(displayedPaths).not.toContain('¥')
  expect(displayedPaths).not.toContain('\\')

  await page.getByTitle('Settings').click()
  await page.locator('.pop select').nth(1).selectOption('zh')
  await expect(page.getByText('右键拖动: 视角 · WASD/方向键: 移动 · Z/X: 倾斜 · F: 聚焦')).toBeVisible()
  await expect(page.getByText(/图像 1 · 点 42 · 注册 100% · 轨迹范围 1\.250 单位/)).toBeVisible()
  await expect(page.getByText('场景视图', { exact: true }).first()).toBeVisible()
  expect(mock.unexpectedRequests).toEqual([])
})

test('photo and dataset camera inspectors share capture summary fields', async ({ page }) => {
  test.setTimeout(60_000)
  const mock = await installUiMock(page, { denoiseMethod: 'fastdvdnet' })
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
  await expect(page.getByText('Denoised', { exact: true })).toBeVisible()
  const frameDenoiseRequest = page.waitForRequest(request => request.url().includes('/denoise/0/image?lens=0'))
  await page.getByText('Denoised', { exact: true }).click()
  await frameDenoiseRequest

  const cameras = page.getByText(/Dataset · Cameras/)
  await expect(cameras).toBeVisible()
  await cameras.click()
  const camera = page.getByText('front/frame_000000.jpg', { exact: true })
  await expect(camera).toBeVisible()
  await camera.click()
  summary = page.getByTestId('capture-summary')
  await expect(summary).toBeVisible()
  await expect(summary.getByText('Frame', { exact: true })).toBeVisible()
  await expect(summary.getByText('Lens', { exact: true })).toBeVisible()
  await expect(summary.getByText('Reconstruction registration', { exact: true })).toBeVisible()
  await expect(summary.getByText('Image 3D points', { exact: true })).toBeVisible()
  await expect(page.getByText('Camera position', { exact: true })).toBeVisible()
  await expect(page.getByText('Denoised', { exact: true })).toBeVisible()
  expect(mock.unexpectedRequests).toEqual([])
})

test('temporal denoise can be configured from the stage inspector', async ({ page }) => {
  const mock = await installUiMock(page)
  await page.goto('/')

  await page.getByText('Training image denoise', { exact: true }).click()
  const method = page.getByText('Temporal denoise', { exact: true }).locator('..').locator('select')
  const runButton = page.getByRole('button', { name: 'Regenerate', exact: true })
  await expect(method).toHaveValue('off')
  await expect(runButton).toBeDisabled()

  await method.selectOption('fastdvdnet')
  await expect(page.getByText('Noise strength σ', { exact: true })).toBeVisible()
  await expect(page.getByText('GPU tile', { exact: true })).toBeVisible()
  await expect(page.getByText('LFStudio export: denoised images (camera poses unchanged)')).toBeVisible()
  await expect(runButton).toBeEnabled()
  await runButton.click()
  await expect.poll(() => mock.reruns.length).toBe(1)
  expect(mock.reruns[0]).toEqual({
    stage: 'denoise_frames',
    body: { params_by_stage: { denoise_frames: {
      method: 'fastdvdnet', sigma: 10, tile_size: 512, tile_overlap: 80, jpeg_quality: 98,
    } } },
  })
  expect(mock.unexpectedRequests).toEqual([])
})

test('denoise applicability and export prerequisite follow current settings', async ({ page }) => {
  const mock = await installUiMock(page)
  await page.goto('/')

  await page.getByRole('button', { name: /Frame 0/ }).click()
  await expect(page.getByRole('tab', { name: 'Denoised' })).toHaveCount(0)
  const denoiseStep = page.getByRole('button', { name: /Training image denoise skip/ })
  await expect(denoiseStep).toBeVisible()
  await denoiseStep.click()
  await page.getByLabel('Temporal denoise').selectOption('fastdvdnet')

  await page.getByRole('button', { name: /Export/ }).click()
  await expect(page.getByText('Generate training-image denoise with the current settings first.')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Regenerate', exact: true })).toBeDisabled()
  expect(mock.unexpectedRequests).toEqual([])
})

test('switching projects resets denoise and inspector selection state', async ({ page }) => {
  const mock = await installUiMock(page, { secondProject: true })
  await page.goto('/')

  await page.getByRole('button', { name: /Training image denoise skip/ }).click()
  await page.getByLabel('Temporal denoise').selectOption('fastdvdnet')
  await page.getByRole('button', { name: /Frame 0/ }).click()
  await expect(page.getByTestId('capture-summary')).toBeVisible()

  await page.getByRole('button', { name: /Projects/ }).click()
  await page.getByText('Second project', { exact: true }).click()
  await expect(page.locator('h1')).toHaveText('Second project')
  await expect(page.getByTestId('capture-summary')).toHaveCount(0)
  await page.getByRole('button', { name: /Training image denoise skip/ }).click()
  await expect(page.getByLabel('Temporal denoise')).toHaveValue('off')
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
    imageName: 'frame_000000.jpg',
    cameraModel: 'EQUIRECTANGULAR',
    denoiseMethod: 'fastdvdnet',
  })
  await page.goto('/')
  await page.getByRole('button', { name: /Dataset · Cameras/ }).click()
  await page.getByRole('button', { name: 'frame_000000.jpg' }).click()
  await expect(page.locator('.inspector-preview-image')).toHaveAttribute('src', /\/frames\/0\/image\?lens=0$/)
  const erpDenoise = page.waitForRequest(request => request.url().includes('/denoise/0/image?lens=0'))
  await page.getByRole('tab', { name: 'Denoised' }).click()
  await erpDenoise
  expect(erpMock.unexpectedRequests).toEqual([])
})

test('pinhole dataset camera uses the reprojected view endpoint', async ({ page }) => {
  const mock = await installUiMock(page, {
    imageName: 'front_lens0/frame_000000.jpg',
    cameraModel: 'PINHOLE',
    reconMode: 'pinhole_rig',
  })
  await page.goto('/')
  await page.getByRole('button', { name: /Dataset · Cameras/ }).click()
  await page.getByRole('button', { name: 'front_lens0/frame_000000.jpg' }).click()
  await expect(page.locator('.inspector-preview-image')).toHaveAttribute('src', /\/pinhole\/0\/front\?lens=0$/)
  await expect(page.getByRole('tab', { name: 'Denoised' })).toHaveCount(0)
  expect(mock.unexpectedRequests).toEqual([])
})

test('stage rows and source dialog support keyboard navigation', async ({ page }) => {
  const mock = await installUiMock(page)
  await page.goto('/')

  const sourceStep = page.getByRole('button', { name: /Source done/ })
  await sourceStep.focus()
  await page.keyboard.press('Enter')
  await page.getByRole('button', { name: 'Select source…' }).click()
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
  await page.getByRole('button', { name: 'Select source…' }).click()
  const dialog = page.getByRole('dialog', { name: 'Select source…' })
  await dialog.getByRole('button', { name: root.replaceAll('\\', '/') }).click()
  const relativeDirectory = filesystem.relative(root, filesystem.dirname(source))
  for (const directory of relativeDirectory.split(/[\\/]/).filter(Boolean)) {
    await dialog.locator('.fs-main').filter({ hasText: directory }).click()
  }
  const sourceSet = page.waitForResponse(response => (
    response.request().method() === 'POST' && response.url().endsWith(`/api/projects/${project!.id}/source`)
  ))
  await dialog.locator('.fs-row-button').filter({ hasText: filesystem.basename(source) }).click()
  await sourceSet
  await expect(dialog).toHaveCount(0)

  for (let step = 0; step < 10; step += 1) {
    const statuses = await (await request.get(`/api/projects/${project!.id}/stages`)).json() as {
      stages: Array<{ stage: string; has_output: boolean }>
    }
    if (statuses.stages.find(item => item.stage === 'export_dataset')?.has_output) break
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
  const exportInfo = await (await request.get(`/api/projects/${project!.id}/export-info`)).json() as {
    dir: string; dataset_dir: string
  }
  expect(exportInfo.dataset_dir).toBe(exportInfo.dir)
  if (process.env.SPHERE_E2E_KEEP !== '1') {
    await request.delete(`/api/projects/${project!.id}`)
  } else {
    console.log(`E2E_PROJECT_ID=${project!.id}`)
  }
})
