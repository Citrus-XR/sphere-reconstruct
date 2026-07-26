import { expect, test } from '@playwright/test'

test('IDE loads the split pipeline and environment diagnostics', async ({ page }) => {
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  await page.goto('/')
  await expect(page.locator('h1')).toBeVisible()
  await expect(page.getByText('Extract features', { exact: true })).toBeVisible()
  await expect(page.getByText('Match features', { exact: true })).toBeVisible()
  await expect(page.getByText('Gravity alignment', { exact: true })).toBeVisible()

  await page.getByTitle('Settings').click()
  await expect(page.getByText(/Environment [✓⚠]/)).toBeVisible()
  expect(errors).toEqual([])
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
  await request.post(`/api/projects/${project!.id}/source`, {
    data: { kind: 'insv', path: source },
  })
  await page.reload()

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
