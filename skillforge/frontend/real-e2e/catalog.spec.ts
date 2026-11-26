import { test, expect } from '@playwright/test'
import { mkdir, readFile, rm, writeFile } from 'node:fs/promises'
import { existsSync, readFileSync } from 'node:fs'

const mode = process.env.SKILLFORGE_EVALUATION_MODE || ''
const catalogs: Record<string, string> = { stdio: 'scala_algorithms', function: 'scala_functions', http: 'scala_http' }
const catalog = catalogs[mode]
const arm = process.env.SKILLFORGE_EVALUATION_ARM || 'baseline'
const archive = `../data/workspaces/scala-evaluation-${mode}-${arm}`
const selected = process.env.SKILLFORGE_EVALUATION_TASK_IDS?.split(',')
type TaskMetadata = { task_id: string; title: string; split: string }
const metadata: TaskMetadata[] = catalog
  ? JSON.parse(readFileSync(`../data/${catalog}.json`, 'utf8')).tasks.map((task: TaskMetadata) => ({
    task_id: task.task_id, title: task.title, split: task.split,
  })) : []

test.skip(process.env.SKILLFORGE_REAL_MODEL !== '1' || !catalog, 'Opt in to a Scala catalog evaluation.')

for (const task of metadata.filter(task => !selected || selected.includes(task.task_id))) {
  test(`${mode}: ${task.task_id}`, async ({ page }, testInfo) => {
    test.setTimeout(600000)
    const resultPath = `${archive}/runs/${task.task_id}.json`
    const pendingPath = `${archive}/runs/${task.task_id}.pending.json`
    const manifest = JSON.parse(await readFile(`${archive}/manifest.json`, 'utf8'))
    if (existsSync(resultPath)) {
      const saved = JSON.parse(await readFile(resultPath, 'utf8'))
      expect(saved.manifest).toEqual(manifest)
      if (saved.job.status === 'completed') {
        testInfo.annotations.push({ type: 'saved run', description: saved.run.run_id })
        if (!existsSync(`${archive}/runs/${task.task_id}.png`)) {
          await page.goto(`/#runs/${saved.run.run_id}`)
          await expect(page.getByRole('heading', { name: task.task_id, exact: true })).toBeVisible()
          await page.screenshot({ path: `${archive}/runs/${task.task_id}.png`, fullPage: true })
        }
        return
      }
      throw new Error(`Saved workflow ${saved.job.status}; use a new arm to retry without overwriting evidence.`)
    }
    await mkdir(`${archive}/runs`, { recursive: true })
    let queued: { job_id: string }
    if (existsSync(pendingPath)) {
      const pending = JSON.parse(await readFile(pendingPath, 'utf8'))
      expect(pending.task_id).toBe(task.task_id)
      queued = pending
      await page.goto(`/#solve/${queued.job_id}`)
      const saved = await page.request.get(`/api/jobs/${queued.job_id}`)
      expect(saved.ok()).toBe(true)
      expect((await saved.json()).request).toMatchObject({
        task_id: task.task_id, catalog, model: manifest.model, config: 'full', seed: 42, max_attempts: 3,
      })
    } else {
      await page.goto('/')
      await expect(page.getByText('API connected')).toBeVisible()
      await page.getByRole('button', { name: 'Choose from catalog' }).click()
      await page.getByRole('combobox', { name: 'Catalog', exact: true }).selectOption(catalog)
      await page.getByRole('combobox', { name: 'Task', exact: true }).selectOption(task.task_id)
      await page.getByLabel('Generator model').fill(manifest.model)
      await page.getByText('Run settings & tests').click()
      await page.getByLabel('Maximum attempts').fill('3')
      await page.getByLabel('Seed', { exact: false }).fill('42')
      await page.getByRole('combobox', { name: /^Experiment strategy/ }).selectOption('full')
      const submitted = page.waitForResponse(response => response.url().endsWith('/api/solves') && response.request().method() === 'POST')
      await page.getByRole('button', { name: 'Start solve' }).click()
      const response = await submitted
      expect(response.status()).toBe(202)
      queued = await response.json()
      await writeFile(pendingPath, JSON.stringify({ task_id: task.task_id, job_id: queued.job_id }))
    }
    let job: any
    await expect.poll(async () => {
      job = await (await page.request.get(`/api/jobs/${queued.job_id}`)).json()
      return ['completed', 'failed', 'interrupted'].includes(job.status)
    }, { timeout: 540000, intervals: [1000, 3000] }).toBe(true)
    const run = job.resource_id ? await (await page.request.get(`/api/runs/${job.resource_id}`)).json() : null
    const record = { task, manifest, job, run }
    await writeFile(resultPath, JSON.stringify(record, null, 2))
    await rm(pendingPath, { force: true })
    await writeFile(testInfo.outputPath('result.json'), JSON.stringify(record, null, 2))
    expect(job.status, job.error || 'Model workflow must complete independently of its solution score.').toBe('completed')
    expect(run.language).toBe('scala')
    expect(run.execution_mode).toBe(mode)
    expect(run.config_name).toBe('full')
    expect(run.seed).toBe(42)
    expect(run.test_source).toBe('trusted')
    expect(run.completed_at).toBeTruthy()
    await page.getByRole('button', { name: 'View saved run' }).click()
    await expect(page.getByRole('heading', { name: task.task_id, exact: true })).toBeVisible()
    await expect(page.getByLabel('Candidate code', { exact: true })).not.toBeEmpty()
    await page.screenshot({ path: `${archive}/runs/${task.task_id}.png`, fullPage: true })
    console.log(`${task.task_id}: first=${run.attempts[0].functional_grade.score}, final=${job.result.evaluation.functional_grade.score}, attempts=${job.result.attempt_count}, calls=${job.result.model_calls}`)
  })
}
