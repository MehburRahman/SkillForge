import { test, expect } from '@playwright/test'
import { mkdir, writeFile } from 'node:fs/promises'

test.skip(process.env.SKILLFORGE_REAL_MODEL !== '1', 'Opt in to real model calls.')

test('real Scala generation, compilation, evaluation, and saved browser history', async ({ page }, testInfo) => {
  await page.goto('/')
  await expect(page.getByText('API connected')).toBeVisible()
  await page.getByLabel('Problem statement').fill(
    'Read n followed by n signed integers from standard input. Print the maximum sum of a nonempty contiguous subarray. '
    + '1 <= n <= 200000, and each integer is between -1000000000 and 1000000000. '
    + 'Use a 64-bit sum. All-negative arrays must choose at least one element. Choose an efficient algorithm yourself.')
  await page.getByRole('combobox', { name: 'Language', exact: true }).selectOption('scala')
  await page.getByLabel('Generator model').fill(process.env.SKILLFORGE_REAL_MODEL_NAME || 'qwen2.5-coder:7b')
  await page.getByText('Run settings & tests').click()
  await page.getByLabel('Maximum attempts').fill('2')
  await page.getByLabel('Seed', { exact: false }).fill('42')
  await page.getByLabel('Tests (optional)', { exact: false }).fill(JSON.stringify([
    { input: '5\n-2 3 -1 4 -5\n', expected_output: '6', category: 'normal' },
    { input: '3\n-8 -2 -5\n', expected_output: '-2', category: 'edge' },
    { input: '3\n1000000000 1000000000 1000000000\n', expected_output: '3000000000', category: 'stress' },
  ]))
  await page.getByRole('button', { name: 'Start solve' }).click()
  await expect(page).toHaveURL(/#solve\//)
  await expect(page.getByText('Solve completed', { exact: true })).toBeVisible({ timeout: 270000 })
  const jobId = page.url().split('#solve/')[1]
  const response = await page.request.get(`/api/jobs/${jobId}`)
  const job = await response.json()
  expect(job.status).toBe('completed')
  expect(job.result.model_calls).toBeGreaterThanOrEqual(2)
  const runId = job.result.history_run_id
  const runResponse = await page.request.get(`/api/runs/${runId}`)
  const run = await runResponse.json()
  expect(run.language).toBe('scala')
  expect(run.execution_mode).toBe('stdio')
  expect(run.completed_at).toBeTruthy()
  expect(run.attempts[0].runtime_grade.execution_environment.scala).toBe('3.3.8')
  expect(run.attempts[0].runtime_grade.compilation_seconds).toBeGreaterThan(0)
  const record = JSON.stringify({ job, run }, null, 2)
  await writeFile(testInfo.outputPath('scala-real-result.json'), record)
  const archive = '../data/workspaces/scala-browser-smoke/runs'
  await mkdir(archive, { recursive: true })
  await writeFile(`${archive}/${runId}.json`, record)
  await testInfo.attach('real-model-outcome', { body: JSON.stringify(job.result), contentType: 'application/json' })
  await page.getByRole('button', { name: 'View saved run' }).click()
  await expect(page.getByRole('heading', { name: 'Solution history' })).toBeVisible()
  await expect(page.getByLabel('Candidate code', { exact: true })).not.toBeEmpty()
  await page.screenshot({ path: testInfo.outputPath('scala-real-history.png'), fullPage: true })
})
