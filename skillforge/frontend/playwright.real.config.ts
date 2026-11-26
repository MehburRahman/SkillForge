import { defineConfig } from '@playwright/test'
import { existsSync } from 'node:fs'

const macChrome = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
export default defineConfig({
  testDir: './real-e2e', outputDir: './real-test-results', workers: 1, timeout: 300000,
  use: {
    baseURL: 'http://127.0.0.1:5174', viewport: { width: 1440, height: 1000 },
    actionTimeout: 15000, navigationTimeout: 30000,
    launchOptions: { executablePath: process.env.CHROME_PATH || (existsSync(macChrome) ? macChrome : undefined) },
    screenshot: 'only-on-failure', trace: 'retain-on-failure',
  },
  webServer: [
    { command: `${process.env.SKILLFORGE_PYTHON || '../../venv/bin/python'} ../tests/serve_real_scala.py`,
      url: 'http://127.0.0.1:18766/health', reuseExistingServer: false, timeout: 30000 },
    { command: 'npm run dev -- --port 5174', url: 'http://127.0.0.1:5174',
      env: { SKILLFORGE_API_URL: 'http://127.0.0.1:18766' }, reuseExistingServer: false },
  ],
})
