import { defineConfig } from '@playwright/test'
import { existsSync } from 'node:fs'

const macChrome = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
const frontendPort = process.env.SKILLFORGE_TEST_FRONTEND_PORT || '5173'
const frontendURL = `http://127.0.0.1:${frontendPort}`
export default defineConfig({
  testDir: './e2e',
  workers: 1,
  timeout: 30000,
  use: {
    baseURL: frontendURL, viewport: { width: 1440, height: 1000 },
    launchOptions: { executablePath: process.env.CHROME_PATH || (existsSync(macChrome) ? macChrome : undefined) },
    screenshot: 'only-on-failure', trace: 'retain-on-failure',
  },
  webServer: [
    { command: `${process.env.SKILLFORGE_PYTHON || '../../venv/bin/python'} ../tests/serve_frontend_fixture.py --frontend-origin ${frontendURL}`, url: 'http://127.0.0.1:18765/health', reuseExistingServer: false },
    { command: `npm run dev -- --port ${frontendPort} --strictPort`, url: frontendURL, env: { SKILLFORGE_API_URL: 'http://127.0.0.1:18765' }, reuseExistingServer: false },
  ],
})
