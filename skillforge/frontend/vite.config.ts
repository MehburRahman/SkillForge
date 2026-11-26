import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

const proxy = {
  '/api': {
    target: process.env.SKILLFORGE_API_URL || 'http://127.0.0.1:8000',
    changeOrigin: true,
    rewrite: (path: string) => path.replace(/^\/api/, ''),
  },
}

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { host: '127.0.0.1', port: 5173, strictPort: true, proxy },
  preview: { host: '127.0.0.1', port: 5173, strictPort: true, proxy },
  test: { include: ['src/**/*.test.ts', 'src/**/*.test.tsx'], environment: 'jsdom', setupFiles: ['./src/test-setup.ts'], restoreMocks: true },
})
