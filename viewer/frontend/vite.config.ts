import { defineConfig } from 'vitest/config'
import { loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const environment = loadEnv(mode, '.', '')
  const apiTarget = environment.VITE_API_PROXY_TARGET ?? 'http://127.0.0.1:8000'
  const proxy = {
    '/api': apiTarget,
    '/health': apiTarget,
  }
  return {
    plugins: [react()],
    server: { proxy },
    preview: { proxy },
    test: {
      environment: 'jsdom',
      setupFiles: './src/test/setup.ts',
      css: true,
    },
  }
})
