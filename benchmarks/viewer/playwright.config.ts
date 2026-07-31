import { defineConfig, devices } from '@playwright/test'
import { fileURLToPath } from 'node:url'

const frontendDirectory = fileURLToPath(
  new URL('../../viewer/frontend', import.meta.url),
)
const baseURL = process.env.MAPPER_BENCH_BASE_URL ?? 'http://127.0.0.1:5173'
const apiTarget = process.env.MAPPER_BENCH_API_TARGET ?? 'http://127.0.0.1:8000'
const browserExecutable = process.env.MAPPER_BENCH_BROWSER_EXECUTABLE
const diagnosticRecording = process.env.MAPPER_BENCH_DIAGNOSTIC_RECORDING === '1'

export default defineConfig({
  testDir: '.',
  testMatch: 'viewer.benchmark.spec.ts',
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 120_000,
  expect: { timeout: 30_000 },
  outputDir: 'artifacts',
  reporter: [['line']],
  use: {
    ...devices['Desktop Chrome'],
    baseURL,
    launchOptions: {
      ...(browserExecutable ? { executablePath: browserExecutable } : {}),
      // The dedicated headed acceptance display can be occluded by the login
      // shell. Prevent Chromium from converting its render loop to 1 Hz while
      // the benchmark page is still the active test target.
      args: [
        '--disable-background-timer-throttling',
        '--disable-backgrounding-occluded-windows',
        '--disable-renderer-backgrounding',
        '--disable-features=CalculateNativeWinOcclusion',
      ],
    },
    // Continuous trace screenshots and video synchronously sample the WebGL
    // canvas on this acceptance host, perturbing frame timing by ~1 second.
    // Keep them opt-in for diagnosis; JSON metrics and scenario-owned
    // screenshots remain the acceptance evidence.
    trace: diagnosticRecording ? 'retain-on-failure' : 'off',
    screenshot: 'only-on-failure',
    video: diagnosticRecording ? 'retain-on-failure' : 'off',
  },
  projects: [{ name: 'chromium', use: { browserName: 'chromium' } }],
  webServer: process.env.MAPPER_BENCH_EXTERNAL_FRONTEND === '1'
    ? undefined
    : {
        command: 'npm run benchmark:serve',
        cwd: frontendDirectory,
        url: baseURL,
        reuseExistingServer: false,
        timeout: 120_000,
        env: {
          ...process.env,
          VITE_BENCHMARK_ENABLED: 'true',
          VITE_API_PROXY_TARGET: apiTarget,
          VITE_BASEMAP_ENABLED: process.env.VITE_BASEMAP_ENABLED ?? 'false',
        },
      },
})
