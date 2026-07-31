import { expect, type Browser, type CDPSession, type Page, type Request } from '@playwright/test'
import { execFileSync } from 'node:child_process'
import { mkdir, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

export const LIMITS = {
  coldFirstGeometryMs: 2_500,
  warmFirstGeometryMs: 1_000,
  visiblePoints: 2_000_000,
  poolBytes: 256 * 1024 * 1024,
  longTaskMs: 50,
} as const

export interface NetworkRecord {
  url: string
  method: string
  resourceType: string
  range: string | null
  startedAt: number
  status?: number
  contentRange?: string | null
  failure?: string | null
  completedAt?: number
}

export interface BrowserMetrics {
  frameTimesMs: number[]
  longTasks: Array<{ startTime: number; duration: number }>
}

export interface BenchmarkReport {
  schemaVersion: 1
  scenario: string
  status: 'passed' | 'skipped'
  skipReason?: string
  identity: {
    commit: string
    browser: string
    platform: string
    hostname: string
    cpu: string
    cpuCount: number
    totalMemoryBytes: number
    node: string
    webglRenderer: string
  }
  configuration: {
    baseURL: string
    siteAArtifactId: string
    siteBArtifactId: string
    comparisonArtifactId: string
    evidenceMode: 'acceptance' | 'functional'
  }
  timings: Record<string, number>
  performanceMarks: Array<{ name: string; startTime: number }>
  network: NetworkRecord[]
  metrics: {
    frameTimeP95Ms: number
    frameTimesMs: number[]
    longTasks: Array<{ startTime: number; duration: number }>
    visiblePointCount: number
    cpuPoolBytes: number
    gpuPoolBytes: number
  }
  behavior: {
    navigationCount: number
    abortedRequestCount: number
    inactiveSiteRequestCount: number
    rendererGeneration: number
    disposedRendererGenerations: number[]
    activeArtifactIds: string[]
  }
}

interface BrowserProbe {
  frameTimesMs: number[]
  longTasks: Array<{ startTime: number; duration: number }>
  lastFrameAt: number | null
  running: boolean
}

const requestedEvidenceMode = process.env.MAPPER_BENCH_EVIDENCE_MODE
  ?? (process.env.MAPPER_BENCH_FUNCTIONAL_ONLY === '1' ? 'functional' : 'acceptance')
if (!['acceptance', 'functional'].includes(requestedEvidenceMode)) {
  throw new Error('MAPPER_BENCH_EVIDENCE_MODE must be acceptance or functional')
}

export const benchmarkConfiguration = {
  baseURL: process.env.MAPPER_BENCH_BASE_URL ?? 'http://127.0.0.1:5173',
  siteAArtifactId: process.env.MAPPER_BENCH_SITE_A ?? 'artifact_phase2_site_a',
  siteBArtifactId: process.env.MAPPER_BENCH_SITE_B ?? 'artifact_phase2_site_b',
  comparisonArtifactId: process.env.MAPPER_BENCH_COMPARE_ARTIFACT_ID
    ?? 'artifact_phase2_site_a_comparison',
  evidenceMode: requestedEvidenceMode as 'acceptance' | 'functional',
}

/**
 * Software rendering can exercise transitions, range traffic, disposal and
 * cancellation, but it is not valid evidence for the GTX 1070 timing gates.
 */
export const performanceAssertionsEnabled =
  benchmarkConfiguration.evidenceMode === 'acceptance'

export async function installBrowserProbe(page: Page): Promise<void> {
  await page.bringToFront()
  const lifecycle = await page.context().newCDPSession(page)
  await lifecycle.send('Emulation.setFocusEmulationEnabled', { enabled: true })
  await lifecycle.send('Page.setWebLifecycleState', { state: 'active' })
  await lifecycle.detach()
  await page.addInitScript(() => {
    const probe: BrowserProbe = {
      frameTimesMs: [],
      longTasks: [],
      lastFrameAt: null,
      running: true,
    }
    ;(window as unknown as { __MAPPER_BROWSER_PROBE__: BrowserProbe }).__MAPPER_BROWSER_PROBE__ = probe
    const frame = (now: number) => {
      if (probe.lastFrameAt !== null) probe.frameTimesMs.push(now - probe.lastFrameAt)
      probe.lastFrameAt = now
      if (probe.running) requestAnimationFrame(frame)
    }
    requestAnimationFrame(frame)
    new PerformanceObserver(list => {
      for (const entry of list.getEntries()) {
        probe.longTasks.push({ startTime: entry.startTime, duration: entry.duration })
      }
    }).observe({ type: 'longtask', buffered: true })
  })
}

export async function resetBrowserProbe(page: Page): Promise<void> {
  await page.evaluate(() => {
    const probe = (window as unknown as { __MAPPER_BROWSER_PROBE__: BrowserProbe }).__MAPPER_BROWSER_PROBE__
    probe.frameTimesMs = []
    probe.longTasks = []
    probe.lastFrameAt = null
  })
}

export async function readBrowserProbe(page: Page): Promise<BrowserMetrics> {
  return page.evaluate(() => {
    const probe = (window as unknown as { __MAPPER_BROWSER_PROBE__: BrowserProbe }).__MAPPER_BROWSER_PROBE__
    return {
      frameTimesMs: [...probe.frameTimesMs],
      longTasks: [...probe.longTasks],
    }
  })
}

export async function readWebglRenderer(page: Page): Promise<string> {
  return page.evaluate(() => {
    const canvas = document.createElement('canvas')
    const gl = canvas.getContext('webgl2') ?? canvas.getContext('webgl')
    if (!gl) return 'unavailable'
    const extension = gl.getExtension('WEBGL_debug_renderer_info')
    if (!extension) return String(gl.getParameter(gl.RENDERER) ?? 'unavailable')
    return String(gl.getParameter(extension.UNMASKED_RENDERER_WEBGL) ?? 'unavailable')
  })
}

export function isHardwareWebglRenderer(renderer: string): boolean {
  return !/(?:swiftshader|software|llvmpipe|lavapipe|softpipe|mesa offscreen|unavailable)/i
    .test(renderer)
}

export function recordNetwork(page: Page): {
  records: NetworkRecord[]
  navigationCount: () => number
} {
  const records: NetworkRecord[] = []
  const byRequest = new Map<Request, NetworkRecord>()
  let documents = 0
  page.on('request', request => {
    if (request.resourceType() === 'document') documents += 1
    const record: NetworkRecord = {
      url: request.url(),
      method: request.method(),
      resourceType: request.resourceType(),
      range: request.headers()['range'] ?? null,
      startedAt: Date.now(),
    }
    records.push(record)
    byRequest.set(request, record)
  })
  page.on('response', response => {
    const record = byRequest.get(response.request())
    if (!record) return
    record.status = response.status()
    record.contentRange = response.headers()['content-range'] ?? null
    record.completedAt = Date.now()
  })
  page.on('requestfailed', request => {
    const record = byRequest.get(request)
    if (record) record.failure = request.failure()?.errorText ?? 'request failed'
  })
  return { records, navigationCount: () => documents }
}

export async function createCdpSession(page: Page): Promise<CDPSession> {
  const session = await page.context().newCDPSession(page)
  await session.send('Network.enable')
  return session
}

export async function setNetworkConditions(
  session: CDPSession,
  options: { cacheDisabled?: boolean; throttled?: boolean },
): Promise<void> {
  if (options.cacheDisabled !== undefined) {
    await session.send('Network.setCacheDisabled', {
      cacheDisabled: options.cacheDisabled,
    })
  }
  if (options.throttled !== undefined) {
    await session.send('Network.emulateNetworkConditions', options.throttled
      ? {
          offline: false,
          latency: 350,
          downloadThroughput: 300 * 1024 / 8,
          uploadThroughput: 150 * 1024 / 8,
          connectionType: 'cellular3g',
        }
      : {
          offline: false,
          latency: 0,
          downloadThroughput: -1,
          uploadThroughput: -1,
          connectionType: 'none',
        })
  }
}

export async function waitForOverview(page: Page): Promise<void> {
  await expect(page.getByTestId('viewer-app')).toHaveAttribute('data-mode', 'overview')
  await expect(
    page.getByTestId(`open-artifact-${benchmarkConfiguration.siteAArtifactId}`),
  ).toBeVisible()
  await waitForOverviewRendererReady(page)
}

export async function waitForOverviewRendererReady(page: Page): Promise<void> {
  await page.waitForFunction(() => {
    const events = (window as unknown as {
      __MAPPER_BENCHMARK__?: {
        events: Array<{ name: string; detail?: Record<string, unknown> }>
      }
    }).__MAPPER_BENCHMARK__?.events ?? []
    const created = [...events].reverse().find(event =>
      event.name === 'renderer-created' && event.detail?.mode === 'overview')
    if (!created) return false
    return events.some(event =>
      event.name === 'overview-ready'
      && event.detail?.generation === created.detail?.generation)
  }, undefined, { timeout: 30_000 })
}

export async function openArtifact(page: Page, artifactId: string): Promise<number> {
  await page.getByTestId(`open-artifact-${artifactId}`).click()
  await expect(page.getByTestId('viewer-app')).toHaveAttribute('data-mode', 'detail')
  await page.waitForFunction(id => {
    const events = (window as unknown as {
      __MAPPER_BENCHMARK__?: {
        events: Array<{
          name: string
          at: number
          detail?: Record<string, unknown>
        }>
      }
    }).__MAPPER_BENCHMARK__?.events ?? []
    const started = [...events].reverse().find(event =>
      event.name === 'open-artifact-start' && event.detail?.artifactId === id)
    if (!started) return false
    return events.some(event =>
      event.name === 'first-geometry-visible'
      && event.at >= started.at
      && Array.isArray(event.detail?.activeArtifactIds)
      && event.detail.activeArtifactIds.includes(id))
  }, artifactId, { timeout: 60_000 })
  const elapsed = await page.evaluate(id => {
    const events = (window as unknown as {
      __MAPPER_BENCHMARK__: {
        events: Array<{
          name: string
          at: number
          detail?: Record<string, unknown>
        }>
      }
    }).__MAPPER_BENCHMARK__.events
    const started = [...events].reverse().find(event =>
      event.name === 'open-artifact-start' && event.detail?.artifactId === id)
    if (!started) throw new Error(`Missing open-artifact-start event for ${id}`)
    const ready = events.find(event =>
      event.name === 'first-geometry-visible'
      && event.at >= started.at
      && Array.isArray(event.detail?.activeArtifactIds)
      && event.detail.activeArtifactIds.includes(id))
    if (!ready) throw new Error(`Missing first-geometry-visible event for ${id}`)
    return ready.at - started.at
  }, artifactId)
  await page.waitForFunction(id => {
    const snapshot = (window as unknown as {
      __MAPPER_BENCHMARK__?: {
        snapshot: {
          activeArtifactIds: string[]
          performance: { visiblePoints: number }
        }
      }
    }).__MAPPER_BENCHMARK__?.snapshot
    return snapshot?.activeArtifactIds.includes(id)
      && snapshot.performance.visiblePoints > 0
  }, artifactId, { timeout: 30_000 })
  await page.evaluate(
    ({ id, value }) => performance.mark('benchmark:first-geometry', {
      detail: { artifactId: id, elapsedMs: value },
    }),
    { id: artifactId, value: elapsed },
  )
  return elapsed
}

export async function enterOverview(page: Page): Promise<void> {
  await page.getByTestId('enter-overview').click()
  await waitForOverview(page)
}

export async function waitForTrajectory(page: Page, artifactId: string): Promise<void> {
  await page.waitForFunction(id => {
    const events = (window as unknown as {
      __MAPPER_BENCHMARK__?: {
        events: Array<{ name: string; detail?: Record<string, unknown> }>
      }
    }).__MAPPER_BENCHMARK__?.events ?? []
    return events.some(event => event.name === 'trajectory-ready'
      && event.detail?.artifactId === id
      && Number(event.detail?.pointCount) >= 2)
  }, artifactId, { timeout: 30_000 })
}

export async function pickVisiblePoint(page: Page): Promise<void> {
  const canvas = page.getByTestId('viewer-canvas')
  const box = await canvas.boundingBox()
  if (!box) throw new Error('viewer canvas has no layout box')
  const offsets = [
    [0.5, 0.25],
    [0.5, 0.33],
    [0.45, 0.25],
    [0.55, 0.25],
    [0.45, 0.33],
    [0.55, 0.33],
    [0.4, 0.2],
    [0.5, 0.2],
    [0.6, 0.2],
    [0.4, 0.4],
    [0.5, 0.4],
    [0.6, 0.4],
    [0.4, 0.5],
    [0.5, 0.5],
    [0.6, 0.5],
  ]
  for (const [x, y] of offsets) {
    await page.mouse.dblclick(box.x + box.width * x, box.y + box.height * y)
    try {
      await expect(page.getByTestId('inspection-panel')).toBeVisible({ timeout: 750 })
      return
    } catch {
      // Try a nearby screen-space sample; sparse coarse geometry may not cover
      // the exact viewport center.
    }
  }
  throw new Error('Giro3D picking returned no visible point after transition')
}

export function p95(values: number[]): number {
  if (!values.length) return 0
  const ordered = [...values].sort((left, right) => left - right)
  return ordered[Math.ceil(ordered.length * 0.95) - 1]
}

export async function finalizeReport(options: {
  scenario: string
  status?: 'passed' | 'skipped'
  skipReason?: string
  page: Page
  browser: Browser
  network: NetworkRecord[]
  navigationCount: number
  timings?: Record<string, number>
  inactiveSiteAssetUrl?: string
}): Promise<BenchmarkReport> {
  const [probe, bridge, marks, webglRenderer] = await Promise.all([
    readBrowserProbe(options.page),
    options.page.evaluate(() => (window as unknown as {
      __MAPPER_BENCHMARK__?: {
        snapshot: {
          performance: {
            visiblePoints: number
            cpuGeometryBytes: number
            gpuGeometryBytes: number
          }
          rendererGeneration: number
          disposedRendererGenerations: number[]
          activeArtifactIds: string[]
        }
      }
    }).__MAPPER_BENCHMARK__),
    options.page.evaluate(() => performance.getEntriesByType('mark').map(entry => ({
      name: entry.name,
      startTime: entry.startTime,
    }))),
    readWebglRenderer(options.page),
  ])
  if (!bridge) throw new Error('VITE_BENCHMARK_ENABLED=true is required')
  const performanceState = bridge.snapshot.performance
  const report: BenchmarkReport = {
    schemaVersion: 1,
    scenario: options.scenario,
    status: options.status ?? 'passed',
    skipReason: options.skipReason,
    identity: {
      commit: commitIdentity(),
      browser: options.browser.version(),
      platform: `${os.platform()} ${os.release()} ${os.arch()}`,
      hostname: os.hostname(),
      cpu: os.cpus()[0]?.model ?? 'unknown',
      cpuCount: os.cpus().length,
      totalMemoryBytes: os.totalmem(),
      node: process.version,
      webglRenderer,
    },
    configuration: benchmarkConfiguration,
    timings: options.timings ?? {},
    performanceMarks: marks,
    network: options.network,
    metrics: {
      frameTimeP95Ms: p95(probe.frameTimesMs),
      frameTimesMs: probe.frameTimesMs,
      longTasks: probe.longTasks,
      visiblePointCount: performanceState.visiblePoints,
      cpuPoolBytes: performanceState.cpuGeometryBytes,
      gpuPoolBytes: performanceState.gpuGeometryBytes,
    },
    behavior: {
      navigationCount: options.navigationCount,
      abortedRequestCount: options.network.filter(item => item.failure?.includes('ERR_ABORTED')).length,
      inactiveSiteRequestCount: options.inactiveSiteAssetUrl
        ? options.network.filter(item => item.url === options.inactiveSiteAssetUrl).length
        : 0,
      rendererGeneration: bridge.snapshot.rendererGeneration,
      disposedRendererGenerations: bridge.snapshot.disposedRendererGenerations,
      activeArtifactIds: bridge.snapshot.activeArtifactIds,
    },
  }
  const resultDirectory = process.env.MAPPER_BENCH_REPORT_DIR
    ?? path.join(fileURLToPath(new URL('.', import.meta.url)), 'results')
  await mkdir(resultDirectory, { recursive: true })
  await writeFile(
    path.join(resultDirectory, `${options.scenario}.json`),
    `${JSON.stringify(report, null, 2)}\n`,
    'utf8',
  )
  if (performanceAssertionsEnabled && !isHardwareWebglRenderer(webglRenderer)) {
    throw new Error(
      `acceptance evidence requires hardware WebGL; reported renderer: ${webglRenderer}`,
    )
  }
  return report
}

export async function pointAssetUrl(page: Page, artifactId: string): Promise<string> {
  const response = await page.request.get(
    `/api/v1/catalog/artifacts/${encodeURIComponent(artifactId)}`,
  )
  expect(response.ok()).toBeTruthy()
  const detail = await response.json() as {
    representations: Array<{ kind: string; format: string; asset_url: string }>
  }
  const representation = detail.representations.find(
    item => item.kind === 'points' && item.format.toLowerCase().includes('copc'),
  )
  if (!representation) throw new Error(`artifact ${artifactId} has no COPC asset`)
  return new URL(representation.asset_url, benchmarkConfiguration.baseURL).href
}

function commitIdentity(): string {
  try {
    return execFileSync('git', ['rev-parse', 'HEAD'], {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
    }).trim()
  } catch {
    return process.env.GIT_COMMIT ?? 'unknown'
  }
}

export function assertResourceLimits(
  report: BenchmarkReport,
  options: { steadyState?: boolean } = {},
): void {
  expect(report.metrics.visiblePointCount).toBeLessThanOrEqual(LIMITS.visiblePoints)
  if (!performanceAssertionsEnabled) return
  expect(report.metrics.cpuPoolBytes).toBeLessThanOrEqual(LIMITS.poolBytes)
  expect(report.metrics.gpuPoolBytes).toBeLessThanOrEqual(LIMITS.poolBytes)
  if (!options.steadyState) return
  expect(
    report.metrics.longTasks.filter(item => item.duration > LIMITS.longTaskMs).length,
  ).toBeLessThanOrEqual(1)
}
