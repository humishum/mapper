import { expect, test } from '@playwright/test'
import { mkdir, writeFile } from 'node:fs/promises'
import path from 'node:path'
import {
  LIMITS,
  benchmarkConfiguration,
  createCdpSession,
  installBrowserProbe,
  isHardwareWebglRenderer,
  openArtifact,
  p95,
  performanceAssertionsEnabled,
  pickVisiblePoint,
  pointAssetUrl,
  readBrowserProbe,
  readWebglRenderer,
  recordNetwork,
  resetBrowserProbe,
  setNetworkConditions,
  waitForTrajectory,
  waitForOverviewRendererReady,
} from './harness'

const artifactId = process.env.MAPPER_MP7_ARTIFACT
  ?? 'art_4e05de949d7540108a56497d53017eb1'
const POINT_BUDGET = 2_000_000

test.beforeEach(async ({ page }) => {
  await installBrowserProbe(page)
})

test('canonical mp7 streams cold and warm with bounded local detail', async ({ page, browser }) => {
  const cdp = await createCdpSession(page)
  await setNetworkConditions(cdp, { cacheDisabled: true })
  const network = recordNetwork(page)
  await page.goto('/')
  await waitForMp7Overview(page)
  await page.waitForTimeout(750)
  expect(assetRequests(network.records)).toHaveLength(0)

  const assetUrl = await pointAssetUrl(page, artifactId)
  const coldMs = await openArtifact(page, artifactId)
  await waitForTrajectory(page, artifactId)
  const coldSnapshot = await detailSnapshot(page)
  assertBudget(coldSnapshot)
  const coldRanges = rangedRequests(network.records, assetUrl)
  expect(coldRanges.length).toBeGreaterThan(0)
  expect(coldRanges.some(item => item.status === 206)).toBe(true)

  const sourceColor = page.getByTestId(`color-mode-${artifactId}`)
  await expect(sourceColor).toBeEnabled()
  await sourceColor.selectOption('source')
  await expect(sourceColor).toHaveValue('source')

  const disposedBefore = coldSnapshot.disposedRendererGenerations
  const detailGeneration = coldSnapshot.rendererGeneration
  await page.getByTestId('enter-overview').click()
  await waitForMp7Overview(page)
  await page.waitForFunction(generation => {
    const bridge = (window as unknown as {
      __MAPPER_BENCHMARK__?: {
        snapshot: { disposedRendererGenerations: number[] }
      }
    }).__MAPPER_BENCHMARK__
    return bridge?.snapshot.disposedRendererGenerations.includes(generation)
  }, detailGeneration)
  const overviewSnapshot = await bridgeSnapshot(page)
  expect(overviewSnapshot.activeArtifactIds).toEqual([])
  expect(overviewSnapshot.disposedRendererGenerations.length)
    .toBeGreaterThan(disposedBefore.length)
  expect(overviewSnapshot.disposedRendererGenerations).toContain(detailGeneration)

  const trajectoryEventsBeforeWarm = await trajectoryEventCount(page)
  network.records.splice(0)
  await setNetworkConditions(cdp, { cacheDisabled: false })
  const warmMs = await openArtifact(page, artifactId)
  await page.waitForFunction(({ id, prior }) => {
    const events = (window as unknown as {
      __MAPPER_BENCHMARK__?: {
        events: Array<{ name: string; detail?: Record<string, unknown> }>
      }
    }).__MAPPER_BENCHMARK__?.events ?? []
    return events.filter(event => event.name === 'trajectory-ready'
      && event.detail?.artifactId === id).length > prior
  }, { id: artifactId, prior: trajectoryEventsBeforeWarm })
  const warmSnapshot = await detailSnapshot(page)
  assertBudget(warmSnapshot)
  const warmRanges = rangedRequests(network.records, assetUrl)
  expect(warmRanges.length).toBeGreaterThan(0)
  expect(warmRanges.some(item => item.status === 206)).toBe(true)

  const warmSourceColor = page.getByTestId(`color-mode-${artifactId}`)
  await warmSourceColor.selectOption('source')
  await expect(warmSourceColor).toHaveValue('source')

  const canvas = page.getByTestId('viewer-canvas')
  const box = await canvas.boundingBox()
  if (!box) throw new Error('viewer canvas has no layout box')

  await resetBrowserProbe(page)
  await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.5)
  await page.mouse.down()
  await page.mouse.move(box.x + box.width * 0.72, box.y + box.height * 0.42, {
    steps: 24,
  })
  await page.mouse.up()
  await page.waitForTimeout(1_000)
  const orbit = await readBrowserProbe(page)

  await resetBrowserProbe(page)
  await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.5)
  for (let index = 0; index < 16; index += 1) {
    await page.mouse.wheel(0, -180)
  }
  await page.waitForTimeout(1_000)
  const dive = await readBrowserProbe(page)
  const steadySnapshot = await detailSnapshot(page)
  assertBudget(steadySnapshot)
  await expect(warmSourceColor).toHaveValue('source')

  await pickVisiblePoint(page)
  const inspectionSource = (await page.getByTestId('inspection-source').innerText()).trim()
  const inspectionCapture = (await page.getByTestId('inspection-capture').innerText()).trim()
  expect(inspectionSource).not.toBe('not present')
  expect(inspectionCapture).not.toBe('—')
  const trajectoryEventsAfterInteractions = await trajectoryEventCount(page)
  const trajectoryStable = trajectoryEventsAfterInteractions > trajectoryEventsBeforeWarm
    && steadySnapshot.rendererGeneration === warmSnapshot.rendererGeneration
    && !steadySnapshot.disposedRendererGenerations.includes(warmSnapshot.rendererGeneration)
  expect(trajectoryStable).toBe(true)

  const screenshotPath = process.env.MAPPER_MP7_SCREENSHOT
    ?? '/tmp/mapper-phase2-mp7-source-colored.png'
  await mkdir(path.dirname(screenshotPath), { recursive: true })
  await page.screenshot({ path: screenshotPath, fullPage: true })

  const webglRenderer = await readWebglRenderer(page)
  const orbitP95Ms = p95(orbit.frameTimesMs)
  const diveP95Ms = p95(dive.frameTimesMs)
  const orbitLongTasksOver50Ms =
    orbit.longTasks.filter(item => item.duration > LIMITS.longTaskMs).length
  const diveLongTasksOver50Ms =
    dive.longTasks.filter(item => item.duration > LIMITS.longTaskMs).length

  const reportPath = process.env.MAPPER_MP7_REPORT
  if (reportPath) {
    await mkdir(path.dirname(reportPath), { recursive: true })
    await writeFile(reportPath, `${JSON.stringify({
      evidenceMode: benchmarkConfiguration.evidenceMode,
      browser: browser.version(),
      webglRenderer,
      catalog: '/tmp/mapper-phase2-mp7.NNlC2W/catalog.sqlite3',
      artifactId,
      declaredPointCount: 146_911_634,
      overviewAssetRequestCount: 0,
      cold: {
        firstGeometryMs: coldMs,
        rangedCopcRequests: coldRanges.length,
        visiblePoints: coldSnapshot.performance.visiblePoints,
        pointBudget: coldSnapshot.performance.pointBudget,
        cpuGeometryBytes: coldSnapshot.performance.cpuGeometryBytes,
        gpuGeometryBytes: coldSnapshot.performance.gpuGeometryBytes,
      },
      warm: {
        firstGeometryMs: warmMs,
        rangedCopcRequests: warmRanges.length,
        visiblePoints: warmSnapshot.performance.visiblePoints,
        pointBudget: warmSnapshot.performance.pointBudget,
        cpuGeometryBytes: warmSnapshot.performance.cpuGeometryBytes,
        gpuGeometryBytes: warmSnapshot.performance.gpuGeometryBytes,
      },
      steadyState: {
        orbit: {
          frameTimeP95Ms: orbitP95Ms,
          longTasksOver50Ms: orbitLongTasksOver50Ms,
          frameTimesMs: orbit.frameTimesMs,
          longTasks: orbit.longTasks,
        },
        dive: {
          frameTimeP95Ms: diveP95Ms,
          longTasksOver50Ms: diveLongTasksOver50Ms,
          frameTimesMs: dive.frameTimesMs,
          longTasks: dive.longTasks,
        },
        visiblePoints: steadySnapshot.performance.visiblePoints,
        cpuGeometryBytes: steadySnapshot.performance.cpuGeometryBytes,
        gpuGeometryBytes: steadySnapshot.performance.gpuGeometryBytes,
      },
      source: {
        modeSelectedAfterReopen: true,
        pickedPointSourceId: inspectionSource,
        pickedPointCaptureId: inspectionCapture,
      },
      trajectory: {
        readyCold: true,
        readyWarm: true,
        stableAfterOrbitAndDive: trajectoryStable,
      },
      screenshotPath,
      disposedDetailGeneration: detailGeneration,
      disposedRendererGenerations: overviewSnapshot.disposedRendererGenerations,
    }, null, 2)}\n`, 'utf8')
  }

  if (performanceAssertionsEnabled) {
    expect(
      isHardwareWebglRenderer(webglRenderer),
      `acceptance evidence requires hardware WebGL; reported renderer: ${webglRenderer}`,
    ).toBe(true)
    expect(coldMs).toBeLessThanOrEqual(LIMITS.coldFirstGeometryMs)
    expect(warmMs).toBeLessThanOrEqual(LIMITS.warmFirstGeometryMs)
    expect(orbitP95Ms).toBeLessThanOrEqual(20)
    expect(diveP95Ms).toBeLessThanOrEqual(20)
    expect(orbitLongTasksOver50Ms).toBeLessThanOrEqual(1)
    expect(diveLongTasksOver50Ms).toBeLessThanOrEqual(1)
    for (const snapshot of [coldSnapshot, warmSnapshot, steadySnapshot]) {
      expect(snapshot.performance.cpuGeometryBytes).toBeLessThanOrEqual(LIMITS.poolBytes)
      expect(snapshot.performance.gpuGeometryBytes).toBeLessThanOrEqual(LIMITS.poolBytes)
    }
  }
})

function assetRequests(records: Array<{ url: string }>) {
  return records.filter(item => item.url.includes('/api/v1/assets/'))
}

function rangedRequests(
  records: Array<{ url: string; range: string | null; status?: number }>,
  assetUrl: string,
) {
  return records.filter(item => item.url === assetUrl && item.range)
}

function assertBudget(snapshot: Awaited<ReturnType<typeof detailSnapshot>>) {
  expect(snapshot.activeArtifactIds).toEqual([artifactId])
  expect(snapshot.performance.visiblePoints).toBeGreaterThan(0)
  expect(snapshot.performance.visiblePoints).toBeLessThanOrEqual(POINT_BUDGET)
  expect(snapshot.performance.pointBudget).toBe(POINT_BUDGET)
}

async function waitForMp7Overview(page: import('@playwright/test').Page) {
  await expect(page.getByTestId('viewer-app')).toHaveAttribute('data-mode', 'overview')
  await expect(page.getByTestId(`open-artifact-${artifactId}`)).toBeVisible()
  await waitForOverviewRendererReady(page)
}

async function bridgeSnapshot(page: import('@playwright/test').Page) {
  const snapshot = await page.evaluate(() => (
    window as unknown as {
      __MAPPER_BENCHMARK__?: {
        snapshot: {
          activeArtifactIds: string[]
          rendererGeneration: number
          disposedRendererGenerations: number[]
          performance: {
            visiblePoints: number
            pointBudget: number
            cpuGeometryBytes: number
            gpuGeometryBytes: number
          }
        }
      }
    }
  ).__MAPPER_BENCHMARK__?.snapshot)
  if (!snapshot) throw new Error('benchmark bridge is unavailable')
  return snapshot
}

async function detailSnapshot(page: import('@playwright/test').Page) {
  const snapshot = await bridgeSnapshot(page)
  expect(await page.getByTestId('viewer-app').getAttribute('data-mode')).toBe('detail')
  return snapshot
}

async function trajectoryEventCount(page: import('@playwright/test').Page) {
  return page.evaluate(id => {
    const events = (window as unknown as {
      __MAPPER_BENCHMARK__?: {
        events: Array<{ name: string; detail?: Record<string, unknown> }>
      }
    }).__MAPPER_BENCHMARK__?.events ?? []
    return events.filter(event => event.name === 'trajectory-ready'
      && event.detail?.artifactId === id).length
  }, artifactId)
}
