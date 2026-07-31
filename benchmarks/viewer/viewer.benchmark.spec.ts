import { expect, test } from '@playwright/test'
import {
  LIMITS,
  assertResourceLimits,
  benchmarkConfiguration,
  createCdpSession,
  enterOverview,
  finalizeReport,
  installBrowserProbe,
  openArtifact,
  pickVisiblePoint,
  pointAssetUrl,
  performanceAssertionsEnabled,
  recordNetwork,
  resetBrowserProbe,
  setNetworkConditions,
  waitForOverview,
  waitForTrajectory,
} from './harness'

test.describe.configure({ mode: 'serial' })

test.beforeEach(async ({ page }) => {
  await installBrowserProbe(page)
})

test('cold first geometry and overview request isolation', async ({ page, browser }) => {
  const cdp = await createCdpSession(page)
  await setNetworkConditions(cdp, { cacheDisabled: true })
  const network = recordNetwork(page)
  await page.goto('/')
  await waitForOverview(page)
  await page.waitForTimeout(750)
  expect(network.records.filter(isDetailedAsset)).toHaveLength(0)
  const siteBAsset = await pointAssetUrl(page, benchmarkConfiguration.siteBArtifactId)

  await resetBrowserProbe(page)
  const firstGeometryMs = await openArtifact(page, benchmarkConfiguration.siteAArtifactId)
  const report = await finalizeReport({
    scenario: 'cold-first-geometry',
    page,
    browser,
    network: network.records,
    navigationCount: network.navigationCount(),
    timings: { firstGeometryMs },
    inactiveSiteAssetUrl: siteBAsset,
  })
  if (performanceAssertionsEnabled) {
    expect(firstGeometryMs).toBeLessThanOrEqual(LIMITS.coldFirstGeometryMs)
  }
  expect(report.network.some(item => isDetailedAsset(item) && item.range)).toBeTruthy()
  expect(report.behavior.inactiveSiteRequestCount).toBe(0)
  assertResourceLimits(report)
})

test('warm first geometry', async ({ page, browser }) => {
  const cdp = await createCdpSession(page)
  const network = recordNetwork(page)
  await page.goto('/')
  await waitForOverview(page)
  await openArtifact(page, benchmarkConfiguration.siteAArtifactId)
  await enterOverview(page)
  await setNetworkConditions(cdp, { cacheDisabled: false })
  network.records.splice(0)
  await resetBrowserProbe(page)

  const firstGeometryMs = await openArtifact(page, benchmarkConfiguration.siteAArtifactId)
  const report = await finalizeReport({
    scenario: 'warm-first-geometry',
    page,
    browser,
    network: network.records,
    navigationCount: network.navigationCount(),
    timings: { firstGeometryMs },
  })
  if (performanceAssertionsEnabled) {
    expect(firstGeometryMs).toBeLessThanOrEqual(LIMITS.warmFirstGeometryMs)
  }
  assertResourceLimits(report)
})

test('orbit steady-state', async ({ page, browser }) => {
  const network = recordNetwork(page)
  await page.goto('/')
  await waitForOverview(page)
  await openArtifact(page, benchmarkConfiguration.siteAArtifactId)
  await resetBrowserProbe(page)
  const canvas = page.getByTestId('viewer-canvas')
  const box = await canvas.boundingBox()
  if (!box) throw new Error('viewer canvas has no layout box')
  await page.mouse.move(box.x + box.width * 0.35, box.y + box.height * 0.5)
  await page.mouse.down()
  for (let step = 0; step < 24; step += 1) {
    await page.mouse.move(
      box.x + box.width * (0.35 + step / 80),
      box.y + box.height * (0.5 + Math.sin(step / 3) * 0.08),
    )
  }
  await page.mouse.up()
  await page.waitForTimeout(1_000)
  const report = await finalizeReport({
    scenario: 'orbit',
    page,
    browser,
    network: network.records,
    navigationCount: network.navigationCount(),
  })
  if (performanceAssertionsEnabled) {
    expect(report.metrics.frameTimeP95Ms).toBeLessThanOrEqual(20)
  }
  assertResourceLimits(report, { steadyState: true })
})

test('dive steady-state', async ({ page, browser }) => {
  const network = recordNetwork(page)
  await page.goto('/')
  await waitForOverview(page)
  await openArtifact(page, benchmarkConfiguration.siteAArtifactId)
  await resetBrowserProbe(page)
  const canvas = page.getByTestId('viewer-canvas')
  await canvas.hover()
  for (let step = 0; step < 16; step += 1) await page.mouse.wheel(0, -180)
  await page.waitForTimeout(1_000)
  const report = await finalizeReport({
    scenario: 'dive',
    page,
    browser,
    network: network.records,
    navigationCount: network.navigationCount(),
  })
  if (performanceAssertionsEnabled) {
    expect(report.metrics.frameTimeP95Ms).toBeLessThanOrEqual(20)
  }
  assertResourceLimits(report, { steadyState: true })
})

test('two-COPC comparison shares one global budget', async ({ page, browser }) => {
  const network = recordNetwork(page)
  await page.goto('/')
  await waitForOverview(page)
  await openArtifact(page, benchmarkConfiguration.siteAArtifactId)
  await page.getByTestId(
    `compare-artifact-${benchmarkConfiguration.comparisonArtifactId}`,
  ).click({ force: true })
  await page.waitForFunction(id => {
    const bridge = (window as unknown as {
      __MAPPER_BENCHMARK__?: { snapshot: { activeArtifactIds: string[] } }
    }).__MAPPER_BENCHMARK__
    return bridge?.snapshot.activeArtifactIds.includes(id)
      && bridge.snapshot.activeArtifactIds.length === 2
  }, benchmarkConfiguration.comparisonArtifactId)
  await page.waitForTimeout(1_000)
  const report = await finalizeReport({
    scenario: 'comparison',
    page,
    browser,
    network: network.records,
    navigationCount: network.navigationCount(),
  })
  expect(report.behavior.activeArtifactIds).toHaveLength(2)
  assertResourceLimits(report)
})

test('overview to site A to overview to site B without reload', async ({ page, browser }) => {
  const network = recordNetwork(page)
  await page.goto('/')
  await waitForOverview(page)
  const siteBAsset = await pointAssetUrl(page, benchmarkConfiguration.siteBArtifactId)
  await openArtifact(page, benchmarkConfiguration.siteAArtifactId)
  await waitForTrajectory(page, benchmarkConfiguration.siteAArtifactId)
  expect(network.records.filter(item => item.url === siteBAsset)).toHaveLength(0)
  await enterOverview(page)
  await openArtifact(page, benchmarkConfiguration.siteBArtifactId)
  await waitForTrajectory(page, benchmarkConfiguration.siteBArtifactId)
  const sourceColor = page.getByTestId(
    `color-mode-${benchmarkConfiguration.siteBArtifactId}`,
  )
  await expect(sourceColor).toBeEnabled()
  await sourceColor.selectOption('source')
  await expect(sourceColor).toHaveValue('source')
  await pickVisiblePoint(page)
  await expect(page.getByTestId('inspection-source')).not.toHaveText('not present')
  await expect(page.getByTestId('inspection-capture')).not.toHaveText('—')
  const report = await finalizeReport({
    scenario: 'site-transition',
    page,
    browser,
    network: network.records,
    navigationCount: network.navigationCount(),
  })
  expect(report.behavior.navigationCount).toBe(1)
  expect(report.behavior.activeArtifactIds).toEqual([
    benchmarkConfiguration.siteBArtifactId,
  ])
  expect(report.behavior.disposedRendererGenerations.length).toBeGreaterThanOrEqual(2)
  assertResourceLimits(report)
})

test('throttled supersession aborts and cannot repopulate disposed state', async ({
  page,
  browser,
}) => {
  const cdp = await createCdpSession(page)
  await setNetworkConditions(cdp, { cacheDisabled: true })
  const network = recordNetwork(page)
  await page.goto('/')
  await waitForOverview(page)
  const siteAAsset = await pointAssetUrl(page, benchmarkConfiguration.siteAArtifactId)
  // Throttle artifact traffic, not the application shell or basemap. Applying
  // 3G emulation before navigation makes the load event depend on unrelated
  // tile requests and does not exercise supersession more accurately.
  await setNetworkConditions(cdp, { throttled: true })
  await page.getByTestId(
    `open-artifact-${benchmarkConfiguration.siteAArtifactId}`,
  ).click()
  await page.waitForTimeout(50)
  const supersededAt = Date.now()
  await page.getByTestId(
    `open-artifact-${benchmarkConfiguration.siteBArtifactId}`,
  ).click()
  await expect(page.getByTestId('viewer-app')).toHaveAttribute('data-mode', 'detail')
  await page.waitForFunction(id => {
    const bridge = (window as unknown as {
      __MAPPER_BENCHMARK__?: {
        snapshot: { activeArtifactIds: string[] }
      }
    }).__MAPPER_BENCHMARK__
    return bridge?.snapshot.activeArtifactIds.length === 1
      && bridge.snapshot.activeArtifactIds[0] === id
  }, benchmarkConfiguration.siteBArtifactId, { timeout: 90_000 })
  // Keep a throttled overlap window long enough for the superseded COPC
  // request to abort, then restore bandwidth so the assertion measures stale
  // lifecycle behavior rather than whether this small fixture can fully stream
  // under an arbitrary 3G timeout.
  await page.waitForTimeout(1_000)
  await setNetworkConditions(cdp, { throttled: false })
  await page.waitForFunction(id => {
    const bridge = (window as unknown as {
      __MAPPER_BENCHMARK__?: {
        snapshot: { activeArtifactIds: string[]; performance: { visiblePoints: number } }
      }
    }).__MAPPER_BENCHMARK__
    return bridge?.snapshot.activeArtifactIds.length === 1
      && bridge.snapshot.activeArtifactIds[0] === id
      && bridge.snapshot.performance.visiblePoints > 0
  }, benchmarkConfiguration.siteBArtifactId, { timeout: 90_000 })
  await page.waitForTimeout(1_000)

  const staleSuccessfulRequests = network.records.filter(
    item => item.url === siteAAsset
      && (item.completedAt ?? 0) > supersededAt
      && item.status !== undefined,
  )
  const staleCommit = await page.evaluate(({ siteA, siteB }) => {
    const events = (window as unknown as {
      __MAPPER_BENCHMARK__?: {
        events: Array<{ name: string; detail?: Record<string, unknown> }>
      }
    }).__MAPPER_BENCHMARK__?.events ?? []
    const switchIndex = events.findIndex(
      item => item.name === 'open-artifact-start' && item.detail?.artifactId === siteB,
    )
    return events.slice(switchIndex + 1).some(
      item => item.name === 'open-artifact-committed' && item.detail?.artifactId === siteA,
    )
  }, {
    siteA: benchmarkConfiguration.siteAArtifactId,
    siteB: benchmarkConfiguration.siteBArtifactId,
  })
  const report = await finalizeReport({
    scenario: 'throttled-cancellation',
    page,
    browser,
    network: network.records,
    navigationCount: network.navigationCount(),
  })
  expect(staleSuccessfulRequests).toHaveLength(0)
  expect(staleCommit).toBe(false)
  expect(report.behavior.activeArtifactIds).toEqual([
    benchmarkConfiguration.siteBArtifactId,
  ])
  expect(report.behavior.abortedRequestCount).toBeGreaterThan(0)
  assertResourceLimits(report)
})

function isDetailedAsset(record: { url: string }): boolean {
  return record.url.includes('/api/v1/assets/')
}
