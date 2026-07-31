import { expect, test } from '@playwright/test'
import { mkdir, writeFile } from 'node:fs/promises'
import path from 'node:path'
import {
  benchmarkConfiguration,
  installBrowserProbe,
  isHardwareWebglRenderer,
  openArtifact,
  readWebglRenderer,
  recordNetwork,
  waitForOverviewRendererReady,
} from './harness'

const POINT_BUDGET = 2_000_000
const smallArtifactId = process.env.MAPPER_UNALIGNED_SMALL_ARTIFACT
  ?? 'art_ec9f614413b54073b2e12ef0ffda17ec'
const largeArtifactId = process.env.MAPPER_UNALIGNED_LARGE_ARTIFACT
  ?? 'art_78a0f4e44d7a464db78dd95b6d9fe86c'

test.beforeEach(async ({ page }) => {
  await installBrowserProbe(page)
})

test('real fresh unaligned scenes remain local across detail transitions', async ({ page, browser }) => {
  const network = recordNetwork(page)
  await page.goto('/')
  await waitForFreshOverview(page)
  await page.waitForTimeout(750)

  const localScenes = page.getByTestId('local-scenes')
  const geographicScenes = page.getByTestId('geographic-scenes')
  for (const [artifactId, pointLabel] of [
    [smallArtifactId, '200,228 points'],
    [largeArtifactId, '11,207,020 points'],
  ] as const) {
    const item = page.getByTestId(`catalog-artifact-${artifactId}`)
    await expect(localScenes.getByTestId(`catalog-artifact-${artifactId}`)).toBeVisible()
    await expect(item).toContainText('Local only')
    await expect(item).toContainText(pointLabel)
    await expect(geographicScenes.getByTestId(`catalog-artifact-${artifactId}`)).toHaveCount(0)

    const manifest = await page.request.get(
      `/api/v1/catalog/artifacts/${encodeURIComponent(artifactId)}/manifest`,
    )
    expect(manifest.ok()).toBeTruthy()
    const body = await manifest.json() as {
      alignment: { status: string; rejection_reason: string | null }
      coordinate_frame: {
        origin_wgs84: unknown
        transform_to_ecef: unknown
      }
      footprint_wgs84: unknown
    }
    expect(body.alignment.status).toBe('unaligned')
    expect(body.alignment.rejection_reason).toBe('gps_unavailable')
    expect(body.coordinate_frame.origin_wgs84).toBeNull()
    expect(body.coordinate_frame.transform_to_ecef).toBeNull()
    expect(body.footprint_wgs84).toBeNull()
  }
  expect(network.records.filter(item => item.url.includes('/api/v1/assets/'))).toHaveLength(0)

  await openArtifact(page, smallArtifactId)
  await expect(page.getByTestId(`alignment-warning-${smallArtifactId}`))
    .toContainText('Local coordinates only: gps_unavailable')
  const smallSnapshot = await assertLocalDetailState(page, smallArtifactId)

  await page.getByTestId('enter-overview').click()
  await waitForFreshOverview(page)
  await expect(localScenes.getByTestId(`catalog-artifact-${largeArtifactId}`)).toBeVisible()
  await openArtifact(page, largeArtifactId)
  await expect(page.getByTestId(`alignment-warning-${largeArtifactId}`))
    .toContainText('Local coordinates only: gps_unavailable')
  const largeSnapshot = await assertLocalDetailState(page, largeArtifactId)

  expect(network.navigationCount()).toBe(1)
  const webglRenderer = await readWebglRenderer(page)
  if (benchmarkConfiguration.evidenceMode === 'acceptance') {
    expect(
      isHardwareWebglRenderer(webglRenderer),
      `acceptance evidence requires hardware WebGL; reported renderer: ${webglRenderer}`,
    ).toBe(true)
  }
  const reportPath = process.env.MAPPER_UNALIGNED_REPORT
  if (reportPath) {
    await mkdir(path.dirname(reportPath), { recursive: true })
    await writeFile(reportPath, `${JSON.stringify({
      evidenceMode: benchmarkConfiguration.evidenceMode,
      browser: browser.version(),
      webglRenderer,
      catalog: '/tmp/mapper-phase2-fresh-catalog.sqlite3',
      overviewAssetRequestCount: 0,
      navigationCount: network.navigationCount(),
      artifacts: [
        {
          artifactId: smallArtifactId,
          declaredPointCount: 200_228,
          alignmentStatus: 'unaligned',
          rejectionReason: 'gps_unavailable',
          originWgs84: null,
          transformToEcef: null,
          visiblePoints: smallSnapshot.performance.visiblePoints,
          pointBudget: smallSnapshot.performance.pointBudget,
        },
        {
          artifactId: largeArtifactId,
          declaredPointCount: 11_207_020,
          alignmentStatus: 'unaligned',
          rejectionReason: 'gps_unavailable',
          originWgs84: null,
          transformToEcef: null,
          visiblePoints: largeSnapshot.performance.visiblePoints,
          pointBudget: largeSnapshot.performance.pointBudget,
        },
      ],
    }, null, 2)}\n`, 'utf8')
  }
})

async function assertLocalDetailState(page: import('@playwright/test').Page, artifactId: string) {
  const snapshot = await page.evaluate(() => (
    window as unknown as {
      __MAPPER_BENCHMARK__?: {
        snapshot: {
          mode: string
          activeArtifactIds: string[]
          performance: { visiblePoints: number; pointBudget: number }
        }
      }
    }
  ).__MAPPER_BENCHMARK__?.snapshot)
  expect(snapshot).toBeTruthy()
  expect(snapshot?.mode).toBe('detail')
  expect(snapshot?.activeArtifactIds).toEqual([artifactId])
  expect(snapshot?.performance.visiblePoints).toBeGreaterThan(0)
  expect(snapshot?.performance.visiblePoints).toBeLessThanOrEqual(POINT_BUDGET)
  expect(snapshot?.performance.pointBudget).toBe(POINT_BUDGET)
  return snapshot!
}

async function waitForFreshOverview(page: import('@playwright/test').Page) {
  await expect(page.getByTestId('viewer-app')).toHaveAttribute('data-mode', 'overview')
  await expect(page.getByTestId(`open-artifact-${smallArtifactId}`)).toBeVisible()
  await expect(page.getByTestId(`open-artifact-${largeArtifactId}`)).toBeVisible()
  await waitForOverviewRendererReady(page)
}
