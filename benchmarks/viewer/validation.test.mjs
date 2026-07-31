import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'

const root = new URL('./', import.meta.url)

test('report schema is a closed v1 contract with all evidence groups', async () => {
  const schema = JSON.parse(
    await readFile(new URL('report.schema.json', root), 'utf8'),
  )
  assert.equal(schema.properties.schemaVersion.const, 1)
  assert.equal(schema.additionalProperties, false)
  for (const field of [
    'identity',
    'configuration',
    'performanceMarks',
    'network',
    'metrics',
    'behavior',
  ]) {
    assert.ok(schema.required.includes(field), `missing required ${field}`)
  }
  assert.equal(schema.properties.network.items.properties.range.type[0], 'string')
  assert.deepEqual(
    schema.properties.configuration.properties.evidenceMode.enum,
    ['acceptance', 'functional'],
  )
  assert.ok(schema.properties.configuration.required.includes('evidenceMode'))
  assert.ok(schema.properties.identity.required.includes('webglRenderer'))
  assert.ok(schema.properties.metrics.required.includes('frameTimeP95Ms'))
  assert.ok(schema.properties.metrics.required.includes('cpuPoolBytes'))
  assert.ok(schema.properties.metrics.required.includes('gpuPoolBytes'))
})

test('harness discovers every required Phase 2 scenario and gate', async () => {
  const spec = await readFile(
    new URL('viewer.benchmark.spec.ts', root),
    'utf8',
  )
  for (const scenario of [
    'cold-first-geometry',
    'warm-first-geometry',
    'orbit',
    'dive',
    'comparison',
    'site-transition',
    'throttled-cancellation',
  ]) {
    assert.match(spec, new RegExp(`scenario: '${scenario}'`))
  }
  for (const assertion of [
    'inactiveSiteRequestCount',
    'navigationCount',
    'disposedRendererGenerations',
    'abortedRequestCount',
    'staleSuccessfulRequests',
  ]) {
    assert.match(spec, new RegExp(assertion))
  }
  assert.doesNotMatch(spec, /status: 'skipped'/)
  const harness = await readFile(new URL('harness.ts', root), 'utf8')
  assert.match(harness, /artifact_phase2_site_a_comparison/)
  for (const limit of ['2_000_000', '256 * 1024 * 1024', 'longTaskMs: 50']) {
    assert.ok(harness.includes(limit), `missing gate ${limit}`)
  }
  for (const softwareRenderer of ['swiftshader', 'software', 'llvmpipe', 'unavailable']) {
    assert.match(harness.toLowerCase(), new RegExp(softwareRenderer))
  }
  assert.match(harness, /acceptance evidence requires hardware WebGL/)
})

test('mp7 focused acceptance records and enforces the complete evidence set', async () => {
  const spec = await readFile(new URL('mp7.functional.spec.ts', root), 'utf8')
  for (const evidence of [
    'coldFirstGeometryMs',
    'warmFirstGeometryMs',
    'orbitP95Ms',
    'diveP95Ms',
    'orbitLongTasksOver50Ms',
    'diveLongTasksOver50Ms',
    'cpuGeometryBytes',
    'gpuGeometryBytes',
    'stableAfterOrbitAndDive',
    'modeSelectedAfterReopen',
    'pickedPointSourceId',
    'pickedPointCaptureId',
    'MAPPER_MP7_SCREENSHOT',
    'isHardwareWebglRenderer',
  ]) {
    assert.ok(spec.includes(evidence), `missing mp7 evidence ${evidence}`)
  }
  assert.match(spec, /if \(performanceAssertionsEnabled\)/)
  assert.match(spec, /toBeLessThanOrEqual\(20\)/)
  assert.match(spec, /toBeLessThanOrEqual\(1\)/)
})

test('managed acceptance uses an isolated production preview', async () => {
  const config = await readFile(new URL('playwright.config.ts', root), 'utf8')
  assert.match(config, /command: 'npm run benchmark:serve'/)
  assert.match(config, /reuseExistingServer: false/)
  assert.match(
    config,
    /VITE_BASEMAP_ENABLED: process\.env\.VITE_BASEMAP_ENABLED \?\? 'false'/,
  )

  const viteConfig = await readFile(
    new URL('../../viewer/frontend/vite.config.ts', root),
    'utf8',
  )
  assert.match(viteConfig, /preview: \{ proxy \}/)

  const canvas = await readFile(
    new URL('../../viewer/frontend/src/components/ViewerCanvas.tsx', root),
    'utf8',
  )
  assert.doesNotMatch(canvas, /\[props\.mode, detailKey, props\.catalog\]/)
  assert.match(canvas, /controller\.setCatalog\(props\.catalog\)/)
})
