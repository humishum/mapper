import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import {
  fetchArtifactDetail,
  fetchManifest,
  fetchSources,
  validateSelectedSidecars,
} from './api/client'
import { benchmarkEvent, updateBenchmarkSnapshot } from './benchmark'
import { CatalogPanel } from './components/CatalogPanel'
import { InspectionPanel } from './components/InspectionPanel'
import { LayerControls } from './components/LayerControls'
import { ViewerCanvas } from './components/ViewerCanvas'
import {
  DEFAULT_POINT_BUDGET,
  GEOMETRY_POOL_BYTES,
  MAX_DETAIL_DISTANCE_METRES,
} from './core/budget'
import {
  distanceBetweenEcef,
  matrixTranslation,
  relativeArtifactMatrix,
} from './core/geo'
import { availableColorModes } from './core/schema'
import { parseUrlState, serializeUrlState } from './core/urlState'
import { useCatalog } from './hooks/useCatalog'
import { basemapConfig } from './renderers/config'
import type {
  Bounds,
  Inspection,
  LayerState,
  Manifest,
  PerformanceState,
  PickResult,
  RendererController,
  ViewerUrlState,
} from './types/contracts'
import type { DetailSceneArtifact } from './types/viewer'

const INITIAL_BOUNDS: Bounds = { west: -180, south: -90, east: 180, north: 90 }
const IDENTITY = [
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
] as const

const EMPTY_PERFORMANCE: PerformanceState = {
  visiblePoints: 0,
  pointBudget: DEFAULT_POINT_BUDGET,
  cpuGeometryBytes: 0,
  gpuGeometryBytes: 0,
  cpuGeometryLimitBytes: GEOMETRY_POOL_BYTES,
  gpuGeometryLimitBytes: GEOMETRY_POOL_BYTES,
}

function sameCamera(a: ViewerUrlState['overviewCamera'], b: ViewerUrlState['overviewCamera']): boolean {
  return [...a.position, ...a.target].every((value, index) => {
    const other = [...b.position, ...b.target][index]
    return Math.abs(value - other) < 1e-6
  })
}

function createLayer(manifest: Manifest, stored?: ViewerUrlState['layers'][string]): LayerState {
  const modes = availableColorModes(manifest)
  const requestedMode = stored?.colorMode
  return {
    artifactId: manifest.artifact_id,
    visible: stored?.visible ?? true,
    opacity: stored?.opacity ?? manifest.layer_default?.opacity ?? 1,
    pointSize: stored?.pointSize ?? manifest.layer_default?.point_size ?? 2,
    colorMode: requestedMode && modes.includes(requestedMode) ? requestedMode : (modes[0] ?? 'elevation'),
    loading: true,
    progress: 0,
    availableColorModes: modes,
  }
}

export default function App() {
  const initialUrl = useRef(parseUrlState(window.location.search))
  const [urlState, setUrlState] = useState<ViewerUrlState>({
    ...initialUrl.current,
    mode: 'overview',
    activeArtifactIds: [],
  })
  const [bounds, setBounds] = useState(INITIAL_BOUNDS)
  const catalog = useCatalog(bounds)
  const [detailArtifacts, setDetailArtifacts] = useState<DetailSceneArtifact[]>([])
  const [layers, setLayers] = useState<LayerState[]>([])
  const [inspection, setInspection] = useState<Inspection | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [transitioning, setTransitioning] = useState(false)
  const [performance, setPerformance] = useState(EMPTY_PERFORMANCE)
  const controller = useRef<RendererController | null>(null)
  const openOperation = useRef<AbortController | null>(null)
  const inspectionOperation = useRef<AbortController | null>(null)
  const detailArtifactsRef = useRef<DetailSceneArtifact[]>([])
  const restored = useRef(false)

  const manifests = useMemo(
    () => new Map(detailArtifacts.map(item => [item.manifest.artifact_id, item.manifest])),
    [detailArtifacts],
  )

  useEffect(() => {
    window.history.replaceState(null, '', serializeUrlState(urlState))
  }, [urlState])

  useEffect(() => {
    updateBenchmarkSnapshot({
      mode: urlState.mode,
      activeArtifactIds: [...urlState.activeArtifactIds],
      transitioning,
      performance,
    })
  }, [performance, transitioning, urlState.activeArtifactIds, urlState.mode])

  useEffect(() => () => {
    openOperation.current?.abort()
    inspectionOperation.current?.abort()
  }, [])

  useEffect(() => {
    const timer = window.setInterval(() => {
      const active = controller.current
      if (!active) return
      setPerformance(active.getPerformance())
      const camera = active.getCamera()
      setUrlState(current => {
        const stored = active.mode === 'overview'
          ? current.overviewCamera
          : current.detailCamera
        if (sameCamera(camera, stored)) return current
        return active.mode === 'overview'
          ? { ...current, overviewCamera: camera }
          : { ...current, detailCamera: camera }
      })
    }, 100)
    return () => window.clearInterval(timer)
  }, [])

  const enterOverview = useCallback(() => {
    benchmarkEvent('enter-overview')
    if (openOperation.current) benchmarkEvent('open-operation-abort', { reason: 'overview' })
    openOperation.current?.abort()
    inspectionOperation.current?.abort()
    const overviewCamera = controller.current?.mode === 'overview'
      ? controller.current.getCamera()
      : urlState.overviewCamera
    setTransitioning(true)
    setInspection(null)
    setUrlState(current => ({ ...current, mode: 'overview', activeArtifactIds: [], overviewCamera }))
    setLayers([])
    detailArtifactsRef.current = []
    setDetailArtifacts([])
    window.setTimeout(() => setTransitioning(false), 280)
  }, [urlState.overviewCamera])

  const openArtifact = useCallback(async (artifactId: string, compare = false) => {
    benchmarkEvent('open-artifact-start', { artifactId, compare })
    if (openOperation.current) benchmarkEvent('open-operation-abort', { reason: 'superseded' })
    openOperation.current?.abort()
    inspectionOperation.current?.abort()
    const abort = new AbortController()
    openOperation.current = abort
    setError(null)
    setTransitioning(true)
    try {
      const [detail, manifest] = await Promise.all([
        fetchArtifactDetail(artifactId, abort.signal),
        fetchManifest(artifactId, abort.signal),
      ])
      await validateSelectedSidecars(detail, abort.signal)
      const sources = availableColorModes(manifest).includes('source')
        ? await fetchSources(artifactId, abort.signal)
        : []
      if (abort.signal.aborted) {
        benchmarkEvent('stale-open-blocked', { artifactId })
        return
      }
      const transform = manifest.coordinate_frame.transform_to_ecef
      const aligned = manifest.alignment.status !== 'unaligned'
      if (aligned && (!transform || transform.length !== 16)) {
        throw new Error('Aligned artifact has no valid float64 ECEF transform')
      }

      let nextArtifacts = compare ? [...detailArtifactsRef.current] : []
      if (compare && nextArtifacts.length) {
        const existing = nextArtifacts[0]
        const existingTransform = existing.manifest.coordinate_frame.transform_to_ecef
        if (!aligned || !existingTransform) {
          throw new Error('Unaligned scenes cannot be globally compared; return to overview first')
        }
        const distance = distanceBetweenEcef(
          matrixTranslation(existingTransform),
          matrixTranslation(transform!),
        )
        if (distance > MAX_DETAIL_DISTANCE_METRES) {
          enterOverview()
          throw new Error(`Artifact is ${(distance / 1000).toFixed(1)} km from the active origin; transitioned to overview`)
        }
      }

      const origin = nextArtifacts.length
        ? matrixTranslation(nextArtifacts[0].manifest.coordinate_frame.transform_to_ecef!)
        : aligned ? matrixTranslation(transform!) : ([0, 0, 0] as const)
      const renderMatrix = relativeArtifactMatrix(origin, transform ?? IDENTITY)
      nextArtifacts = [...nextArtifacts.filter(item => item.detail.artifact_id !== artifactId), {
        detail,
        manifest,
        renderMatrix,
        sources,
      }].slice(0, 2)

      const priorCamera = controller.current?.mode === 'overview'
        ? controller.current.getCamera()
        : urlState.overviewCamera
      detailArtifactsRef.current = nextArtifacts
      setDetailArtifacts(nextArtifacts)
      setLayers(current => {
        const previous = new Map(current.map(layer => [layer.artifactId, layer]))
        return nextArtifacts.map(item => previous.get(item.detail.artifact_id)
          ?? createLayer(item.manifest, initialUrl.current.layers[item.detail.artifact_id]))
      })
      setUrlState(current => ({
        ...current,
        mode: 'detail',
        activeArtifactIds: nextArtifacts.map(item => item.detail.artifact_id),
        overviewCamera: priorCamera,
      }))
      benchmarkEvent('open-artifact-committed', { artifactId, compare })
    } catch (caught) {
      if (!abort.signal.aborted) {
        setError(caught instanceof Error ? caught.message : 'Could not open artifact')
      }
    } finally {
      if (!abort.signal.aborted) window.setTimeout(() => setTransitioning(false), 280)
    }
  }, [enterOverview, urlState.overviewCamera])

  useEffect(() => {
    if (restored.current || catalog.loading) return
    restored.current = true
    const ids = initialUrl.current.activeArtifactIds
    if (initialUrl.current.mode === 'detail' && ids[0]) {
      void openArtifact(ids[0]).then(() => {
        if (ids[1]) void openArtifact(ids[1], true)
      })
    }
  }, [catalog.loading, openArtifact])

  const updateLayer = useCallback((artifactId: string, patch: Partial<LayerState>) => {
    setLayers(current => current.map(layer => layer.artifactId === artifactId ? { ...layer, ...patch } : layer))
  }, [])

  useEffect(() => {
    if (!layers.length) return
    setUrlState(current => ({
      ...current,
      layers: Object.fromEntries(layers.map(layer => [layer.artifactId, {
        visible: layer.visible,
        opacity: layer.opacity,
        pointSize: layer.pointSize,
        colorMode: layer.colorMode,
      }])),
    }))
  }, [layers])

  const handlePick = useCallback(async (pick: PickResult) => {
    inspectionOperation.current?.abort()
    const abort = new AbortController()
    inspectionOperation.current = abort
    const artifact = detailArtifactsRef.current.find(
      item => item.detail.artifact_id === pick.artifactId,
    )
    if (!artifact) return
    try {
      const sources = pick.pointSourceId === undefined
        ? []
        : artifact.sources.length
          ? artifact.sources
          : await fetchSources(pick.artifactId, abort.signal)
      if (abort.signal.aborted) return
      setInspection({
        ...pick,
        source: sources.find(source => source.source_index === pick.pointSourceId),
        alignmentStatus: artifact.manifest.alignment.status,
        horizontalRmseM: artifact.manifest.alignment.horizontal_rmse_m,
        verticalRmseM: artifact.manifest.alignment.vertical_rmse_m,
      })
      setUrlState(current => ({ ...current, selectedSource: pick.pointSourceId ?? null }))
    } catch (caught) {
      if (!abort.signal.aborted) {
        setError(caught instanceof Error ? caught.message : 'Could not resolve point source')
      }
    }
  }, [])

  const handleController = useCallback((next: RendererController | null) => {
    controller.current = next
    if (!next) {
      setPerformance(EMPTY_PERFORMANCE)
      return
    }
    setPerformance(next.getPerformance())
  }, [])

  return (
    <main
      className={`app mode-${urlState.mode} ${transitioning ? 'transitioning' : ''}`}
      data-testid="viewer-app"
      data-mode={urlState.mode}
    >
      <ViewerCanvas
        mode={urlState.mode}
        overviewCamera={urlState.overviewCamera}
        detailCamera={urlState.detailCamera}
        catalog={catalog.geographic}
        detailArtifacts={detailArtifacts}
        layers={layers}
        onArtifactClick={artifactId => void openArtifact(artifactId)}
        onBoundsChange={setBounds}
        onController={handleController}
        onPick={pick => void handlePick(pick)}
        onError={setError}
        onLayerReady={artifactId => updateLayer(artifactId, { loading: false, progress: 1 })}
      />

      {urlState.mode === 'overview' ? (
        <>
          <CatalogPanel
            geographic={catalog.geographic}
            unaligned={catalog.unaligned}
            activeIds={urlState.activeArtifactIds}
            mode={urlState.mode}
            loading={catalog.loading}
            onOpen={(id, compare) => void openArtifact(id, compare)}
          />
          {basemapConfig.enabled && <div className="attribution">{basemapConfig.attribution}</div>}
        </>
      ) : (
        <>
          <button
            className="overview-button"
            data-testid="enter-overview"
            onClick={enterOverview}
          >
            ← Globe overview
          </button>
          <CatalogPanel
            geographic={catalog.geographic}
            unaligned={catalog.unaligned}
            activeIds={urlState.activeArtifactIds}
            mode={urlState.mode}
            loading={catalog.loading}
            onOpen={(id, compare) => void openArtifact(id, compare)}
          />
          <LayerControls
            layers={layers}
            manifests={manifests}
            performance={performance}
            onChange={updateLayer}
          />
        </>
      )}
      {inspection && <InspectionPanel inspection={inspection} onClose={() => setInspection(null)} />}
      <div className="interaction-hint">
        {urlState.mode === 'overview'
          ? 'Drag to orbit · select a footprint or catalog scene'
          : 'Orbit / pan / zoom · double-click a point to inspect'}
      </div>
      {(catalog.error || error) && (
        <div className="error-toast" role="alert">
          <span>{error ?? catalog.error}</span>
          <button onClick={() => setError(null)} aria-label="Dismiss error">×</button>
        </div>
      )}
      <div className="crossfade" aria-hidden="true" />
    </main>
  )
}
