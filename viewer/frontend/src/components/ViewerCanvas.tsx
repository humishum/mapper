import { useEffect, useRef } from 'react'
import type {
  Bounds,
  CameraState,
  CatalogArtifact,
  LayerState,
  PickResult,
  RendererController,
} from '../types/contracts'
import type { DetailSceneArtifact } from '../types/viewer'
import { OverviewRenderer } from '../renderers/OverviewRenderer'
import { DetailRenderer } from '../renderers/DetailRenderer'
import { loadTrajectoryRange } from '../core/trajectory'
import {
  benchmarkEvent,
  benchmarkRendererCreated,
  benchmarkRendererDisposed,
} from '../benchmark'

interface Props {
  mode: 'overview' | 'detail'
  overviewCamera: CameraState
  detailCamera: CameraState
  catalog: CatalogArtifact[]
  detailArtifacts: DetailSceneArtifact[]
  layers: LayerState[]
  onArtifactClick: (artifactId: string) => void
  onBoundsChange: (bounds: Bounds) => void
  onController: (controller: RendererController | null) => void
  onPick: (pick: PickResult) => void
  onError: (message: string) => void
  onLayerReady: (artifactId: string) => void
}

export function ViewerCanvas(props: Props) {
  const host = useRef<HTMLDivElement>(null)
  const controllerRef = useRef<RendererController | null>(null)
  const detailKey = props.detailArtifacts.map(item => item.detail.artifact_id).join(',')

  useEffect(() => {
    if (!host.current) return
    const generation = benchmarkRendererCreated(props.mode)
    const controller = props.mode === 'overview'
      ? new OverviewRenderer(
          host.current,
          props.overviewCamera,
          [],
          props.onArtifactClick,
          props.onBoundsChange,
          () => benchmarkEvent('overview-ready', { generation }),
        )
      : new DetailRenderer(host.current, props.detailCamera)
    controllerRef.current = controller
    props.onController(controller)
    const abort = new AbortController()
    let firstGeometryFrame: number | undefined

    if (controller instanceof DetailRenderer) {
      if (import.meta.env.VITE_BENCHMARK_ENABLED === 'true') {
        const reportFirstGeometry = () => {
          const visiblePoints = controller.getPerformance().visiblePoints
          if (visiblePoints > 0) {
            benchmarkEvent('first-geometry-visible', {
              generation,
              activeArtifactIds: props.detailArtifacts.map(
                artifact => artifact.detail.artifact_id,
              ),
              visiblePoints,
            })
            firstGeometryFrame = undefined
            return
          }
          firstGeometryFrame = requestAnimationFrame(reportFirstGeometry)
        }
        firstGeometryFrame = requestAnimationFrame(reportFirstGeometry)
      }
      void Promise.all(props.detailArtifacts.map(async artifact => {
        await controller.addArtifact(artifact, abort.signal)
        benchmarkEvent('layer-ready', { artifactId: artifact.detail.artifact_id, generation })
        props.onLayerReady(artifact.detail.artifact_id)
        const poses = artifact.detail.representations.find(item => item.kind === 'poses' && item.format === 'parquet')
        if (poses) {
          const points = await loadTrajectoryRange(
            poses.asset_url,
            poses.byte_size,
            artifact.renderMatrix,
            abort.signal,
          )
          controller.addTrajectory(points)
          benchmarkEvent('trajectory-ready', {
            artifactId: artifact.detail.artifact_id,
            generation,
            pointCount: points.length,
          })
        }
      })).catch(error => {
        if (!abort.signal.aborted) props.onError(error instanceof Error ? error.message : 'Detail renderer failed')
      })
    }

    const handleDoubleClick = (event: MouseEvent) => {
      if (!(controller instanceof DetailRenderer)) return
      void controller.pick(event as PointerEvent).then(pick => {
        if (pick && !abort.signal.aborted) props.onPick(pick)
      }).catch(error => {
        if (!abort.signal.aborted) props.onError(error instanceof Error ? error.message : 'Picking failed')
      })
    }
    host.current.addEventListener('dblclick', handleDoubleClick)
    const element = host.current
    return () => {
      abort.abort()
      if (firstGeometryFrame !== undefined) cancelAnimationFrame(firstGeometryFrame)
      element.removeEventListener('dblclick', handleDoubleClick)
      props.onController(null)
      controllerRef.current = null
      controller.dispose()
      benchmarkRendererDisposed(props.mode, generation)
      element.replaceChildren()
    }
    // Renderer lifecycle is keyed only by mode and selected artifacts.
    // Layer display changes are applied below without destroying GPU resources.
    // Callback props intentionally do not own the renderer lifecycle. Current
    // callbacks are read at construction; including them would tear down GPU
    // state on every React render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.mode, detailKey])

  useEffect(() => {
    const controller = controllerRef.current
    if (controller instanceof OverviewRenderer) controller.setCatalog(props.catalog)
  }, [props.catalog])

  useEffect(() => {
    controllerRef.current?.setLayers(props.layers)
  }, [props.layers])

  return (
    <div
      ref={host}
      className={`viewer-canvas viewer-canvas-${props.mode}`}
      data-testid="viewer-canvas"
      data-mode={props.mode}
    />
  )
}
