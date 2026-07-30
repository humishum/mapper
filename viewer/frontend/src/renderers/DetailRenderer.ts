import Instance from '@giro3d/giro3d/core/Instance'
import ColorMap from '@giro3d/giro3d/core/ColorMap'
import { CoordinateSystem } from '@giro3d/giro3d/core/geographic/CoordinateSystem'
import PointCloud from '@giro3d/giro3d/entities/PointCloud'
import COPCSource from '@giro3d/giro3d/sources/COPCSource'
import { traverseNode, type PointCloudNode } from '@giro3d/giro3d/sources/PointCloudSource'
import {
  BufferGeometry,
  Color,
  Line,
  LineBasicMaterial,
  PerspectiveCamera,
  Vector3,
} from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import type {
  CameraState,
  CatalogArtifactDetail,
  LayerState,
  Manifest,
  Matrix4Tuple,
  PerformanceState,
  PickResult,
  RendererController,
  SourceRecord,
} from '../types/contracts'
import type { TrajectoryPoint } from '../core/trajectory'
import {
  BoundedAbortableCache,
  DEFAULT_POINT_BUDGET,
  GEOMETRY_POOL_BYTES,
  allocatePointBudget,
} from '../core/budget'
import { rowMajorMatrix4 } from '../core/geo'
import { AbortableRangeGetter } from '../core/rangeGetter'

interface DetailArtifact {
  detail: CatalogArtifactDetail
  manifest: Manifest
  renderMatrix: Matrix4Tuple
  sources: SourceRecord[]
}

interface CloudEntry {
  artifact: DetailArtifact
  source: COPCSource
  ranges: AbortableRangeGetter
  cloud: PointCloud
  nodes: Map<string, PointCloudNode>
}

export class DetailRenderer implements RendererController {
  readonly mode = 'detail' as const
  private readonly instance: Instance
  private readonly controls: OrbitControls
  private readonly clouds = new Map<string, CloudEntry>()
  private readonly trajectories: Line[] = []
  private readonly attributeColorMaps = new Map<string, ColorMap>()
  private readonly attributeCache = new BoundedAbortableCache<Record<string, number>>(32 * 1024 * 1024, 24)
  private layerState: LayerState[] = []
  private lastMemoryGuardAt = 0
  private pickOperation: AbortController | null = null
  private disposed = false

  constructor(target: HTMLDivElement, cameraState: CameraState) {
    const camera = new PerspectiveCamera(50, 1, 0.01, 100_000)
    this.instance = new Instance({
      target,
      crs: CoordinateSystem.unknown,
      camera,
      backgroundColor: '#070b10',
    })
    this.controls = new OrbitControls(camera, this.instance.domElement)
    this.controls.enableDamping = true
    this.controls.addEventListener('change', () => this.instance.notifyChange(camera))
    this.setCamera(cameraState)
  }

  async addArtifact(artifact: DetailArtifact, signal: AbortSignal): Promise<void> {
    if (this.disposed || signal.aborted) throw new DOMException('Detail load aborted', 'AbortError')
    const representation = artifact.detail.representations.find(
      item => item.kind === 'points' && item.format.toLowerCase().includes('copc'),
    )
    if (!representation) throw new Error(`Artifact ${artifact.detail.artifact_id} has no COPC representation`)
    const ranges = new AbortableRangeGetter(representation.asset_url)
    signal.addEventListener('abort', () => ranges.abort(), { once: true })
    const source = new COPCSource({
      url: ranges.get,
      decimate: 2,
      enableWorkers: true,
    })
    const cloud = new PointCloud({
      source,
      name: artifact.detail.artifact_id,
      cleanupDelay: 0,
    })
    cloud.object3d.matrixAutoUpdate = false
    cloud.object3d.matrix.copy(rowMajorMatrix4(artifact.renderMatrix))
    cloud.pointSize = 2
    cloud.subdivisionThreshold = 1
    cloud.decimation = 2
    cloud.pointBudget = DEFAULT_POINT_BUDGET
    try {
      await this.instance.add(cloud)
      if (this.disposed || signal.aborted) {
        this.instance.remove(cloud)
        cloud.dispose()
        source.dispose()
        throw new DOMException('Detail load aborted', 'AbortError')
      }
      const root = await source.getHierarchy()
      if (this.disposed || signal.aborted) {
        this.instance.remove(cloud)
        cloud.dispose()
        source.dispose()
        throw new DOMException('Detail hierarchy load aborted', 'AbortError')
      }
      const nodes = new Map<string, PointCloudNode>()
      traverseNode(root, node => {
        nodes.set(node.id, node)
        return true
      })
      this.clouds.set(artifact.detail.artifact_id, {
        artifact,
        source,
        ranges,
        cloud,
        nodes,
      })
      this.setLayers(this.layerState)
      this.instance.notifyChange(cloud)
    } catch (error) {
      cloud.dispose()
      source.dispose()
      ranges.abort()
      throw error
    }
  }

  addTrajectory(points: readonly TrajectoryPoint[]): void {
    if (points.length < 2 || this.disposed) return
    const geometry = new BufferGeometry().setFromPoints(
      points.map(point => new Vector3(...point.position)),
    )
    const line = new Line(geometry, new LineBasicMaterial({ color: '#ff5e8a' }))
    line.name = 'camera trajectory'
    this.trajectories.push(line)
    this.instance.add(line)
    this.instance.notifyChange(line)
  }

  setCamera(state: CameraState): void {
    this.instance.view.camera.position.fromArray([...state.position])
    this.instance.view.camera.up.fromArray([...(state.up ?? [0, 0, 1])])
    this.controls.target.fromArray([...state.target])
    this.controls.update()
    this.instance.notifyChange()
  }

  getCamera(): CameraState {
    const position = this.instance.view.camera.position
    const target = this.controls.target
    return {
      position: [position.x, position.y, position.z],
      target: [target.x, target.y, target.z],
      up: [this.instance.view.camera.up.x, this.instance.view.camera.up.y, this.instance.view.camera.up.z],
    }
  }

  setLayers(layers: LayerState[]): void {
    this.layerState = layers
    const visible = new Set(layers.filter(layer => layer.visible).map(layer => layer.artifactId))
    const allocations = allocatePointBudget(
      layers.map(layer => layer.artifactId),
      visible,
    )
    for (const layer of layers) {
      const entry = this.clouds.get(layer.artifactId)
      if (!entry) continue
      entry.cloud.visible = layer.visible
      entry.cloud.opacity = layer.opacity
      entry.cloud.pointSize = layer.pointSize
      entry.cloud.pointBudget = allocations[layer.artifactId] || 1
      if (layer.visible) this.applyColorMode(entry, layer.colorMode)
      this.instance.notifyChange(entry.cloud)
    }
  }

  private applyColorMode(entry: CloudEntry, mode: LayerState['colorMode']): void {
    const { cloud } = entry
    const candidates: Record<LayerState['colorMode'], string[]> = {
      rgb: ['Color', 'RGB'],
      elevation: ['Z', 'Elevation'],
      source: ['PointSourceId', 'SourceIndex'],
      confidence: ['Confidence'],
    }
    const supported = new Set(cloud.getSupportedAttributes().map(attribute => attribute.name))
    const attribute = candidates[mode].find(name => supported.has(name))
    if (!attribute) return
    if (mode === 'source') {
      const sourceIndices = entry.artifact.sources.map(source => source.source_index)
      const min = sourceIndices.length ? Math.min(...sourceIndices) : 0
      const max = sourceIndices.length ? Math.max(...sourceIndices) : 255
      const key = `${entry.artifact.detail.artifact_id}:${attribute}`
      let colorMap = this.attributeColorMaps.get(key)
      if (!colorMap) {
        colorMap = new ColorMap({
          colors: [
            '#36d6c0',
            '#ffcb52',
            '#ff668f',
            '#7ea6ff',
            '#c68aff',
            '#ff843d',
            '#49d17d',
            '#f184dc',
            '#4fc7ef',
            '#e8ed55',
            '#ff806f',
            '#a8df4d',
          ].map(value => new Color(value)),
          min,
          max: Math.max(min + 1, max),
        })
        this.attributeColorMaps.set(key, colorMap)
      }
      cloud.setAttributeColorMap(attribute, colorMap)
    }
    cloud.setColoringMode('attribute')
    cloud.setActiveAttribute(attribute)
  }

  async pick(event: PointerEvent): Promise<PickResult | null> {
    this.pickOperation?.abort()
    const operation = new AbortController()
    this.pickOperation = operation
    const hit = this.instance.pickObjectsAt(event, {
      where: [...this.clouds.values()].map(entry => entry.cloud),
      // Sparse coarse LODs rarely place a point on the exact device pixel
      // beneath a double-click. Giro3D supports a screen-space search radius;
      // eight pixels makes inspection forgiving without selecting distant
      // geometry or changing the rendered point size.
      radius: 8,
      sortByDistance: true,
      limit: 1,
    })[0]
    if (!hit?.entity || hit.index === undefined) return null
    const entry = [...this.clouds.values()].find(candidate => candidate.cloud === hit.entity)
    if (!entry) return null
    const nodeId = hit.object.name
    const node = entry.nodes.get(nodeId)
    const pointIndex = hit.index
    const localPoint = hit.point.clone().applyMatrix4(
      rowMajorMatrix4(entry.artifact.renderMatrix).invert(),
    )
    const result: PickResult = {
      artifactId: entry.artifact.detail.artifact_id,
      nodeId,
      pointIndex,
      localCoordinate: [localPoint.x, localPoint.y, localPoint.z],
    }
    if (!node) return result
    const attributes = entry.cloud.getSupportedAttributes().filter(
      attribute => ['PointSourceId', 'SourceIndex', 'Confidence', 'ContributorCount'].includes(attribute.name),
    )
    const cacheKey = `${result.artifactId}:${nodeId}:${pointIndex}`
    const cached = this.attributeCache.get(cacheKey)
    if (cached) return { ...result, ...attributeValues(cached) }
    const data = await entry.source.getNodeData({ node, position: false, attributes })
    if (this.disposed || operation.signal.aborted) return null
    const values: Record<string, number> = {}
    attributes.forEach((attribute, index) => {
      const buffer = data.attributes[index]
      if (buffer && pointIndex < buffer.count) values[attribute.name] = Number(buffer.getX(pointIndex))
    })
    this.attributeCache.set(cacheKey, values, attributes.length * 8)
    return { ...result, ...attributeValues(values) }
  }

  getPerformance(): PerformanceState {
    const visiblePoints = [...this.clouds.values()]
      .reduce((sum, entry) => sum + entry.cloud.displayedPointCount, 0)
    const memory = this.instance.getMemoryUsage()
    this.applyMemoryPressure(memory.cpuMemory, memory.gpuMemory)
    return {
      visiblePoints,
      pointBudget: DEFAULT_POINT_BUDGET,
      cpuGeometryBytes: memory.cpuMemory,
      gpuGeometryBytes: memory.gpuMemory,
      cpuGeometryLimitBytes: GEOMETRY_POOL_BYTES,
      gpuGeometryLimitBytes: GEOMETRY_POOL_BYTES,
    }
  }

  private applyMemoryPressure(cpuBytes: number, gpuBytes: number): void {
    if (cpuBytes <= GEOMETRY_POOL_BYTES && gpuBytes <= GEOMETRY_POOL_BYTES) return
    const now = performance.now()
    if (now - this.lastMemoryGuardAt < 1_000) return
    this.lastMemoryGuardAt = now
    const ratio = Math.min(
      GEOMETRY_POOL_BYTES / Math.max(cpuBytes, 1),
      GEOMETRY_POOL_BYTES / Math.max(gpuBytes, 1),
    )
    for (const { cloud } of this.clouds.values()) {
      // Giro3D 2 does not expose a byte-sized point-cloud cache knob. Raising
      // SSE pressure selects a coarser hierarchy; immediate cleanup then
      // releases the fine nodes. Raw memory stays visible to the benchmark so
      // failure to converge under the hard gate is never hidden.
      cloud.subdivisionThreshold = Math.min(
        16,
        Math.max(1, cloud.subdivisionThreshold / Math.max(ratio * 0.9, 0.25)),
      )
      cloud.cleanupDelay = 0
      cloud.clear()
      this.instance.notifyChange(cloud)
    }
  }

  dispose(): void {
    if (this.disposed) return
    this.disposed = true
    this.controls.dispose()
    this.pickOperation?.abort()
    this.pickOperation = null
    this.attributeCache.clear()
    for (const colorMap of this.attributeColorMaps.values()) colorMap.dispose()
    this.attributeColorMaps.clear()
    for (const { cloud, source, ranges } of this.clouds.values()) {
      this.instance.remove(cloud)
      cloud.clear()
      cloud.dispose()
      source.dispose()
      ranges.abort()
    }
    this.clouds.clear()
    for (const line of this.trajectories) {
      this.instance.remove(line)
      line.geometry.dispose()
      ;(line.material as LineBasicMaterial).dispose()
    }
    this.trajectories.length = 0
    const renderer = this.instance.renderer
    this.instance.dispose()
    // WebGLRenderer.dispose() frees Three.js resources but deliberately keeps
    // the browser context alive. Detail/overview transitions create distinct
    // Giro3D instances, so explicitly release the retired context instead of
    // accumulating contexts and driver state across site transitions.
    renderer.forceContextLoss()
  }
}

function attributeValues(values: Record<string, number>): Partial<PickResult> {
  return {
    pointSourceId: values.PointSourceId ?? values.SourceIndex,
    confidence: values.Confidence,
    contributorCount: values.ContributorCount,
  }
}
