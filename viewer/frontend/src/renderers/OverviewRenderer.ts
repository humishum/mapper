import Instance from '@giro3d/giro3d/core/Instance'
import { CoordinateSystem } from '@giro3d/giro3d/core/geographic/CoordinateSystem'
import ColorLayer from '@giro3d/giro3d/core/layer/ColorLayer'
import Globe from '@giro3d/giro3d/entities/Globe'
import TiledImageSource from '@giro3d/giro3d/sources/TiledImageSource'
import XYZ from 'ol/source/XYZ'
import {
  BufferGeometry,
  Line,
  LineBasicMaterial,
  Mesh,
  MeshBasicMaterial,
  PerspectiveCamera,
  SphereGeometry,
  Vector3,
} from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import type {
  CameraState,
  CatalogArtifact,
  LayerState,
  PerformanceState,
  PickResult,
  RendererController,
} from '../types/contracts'
import { wgs84ToEcef } from '../core/geo'
import { DEFAULT_POINT_BUDGET, GEOMETRY_POOL_BYTES } from '../core/budget'
import { basemapConfig } from './config'

const EARTH_RADIUS = 6_371_008.8

export class OverviewRenderer implements RendererController {
  readonly mode = 'overview' as const
  private readonly instance: Instance
  private readonly controls: OrbitControls
  private readonly globe: Globe
  private readonly basemapLayer: ColorLayer | null
  private readonly catalogObjects: Array<Mesh | Line> = []
  private disposed = false

  constructor(
    target: HTMLDivElement,
    cameraState: CameraState,
    artifacts: CatalogArtifact[],
    private readonly onArtifactClick: (artifactId: string) => void,
    private readonly onBoundsChange?: (bounds: { west: number; south: number; east: number; north: number }) => void,
    private readonly onReady?: () => void,
  ) {
    const camera = new PerspectiveCamera(35, 1, 10, 50_000_000)
    this.instance = new Instance({
      target,
      crs: CoordinateSystem.epsg4978,
      camera,
      backgroundColor: '#02060b',
    })
    this.controls = new OrbitControls(camera, this.instance.domElement)
    this.controls.enableDamping = true
    this.controls.minDistance = EARTH_RADIUS + 100
    this.controls.maxDistance = 40_000_000
    this.controls.addEventListener('change', () => this.instance.notifyChange(camera))
    this.controls.addEventListener('end', this.emitBounds)
    this.setCamera(cameraState)
    this.globe = new Globe({
      backgroundColor: '#173a50',
      subdivisionThreshold: 1,
      terrain: false,
    })
    this.basemapLayer = basemapConfig.enabled
      ? new ColorLayer({
          name: 'configured basemap',
          source: new TiledImageSource({
            source: new XYZ({
              url: basemapConfig.url,
              attributions: basemapConfig.attribution,
              crossOrigin: 'anonymous',
              wrapX: true,
            }),
          }),
        })
      : null
    void this.addGlobe()
    this.setCatalog(artifacts)
    this.instance.domElement.addEventListener('click', this.handleClick)
  }

  private async addGlobe(): Promise<void> {
    await this.instance.add(this.globe)
    if (this.disposed) {
      this.instance.remove(this.globe)
      return
    }
    if (this.basemapLayer) {
      await this.globe.addLayer(this.basemapLayer)
      if (this.disposed) {
        this.globe.removeLayer(this.basemapLayer, { disposeLayer: true })
        return
      }
    }
    this.instance.notifyChange(this.globe)
    await new Promise<void>(resolve => {
      requestAnimationFrame(() => requestAnimationFrame(() => resolve()))
    })
    if (!this.disposed) this.onReady?.()
  }

  setCatalog(artifacts: CatalogArtifact[]): void {
    for (const object of this.catalogObjects) {
      this.instance.remove(object)
      object.geometry.dispose()
      ;(object.material as MeshBasicMaterial | LineBasicMaterial).dispose()
    }
    this.catalogObjects.length = 0
    for (const artifact of artifacts) {
      if (!artifact.footprint) continue
      const ring = artifact.footprint.coordinates[0] ?? []
      const positions = ring.map(([lon, lat]) => {
        const [x, y, z] = wgs84ToEcef(lon, lat, 150)
        return new Vector3(x, y, z)
      })
      if (positions.length >= 2) {
        const geometry = new BufferGeometry().setFromPoints(positions)
        const line = new Line(geometry, new LineBasicMaterial({ color: '#58e6c2' }))
        line.userData.artifactId = artifact.artifact_id
        this.catalogObjects.push(line)
        this.instance.add(line)
      }
      const centroid = positions.reduce((sum, point) => sum.add(point), new Vector3())
        .multiplyScalar(positions.length ? 1 / positions.length : 0)
        .setLength(EARTH_RADIUS + 4_000)
      const marker = new Mesh(
        new SphereGeometry(8_000, 12, 8),
        new MeshBasicMaterial({ color: '#ffbf47' }),
      )
      marker.position.copy(centroid)
      marker.userData.artifactId = artifact.artifact_id
      this.catalogObjects.push(marker)
      this.instance.add(marker)
    }
    this.instance.notifyChange()
  }

  private readonly handleClick = (event: MouseEvent) => {
    const hit = this.instance.pickObjectsAt(event, {
      where: this.catalogObjects,
      sortByDistance: true,
      limit: 1,
    })[0]
    const artifactId = hit?.object.userData.artifactId as string | undefined
    if (artifactId) this.onArtifactClick(artifactId)
  }

  private readonly emitBounds = () => {
    if (!this.onBoundsChange) return
    const position = this.instance.view.camera.position
    const longitude = Math.atan2(position.y, position.x) * 180 / Math.PI
    const latitude = Math.atan2(position.z, Math.hypot(position.x, position.y)) * 180 / Math.PI
    const altitude = Math.max(0, position.length() - EARTH_RADIUS)
    const span = Math.min(180, Math.max(2, 8 + altitude / 100_000))
    this.onBoundsChange({
      west: longitude - span,
      south: Math.max(-90, latitude - span / 2),
      east: longitude + span,
      north: Math.min(90, latitude + span / 2),
    })
  }

  setCamera(state: CameraState): void {
    this.instance.view.camera.position.fromArray([...state.position])
    this.controls.target.fromArray([...state.target])
    this.instance.view.camera.up.fromArray([...(state.up ?? [0, 0, 1])])
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
    void layers
  }

  async pick(event: PointerEvent): Promise<PickResult | null> {
    void event
    return null
  }

  getPerformance(): PerformanceState {
    return {
      visiblePoints: 0,
      pointBudget: DEFAULT_POINT_BUDGET,
      cpuGeometryBytes: 0,
      gpuGeometryBytes: 0,
      cpuGeometryLimitBytes: GEOMETRY_POOL_BYTES,
      gpuGeometryLimitBytes: GEOMETRY_POOL_BYTES,
    }
  }

  dispose(): void {
    if (this.disposed) return
    this.disposed = true
    this.instance.domElement.removeEventListener('click', this.handleClick)
    this.controls.removeEventListener('end', this.emitBounds)
    this.controls.dispose()
    for (const object of this.catalogObjects) {
      object.geometry.dispose()
      ;(object.material as MeshBasicMaterial | LineBasicMaterial).dispose()
    }
    if (this.basemapLayer && this.globe.contains(this.basemapLayer)) {
      this.globe.removeLayer(this.basemapLayer, { disposeLayer: true })
    }
    this.instance.remove(this.globe)
    const renderer = this.instance.renderer
    this.instance.dispose()
    renderer.forceContextLoss()
  }
}
