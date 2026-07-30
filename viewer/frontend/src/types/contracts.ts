export type SceneMode = 'overview' | 'detail'
export type AlignmentStatus = 'unaligned' | 'approximate' | 'aligned' | 'reviewed'
export type ColorMode = 'rgb' | 'elevation' | 'source' | 'confidence'
export type Vector3Tuple = readonly [number, number, number]
export type Matrix4Tuple = readonly [
  number, number, number, number,
  number, number, number, number,
  number, number, number, number,
  number, number, number, number,
]

export interface CameraState {
  position: Vector3Tuple
  target: Vector3Tuple
  up?: Vector3Tuple
}

export interface Bounds {
  west: number
  south: number
  east: number
  north: number
}

export interface GeoJsonPolygon {
  type: 'Polygon'
  coordinates: number[][][]
}

export interface CatalogArtifact {
  artifact_id: string
  run_id: string
  capture_id: string
  kind: string
  alignment_status: AlignmentStatus
  frame_name: string
  units: 'metre' | 'unknown'
  point_count: number | null
  footprint: GeoJsonPolygon | null
  created_at: string
}

export interface Representation {
  representation_id: string
  kind: string
  format: string
  relative_path: string
  media_type: string
  byte_size: number
  sha256: string
  asset_url: string
}

export interface LayerDefault {
  visible: boolean
  opacity: number
  point_budget?: number | null
  point_size?: number | null
  color_dimension?: string | null
}

export interface CatalogArtifactDetail extends CatalogArtifact {
  manifest_sha256: string
  layer_default: LayerDefault | null
  representations: Representation[]
}

export interface ArtifactFile {
  representation_id: string
  kind: string
  format: string
  path: string
  media_type: string
  byte_size: number
  sha256: string
  frame?: string | null
  point_count?: number | null
  required_dimensions?: string[]
  columns?: Array<{ name: string; dtype: string; nullable?: boolean; unit?: string | null }>
  bounds_min?: Vector3Tuple | null
  bounds_max?: Vector3Tuple | null
  metadata?: Record<string, unknown>
}

export interface Manifest {
  schema_version: '1.0.0'
  artifact_id: string
  capture_id: string
  run_id: string
  created_at: string
  artifact_kind?: string
  coordinate_frame: {
    name: string
    units: 'metre' | 'unknown'
    axis_order: string[]
    handedness: string
    origin_wgs84?: Vector3Tuple | null
    transform_to_ecef?: number[] | null
  }
  alignment: {
    status: AlignmentStatus
    method: string
    model_to_artifact_local: number[]
    scale: number
    inlier_count: number
    rejection_reason?: string | null
    horizontal_rmse_m?: number | null
    vertical_rmse_m?: number | null
  }
  footprint_wgs84?: { coordinates: number[][] } | null
  artifacts: ArtifactFile[]
  layer_default?: LayerDefault | null
  [key: string]: unknown
}

export interface SourceRecord {
  source_index: number
  kind: 'window' | 'submap' | 'keyframe_group' | 'batch' | 'capture'
  capture_id: string
  run_id: string
  name?: string | null
  frame_start?: number | null
  frame_end?: number | null
  frame_indices?: number[] | null
  timestamp_start_s?: number | null
  timestamp_end_s?: number | null
  point_count?: number | null
  metadata?: Record<string, unknown>
}

export interface LayerState {
  artifactId: string
  visible: boolean
  opacity: number
  pointSize: number
  colorMode: ColorMode
  loading: boolean
  progress: number
  availableColorModes: ColorMode[]
}

export interface ViewerUrlState {
  version: 1
  mode: SceneMode
  activeArtifactIds: string[]
  selectedSource: number | null
  overviewCamera: CameraState
  detailCamera: CameraState
  layers: Record<string, Pick<LayerState, 'visible' | 'opacity' | 'pointSize' | 'colorMode'>>
}

export interface PickResult {
  artifactId: string
  nodeId: string
  pointIndex: number
  localCoordinate: Vector3Tuple
  pointSourceId?: number
  confidence?: number
  contributorCount?: number
}

export interface Inspection extends PickResult {
  source?: SourceRecord
  alignmentStatus: AlignmentStatus
  horizontalRmseM?: number | null
  verticalRmseM?: number | null
}

export interface PerformanceState {
  visiblePoints: number
  pointBudget: number
  cpuGeometryBytes: number
  gpuGeometryBytes: number
  cpuGeometryLimitBytes: number
  gpuGeometryLimitBytes: number
}

export interface Disposable {
  dispose(): void
}

export interface RendererController extends Disposable {
  readonly mode: SceneMode
  setCamera(state: CameraState): void
  getCamera(): CameraState
  setLayers(layers: LayerState[]): void
  pick(event: PointerEvent): Promise<PickResult | null>
  getPerformance(): PerformanceState
}

export interface LoadingOperation {
  id: string
  artifactId: string
  abortController: AbortController
  startedAt: number
}
