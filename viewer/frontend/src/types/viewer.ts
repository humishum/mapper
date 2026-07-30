import type {
  CatalogArtifactDetail,
  Manifest,
  Matrix4Tuple,
  SourceRecord,
} from './contracts'

export interface DetailSceneArtifact {
  detail: CatalogArtifactDetail
  manifest: Manifest
  renderMatrix: Matrix4Tuple
  sources: SourceRecord[]
}
