import type { components } from './openapi'

/**
 * Wire contracts generated from the checked-in backend OpenAPI document.
 * Domain adapters in client.ts narrow permissive wire strings at the boundary.
 */
export type CatalogArtifactsResponse = components['schemas']['CatalogArtifactsResponse']
export type SourcesResponse = components['schemas']['SourcesResponse']
export type ArtifactDetailResponse = components['schemas']['ArtifactDetail']
