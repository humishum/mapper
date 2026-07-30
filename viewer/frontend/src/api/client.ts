import type { ArtifactDetailResponse, CatalogArtifactsResponse, SourcesResponse } from './models'
import type {
  Bounds,
  CatalogArtifact,
  CatalogArtifactDetail,
  Manifest,
} from '../types/contracts'
import { splitAntimeridianBounds } from '../core/geo'
import { validateManifest, validateMetrics, validateSources } from '../core/schema'

const API_ROOT = (import.meta.env.VITE_API_ROOT as string | undefined)?.replace(/\/$/, '') ?? ''

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly url: string,
  ) {
    super(message)
  }
}

async function requestJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${API_ROOT}${path}`, {
    signal,
    headers: { Accept: 'application/json' },
  })
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`
    try {
      const body = await response.json() as { detail?: string }
      detail = body.detail ?? detail
    } catch {
      // The HTTP status remains useful if an upstream proxy returned non-JSON.
    }
    throw new ApiError(detail, response.status, path)
  }
  return response.json() as Promise<T>
}

function bboxParam(bounds: Bounds): string {
  return [bounds.west, bounds.south, bounds.east, bounds.north].join(',')
}

function domainArtifact(
  item: CatalogArtifactsResponse['artifacts'][number],
): CatalogArtifact {
  if (item.units !== 'metre' && item.units !== 'unknown') {
    throw new Error(`Catalog artifact ${item.artifact_id} has unsupported units: ${item.units}`)
  }
  return {
    ...item,
    units: item.units,
    point_count: item.point_count ?? null,
    footprint: item.footprint ?? null,
  }
}

export async function fetchCatalogBounds(
  bounds: Bounds,
  signal?: AbortSignal,
): Promise<CatalogArtifact[]> {
  const queries = splitAntimeridianBounds(bounds).map(async part => {
    const params = new URLSearchParams({ bbox: bboxParam(part), limit: '1000' })
    return requestJson<CatalogArtifactsResponse>(`/api/v1/catalog/artifacts?${params}`, signal)
  })
  const pages = await Promise.all(queries)
  const artifacts = pages.flatMap(page => page.artifacts).map(domainArtifact)
  return [...new Map(artifacts.map(item => [item.artifact_id, item])).values()]
}

export async function fetchUnaligned(signal?: AbortSignal): Promise<CatalogArtifact[]> {
  const params = new URLSearchParams({ alignment_status: 'unaligned', limit: '1000' })
  const response = await requestJson<CatalogArtifactsResponse>(
    `/api/v1/catalog/artifacts?${params}`,
    signal,
  )
  return response.artifacts.map(domainArtifact)
}

export async function fetchArtifactDetail(
  artifactId: string,
  signal?: AbortSignal,
): Promise<CatalogArtifactDetail> {
  const item = await requestJson<ArtifactDetailResponse>(
    `/api/v1/catalog/artifacts/${encodeURIComponent(artifactId)}`,
    signal,
  )
  const summary = domainArtifact(item)
  return {
    ...summary,
    manifest_sha256: item.manifest_sha256,
    layer_default: item.layer_default ?? null,
    representations: item.representations,
  }
}

export async function fetchManifest(artifactId: string, signal?: AbortSignal): Promise<Manifest> {
  const value = await requestJson<unknown>(
    `/api/v1/catalog/artifacts/${encodeURIComponent(artifactId)}/manifest`,
    signal,
  )
  return validateManifest(value)
}

export async function validateSelectedSidecars(
  detail: CatalogArtifactDetail,
  signal?: AbortSignal,
): Promise<void> {
  const metrics = detail.representations.find(item => item.kind === 'metrics')
  if (!metrics) return
  const value = await requestJson<unknown>(metrics.asset_url, signal)
  validateMetrics(value)
}

export async function fetchSources(artifactId: string, signal?: AbortSignal) {
  const response = await requestJson<SourcesResponse>(
    `/api/v1/catalog/artifacts/${encodeURIComponent(artifactId)}/sources`,
    signal,
  )
  return validateSources(response.sources)
}
