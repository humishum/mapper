import type { CameraState, ViewerUrlState } from '../types/contracts'

const OVERVIEW_CAMERA: CameraState = {
  position: [10_000_000, 0, 5_000_000],
  target: [0, 0, 0],
  up: [0, 0, 1],
}
const DETAIL_CAMERA: CameraState = {
  position: [30, 30, 20],
  target: [0, 0, 0],
  up: [0, 0, 1],
}

export const DEFAULT_URL_STATE: ViewerUrlState = {
  version: 1,
  mode: 'overview',
  activeArtifactIds: [],
  selectedSource: null,
  overviewCamera: OVERVIEW_CAMERA,
  detailCamera: DETAIL_CAMERA,
  layers: {},
}

function camera(value: unknown, fallback: CameraState): CameraState {
  if (!Array.isArray(value) || value.length !== 2) return fallback
  const [position, target] = value
  if (![position, target].every(v => Array.isArray(v) && v.length === 3 && v.every(Number.isFinite))) {
    return fallback
  }
  return { position: position as unknown as CameraState['position'], target: target as unknown as CameraState['target'], up: [0, 0, 1] }
}

export function parseUrlState(search: string): ViewerUrlState {
  const params = new URLSearchParams(search)
  if (params.get('v') !== '1') return DEFAULT_URL_STATE
  const activeArtifactIds = (params.get('artifacts') ?? '').split(',').filter(Boolean).slice(0, 2)
  let layers: ViewerUrlState['layers'] = {}
  try {
    layers = JSON.parse(params.get('layers') ?? '{}') as ViewerUrlState['layers']
  } catch {
    layers = {}
  }
  const parseCamera = (name: string, fallback: CameraState) => {
    try {
      return camera(JSON.parse(params.get(name) ?? 'null'), fallback)
    } catch {
      return fallback
    }
  }
  const selected = params.get('source')
  return {
    version: 1,
    mode: params.get('mode') === 'detail' && activeArtifactIds.length ? 'detail' : 'overview',
    activeArtifactIds,
    selectedSource: selected !== null && Number.isInteger(Number(selected)) ? Number(selected) : null,
    overviewCamera: parseCamera('overviewCamera', OVERVIEW_CAMERA),
    detailCamera: parseCamera('detailCamera', DETAIL_CAMERA),
    layers,
  }
}

export function serializeUrlState(state: ViewerUrlState): string {
  const params = new URLSearchParams()
  params.set('v', '1')
  params.set('mode', state.mode)
  if (state.activeArtifactIds.length) params.set('artifacts', state.activeArtifactIds.slice(0, 2).join(','))
  if (state.selectedSource !== null) params.set('source', String(state.selectedSource))
  params.set('overviewCamera', JSON.stringify([state.overviewCamera.position, state.overviewCamera.target]))
  params.set('detailCamera', JSON.stringify([state.detailCamera.position, state.detailCamera.target]))
  if (Object.keys(state.layers).length) params.set('layers', JSON.stringify(state.layers))
  return `?${params}`
}
