import { Matrix4 } from 'three'
import type { Bounds, Matrix4Tuple, Vector3Tuple } from '../types/contracts'

const WGS84_A = 6378137
const WGS84_E2 = 6.69437999014e-3

export function splitAntimeridianBounds(bounds: Bounds): Bounds[] {
  const west = normalizeLongitude(bounds.west)
  const east = normalizeLongitude(bounds.east)
  if (bounds.east - bounds.west >= 360) {
    return [{ ...bounds, west: -180, east: 180 }]
  }
  if (west <= east && bounds.west <= bounds.east) {
    return [{ ...bounds, west, east }]
  }
  return [
    { ...bounds, west, east: 180 },
    { ...bounds, west: -180, east },
  ]
}

export function normalizeLongitude(value: number): number {
  const normalized = ((value + 180) % 360 + 360) % 360 - 180
  return normalized === -180 && value > 0 ? 180 : normalized
}

export function wgs84ToEcef(longitude: number, latitude: number, height = 0): Vector3Tuple {
  const lon = longitude * Math.PI / 180
  const lat = latitude * Math.PI / 180
  const sinLat = Math.sin(lat)
  const cosLat = Math.cos(lat)
  const n = WGS84_A / Math.sqrt(1 - WGS84_E2 * sinLat * sinLat)
  return [
    (n + height) * cosLat * Math.cos(lon),
    (n + height) * cosLat * Math.sin(lon),
    (n * (1 - WGS84_E2) + height) * sinLat,
  ]
}

export function relativeArtifactMatrix(
  activeOriginEcef: Vector3Tuple,
  artifactToEcef: readonly number[],
): Matrix4Tuple {
  if (artifactToEcef.length !== 16) {
    throw new Error('transform_to_ecef must contain 16 float64 values')
  }
  const result = [...artifactToEcef]
  result[3] -= activeOriginEcef[0]
  result[7] -= activeOriginEcef[1]
  result[11] -= activeOriginEcef[2]
  return result as unknown as Matrix4Tuple
}

export function matrixTranslation(matrix: readonly number[]): Vector3Tuple {
  if (matrix.length !== 16) throw new Error('matrix must contain 16 values')
  return [matrix[3], matrix[7], matrix[11]]
}

export function distanceBetweenEcef(a: Vector3Tuple, b: Vector3Tuple): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])
}

export function rowMajorMatrix4(values: readonly number[]): Matrix4 {
  if (values.length !== 16) throw new Error('matrix must contain 16 values')
  return new Matrix4().set(
    values[0], values[1], values[2], values[3],
    values[4], values[5], values[6], values[7],
    values[8], values[9], values[10], values[11],
    values[12], values[13], values[14], values[15],
  )
}
