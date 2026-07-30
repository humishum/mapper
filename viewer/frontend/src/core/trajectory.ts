import { asyncBufferFromUrl, parquetReadObjects } from 'hyparquet'
import { Vector3 } from 'three'
import type { Matrix4Tuple, Vector3Tuple } from '../types/contracts'
import { rowMajorMatrix4 } from './geo'

export interface TrajectoryPoint {
  frameIndex: number
  timestampS: number
  position: Vector3Tuple
}

const REQUIRED_COLUMNS = [
  'frame_index',
  'timestamp_s',
  'tx',
  'ty',
  'tz',
  'qx',
  'qy',
  'qz',
  'qw',
  'fx',
  'fy',
  'cx',
  'cy',
] as const

export async function loadTrajectoryRange(
  assetUrl: string,
  byteLength: number,
  transform: Matrix4Tuple,
  signal: AbortSignal,
  rowStart = 0,
  rowEnd?: number,
): Promise<TrajectoryPoint[]> {
  const file = await asyncBufferFromUrl({
    url: assetUrl,
    byteLength,
    requestInit: { signal },
  })
  const rows = await parquetReadObjects({
    file,
    columns: [...REQUIRED_COLUMNS],
    rowStart,
    rowEnd,
    rowFormat: 'object',
    useOffsetIndex: true,
  })
  if (signal.aborted) throw new DOMException('Trajectory request aborted', 'AbortError')
  const matrix = rowMajorMatrix4(transform)
  return rows.map((row, index) => {
    for (const column of REQUIRED_COLUMNS) {
      if (!Number.isFinite(Number(row[column]))) {
        throw new Error(`poses.parquet row ${index} has invalid ${column}`)
      }
    }
    const local = new Vector3(Number(row.tx), Number(row.ty), Number(row.tz)).applyMatrix4(matrix)
    return {
      frameIndex: Number(row.frame_index),
      timestampS: Number(row.timestamp_s),
      position: [local.x, local.y, local.z],
    }
  })
}
