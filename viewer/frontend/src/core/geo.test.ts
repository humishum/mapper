import { describe, expect, it } from 'vitest'
import {
  matrixTranslation,
  relativeArtifactMatrix,
  rowMajorMatrix4,
  splitAntimeridianBounds,
  wgs84ToEcef,
} from './geo'

describe('geographic transforms', () => {
  it('splits an antimeridian-crossing query into valid backend bboxes', () => {
    expect(splitAntimeridianBounds({ west: 170, south: -10, east: -170, north: 10 })).toEqual([
      { west: 170, south: -10, east: 180, north: 10 },
      { west: -180, south: -10, east: -170, north: 10 },
    ])
  })

  it('preserves a normal bounding box', () => {
    expect(splitAntimeridianBounds({ west: -122, south: 36, east: -120, north: 38 }))
      .toEqual([{ west: -122, south: 36, east: -120, north: 38 }])
  })

  it('applies T(-origin ECEF) to a row-major artifact transform', () => {
    const transform = [
      1, 0, 0, 6_378_140,
      0, 1, 0, 20,
      0, 0, 1, 30,
      0, 0, 0, 1,
    ]
    const relative = relativeArtifactMatrix([6_378_137, 10, 20], transform)
    expect(matrixTranslation(relative)).toEqual([3, 10, 10])
    expect(new Array(...rowMajorMatrix4(relative).elements).slice(12, 15)).toEqual([3, 10, 10])
  })

  it('uses float64 WGS84 ECEF conversion', () => {
    const [x, y, z] = wgs84ToEcef(0, 0, 0)
    expect(x).toBeCloseTo(6_378_137, 6)
    expect(y).toBe(0)
    expect(z).toBe(0)
  })
})
