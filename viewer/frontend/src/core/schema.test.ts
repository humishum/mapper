import { describe, expect, it } from 'vitest'
import {
  availableColorModes,
  validateManifest,
  validateMetrics,
  validateSources,
} from './schema'

const identity = [
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
]

function manifest() {
  return {
    schema_version: '1.0.0',
    run_id: 'run',
    capture_id: 'capture',
    artifact_id: 'artifact',
    created_at: '2026-07-25T00:00:00Z',
    capture: {},
    producer: {
      model_name: 'test',
      adapter_name: 'test',
      adapter_version: '1',
      publisher_name: 'test',
      publisher_version: '1',
    },
    coordinate_frame: {
      name: 'artifact_local',
      units: 'metre',
      axis_order: ['east', 'north', 'up'],
      handedness: 'right',
      origin_wgs84: [-121, 37, 10],
      transform_to_ecef: identity,
    },
    alignment: {
      status: 'aligned',
      method: 'synthetic',
      model_to_artifact_local: identity,
      scale: 1,
      inlier_count: 10,
    },
    artifacts: [{
      representation_id: 'copc',
      kind: 'points',
      format: 'copc/laz',
      path: 'geometry/a.copc.laz',
      media_type: 'application/vnd.laszip',
      byte_size: 100,
      sha256: 'a'.repeat(64),
      required_dimensions: ['X', 'Y', 'Z', 'Red', 'Green', 'Blue', 'PointSourceId'],
    }],
  }
}

describe('package schema validation', () => {
  it('accepts canonical manifests and derives supported color controls', () => {
    const value = validateManifest(manifest())
    expect(availableColorModes(value)).toEqual(['rgb', 'elevation', 'source'])
  })

  it('reports schema failures before rendering', () => {
    expect(() => validateManifest({ ...manifest(), schema_version: '2' }))
      .toThrow(/schema validation/)
  })

  it('validates generic provenance kinds without viewer-specific assumptions', () => {
    expect(validateSources([{
      source_index: 0,
      kind: 'submap',
      capture_id: 'capture',
      run_id: 'run',
      metadata: {},
    }])[0].kind).toBe('submap')
  })

  it('rejects incomplete declared Parquet contracts and invalid metrics sidecars', () => {
    const withIncompletePoses = {
      ...manifest(),
      artifacts: [
        ...manifest().artifacts,
        {
          representation_id: 'poses',
          kind: 'poses',
          format: 'parquet',
          path: 'cameras/poses.parquet',
          media_type: 'application/vnd.apache.parquet',
          byte_size: 100,
          sha256: 'b'.repeat(64),
          columns: [{ name: 'timestamp_s', dtype: 'float64', nullable: false }],
        },
      ],
    }
    expect(() => validateManifest(withIncompletePoses)).toThrow(/frame_index/)
    expect(() => validateMetrics({ stages: [{ name: 'bad', wall_time_s: -1 }] }))
      .toThrow(/Metrics failed/)
  })
})
