import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { Manifest } from '../types/contracts'
import { LayerControls } from './LayerControls'

describe('detail controls', () => {
  it('disables attribute modes absent from the package and updates display controls', () => {
    const onChange = vi.fn()
    render(<LayerControls
      layers={[{
        artifactId: 'a',
        visible: true,
        opacity: 1,
        pointSize: 2,
        colorMode: 'rgb',
        loading: false,
        progress: 1,
        availableColorModes: ['rgb', 'elevation'],
      }]}
      manifests={new Map()}
      performance={{
        visiblePoints: 1_000_000,
        pointBudget: 2_000_000,
        cpuGeometryBytes: 0,
        gpuGeometryBytes: 0,
        cpuGeometryLimitBytes: 256 * 1024 * 1024,
        gpuGeometryLimitBytes: 256 * 1024 * 1024,
      }}
      onChange={onChange}
    />)
    expect(screen.getByRole('option', { name: /Source/ })).toBeDisabled()
    const opacity = document.querySelectorAll<HTMLInputElement>('input[type="range"]')[0]
    fireEvent.change(opacity, { target: { value: '0.5' } })
    expect(onChange).toHaveBeenCalledWith('a', { opacity: 0.5 })
    expect(screen.getByText(/2,000,000 visible points/)).toBeInTheDocument()
  })

  it('keeps the specific unaligned rejection reason visible', () => {
    const manifest: Manifest = {
      schema_version: '1.0.0',
      artifact_id: 'a',
      capture_id: 'capture',
      run_id: 'run',
      created_at: '2026-07-26T00:00:00Z',
      coordinate_frame: {
        name: 'artifact_local',
        units: 'unknown',
        axis_order: ['x', 'y', 'z'],
        handedness: 'right',
      },
      alignment: {
        status: 'unaligned',
        method: 'none',
        model_to_artifact_local: [
          1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1, 0,
          0, 0, 0, 1,
        ],
        scale: 1,
        inlier_count: 0,
        rejection_reason: 'gps_telemetry_unavailable',
      },
      artifacts: [],
    }
    render(<LayerControls
      layers={[{
        artifactId: 'a',
        visible: true,
        opacity: 1,
        pointSize: 2,
        colorMode: 'elevation',
        loading: false,
        progress: 1,
        availableColorModes: ['elevation'],
      }]}
      manifests={new Map([['a', manifest]])}
      performance={{
        visiblePoints: 0,
        pointBudget: 2_000_000,
        cpuGeometryBytes: 0,
        gpuGeometryBytes: 0,
        cpuGeometryLimitBytes: 256 * 1024 * 1024,
        gpuGeometryLimitBytes: 256 * 1024 * 1024,
      }}
      onChange={vi.fn()}
    />)

    expect(screen.getByText(/gps_telemetry_unavailable/)).toBeInTheDocument()
  })
})
