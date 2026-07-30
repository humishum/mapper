import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { CatalogArtifact } from '../types/contracts'
import { CatalogPanel } from './CatalogPanel'

const localArtifact: CatalogArtifact = {
  artifact_id: 'local-artifact',
  run_id: 'run',
  capture_id: 'capture',
  kind: 'points',
  alignment_status: 'unaligned',
  frame_name: 'artifact_local',
  units: 'unknown',
  point_count: null,
  footprint: null,
  created_at: '2026-07-26T00:00:00Z',
}

describe('catalog panel', () => {
  it('separates unaligned scenes, shows their warning, and opens them locally', () => {
    const onOpen = vi.fn()
    render(<CatalogPanel
      geographic={[]}
      unaligned={[localArtifact]}
      activeIds={[]}
      mode="overview"
      loading={false}
      onOpen={onOpen}
    />)

    expect(screen.getByRole('heading', { name: 'Unaligned local scenes' }))
      .toBeInTheDocument()
    expect(screen.getByText('Local only')).toBeInTheDocument()
    expect(screen.getByText('points · unknown size')).toBeInTheDocument()
    fireEvent.click(screen.getByTestId('open-artifact-local-artifact'))
    expect(onOpen).toHaveBeenCalledWith('local-artifact')
  })
})
