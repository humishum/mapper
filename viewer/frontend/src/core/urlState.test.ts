import { describe, expect, it } from 'vitest'
import { DEFAULT_URL_STATE, parseUrlState, serializeUrlState } from './urlState'

describe('versioned URL state', () => {
  it('round trips mode, two artifacts, selection, cameras, and layers', () => {
    const state = {
      ...DEFAULT_URL_STATE,
      mode: 'detail' as const,
      activeArtifactIds: ['a', 'b', 'ignored'],
      selectedSource: 7,
      layers: {
        a: { visible: true, opacity: 0.5, pointSize: 3, colorMode: 'source' as const },
      },
    }
    const parsed = parseUrlState(serializeUrlState(state))
    expect(parsed.mode).toBe('detail')
    expect(parsed.activeArtifactIds).toEqual(['a', 'b'])
    expect(parsed.selectedSource).toBe(7)
    expect(parsed.layers.a.colorMode).toBe('source')
  })

  it('fails closed for unknown versions and malformed state', () => {
    expect(parseUrlState('?v=9&mode=detail&artifacts=a')).toEqual(DEFAULT_URL_STATE)
    expect(parseUrlState('?v=1&mode=detail&layers=%7Bbad')).toMatchObject({
      mode: 'overview',
      layers: {},
    })
  })
})
