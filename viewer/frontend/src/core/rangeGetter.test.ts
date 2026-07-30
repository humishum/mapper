import { describe, expect, it, vi } from 'vitest'
import { AbortableRangeGetter } from './rangeGetter'

describe('AbortableRangeGetter', () => {
  it('requests an inclusive HTTP range for COPC end-exclusive offsets', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(new Uint8Array([3, 4, 5]), { status: 206 }),
    )
    vi.stubGlobal('fetch', fetchMock)
    const ranges = new AbortableRangeGetter('/asset')

    await expect(ranges.get(3, 6)).resolves.toEqual(new Uint8Array([3, 4, 5]))
    expect(fetchMock).toHaveBeenCalledWith('/asset', expect.objectContaining({
      headers: { Range: 'bytes=3-5' },
    }))
    vi.unstubAllGlobals()
  })

  it('aborts in-flight work and refuses requests after disposal', async () => {
    vi.stubGlobal('fetch', vi.fn((_url, init?: RequestInit) => new Promise((_resolve, reject) => {
      init?.signal?.addEventListener('abort', () => {
        reject(new DOMException('aborted', 'AbortError'))
      })
    })))
    const ranges = new AbortableRangeGetter('/asset')
    const pending = ranges.get(0, 8)
    ranges.abort()

    await expect(pending).rejects.toMatchObject({ name: 'AbortError' })
    await expect(ranges.get(0, 8)).rejects.toMatchObject({ name: 'AbortError' })
    vi.unstubAllGlobals()
  })
})
