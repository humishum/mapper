import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, fetchCatalogBounds } from './client'

describe('catalog client', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('debounces upstream bounds into split catalog requests without touching COPC assets', async () => {
    const fetchMock = vi.fn<(input: RequestInfo | URL, init?: RequestInit) => Promise<Response>>(
      async () => new Response(JSON.stringify({ count: 0, artifacts: [] }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      }),
    )
    vi.stubGlobal('fetch', fetchMock)
    await fetchCatalogBounds({ west: 170, south: -5, east: -170, north: 5 })
    expect(fetchMock).toHaveBeenCalledTimes(2)
    const urls = fetchMock.mock.calls.map(call => String(call[0]))
    expect(urls.every(url => url.startsWith('/api/v1/catalog/artifacts?'))).toBe(true)
    expect(urls.some(url => url.includes('/assets/'))).toBe(false)
  })

  it('surfaces a typed catalog error with the backend detail and request path', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(
      JSON.stringify({ detail: 'catalog unavailable' }),
      {
        status: 503,
        headers: { 'content-type': 'application/json' },
      },
    )))

    const failure = await fetchCatalogBounds({
      west: -123,
      south: 37,
      east: -122,
      north: 38,
    }).catch(error => error)

    expect(failure).toBeInstanceOf(ApiError)
    expect(failure).toMatchObject({
      message: 'catalog unavailable',
      status: 503,
    })
    expect((failure as ApiError).url).toMatch(/^\/api\/v1\/catalog\/artifacts\?/)
  })
})
