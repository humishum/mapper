import { useEffect, useState } from 'react'
import { fetchCatalogBounds, fetchUnaligned } from '../api/client'
import type { Bounds, CatalogArtifact } from '../types/contracts'

export interface CatalogState {
  geographic: CatalogArtifact[]
  unaligned: CatalogArtifact[]
  loading: boolean
  error: string | null
}

export function useCatalog(bounds: Bounds): CatalogState {
  const [state, setState] = useState<CatalogState>({
    geographic: [],
    unaligned: [],
    loading: true,
    error: null,
  })

  useEffect(() => {
    const controller = new AbortController()
    const timer = window.setTimeout(async () => {
      setState(current => ({ ...current, loading: true, error: null }))
      try {
        const [geographic, unaligned] = await Promise.all([
          fetchCatalogBounds(bounds, controller.signal),
          fetchUnaligned(controller.signal),
        ])
        if (!controller.signal.aborted) {
          setState({ geographic, unaligned, loading: false, error: null })
        }
      } catch (error) {
        if (!controller.signal.aborted) {
          setState(current => ({
            ...current,
            loading: false,
            error: error instanceof Error ? error.message : 'Catalog query failed',
          }))
        }
      }
    }, 150)
    return () => {
      window.clearTimeout(timer)
      controller.abort()
    }
  }, [bounds])

  return state
}
