import { GlobalCache } from '@giro3d/giro3d/core/Cache'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import { GEOMETRY_POOL_BYTES } from './core/budget'
import './index.css'

// Giro3D's singleton cache defaults to 512 MiB. Configure it before any
// renderer or source exists so COPC chunks share the Phase 2 CPU pool limit.
if (GlobalCache.count === 0 && GlobalCache.maxSize !== GEOMETRY_POOL_BYTES) {
  GlobalCache.configure({ byteCapacity: GEOMETRY_POOL_BYTES })
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
