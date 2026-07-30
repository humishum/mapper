import type { PerformanceState, SceneMode } from './types/contracts'

export interface BenchmarkEvent {
  name: string
  at: number
  detail?: Record<string, unknown>
}

export interface BenchmarkSnapshot {
  mode: SceneMode
  activeArtifactIds: string[]
  transitioning: boolean
  performance: PerformanceState
  rendererGeneration: number
  disposedRendererGenerations: number[]
}

interface BenchmarkBridge {
  snapshot: BenchmarkSnapshot
  events: BenchmarkEvent[]
}

const enabled = import.meta.env.VITE_BENCHMARK_ENABLED === 'true'
let rendererGeneration = 0
const disposedRendererGenerations: number[] = []

function bridge(): BenchmarkBridge | undefined {
  if (!enabled) return undefined
  window.__MAPPER_BENCHMARK__ ??= {
    snapshot: {
      mode: 'overview',
      activeArtifactIds: [],
      transitioning: false,
      performance: {
        visiblePoints: 0,
        pointBudget: 0,
        cpuGeometryBytes: 0,
        gpuGeometryBytes: 0,
        cpuGeometryLimitBytes: 0,
        gpuGeometryLimitBytes: 0,
      },
      rendererGeneration: 0,
      disposedRendererGenerations: [],
    },
    events: [],
  }
  return window.__MAPPER_BENCHMARK__
}

export function benchmarkEvent(name: string, detail?: Record<string, unknown>): void {
  const target = bridge()
  if (!target) return
  target.events.push({ name, at: performance.now(), detail })
  performance.mark(`mapper:${name}`, { detail })
}

export function benchmarkRendererCreated(mode: SceneMode): number {
  rendererGeneration += 1
  benchmarkEvent('renderer-created', { mode, generation: rendererGeneration })
  return rendererGeneration
}

export function benchmarkRendererDisposed(mode: SceneMode, generation: number): void {
  disposedRendererGenerations.push(generation)
  benchmarkEvent('renderer-disposed', { mode, generation })
}

export function updateBenchmarkSnapshot(
  value: Omit<BenchmarkSnapshot, 'rendererGeneration' | 'disposedRendererGenerations'>,
): void {
  const target = bridge()
  if (!target) return
  target.snapshot = {
    ...value,
    rendererGeneration,
    disposedRendererGenerations: [...disposedRendererGenerations],
  }
}
