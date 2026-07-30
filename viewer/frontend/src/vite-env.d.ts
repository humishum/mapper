/// <reference types="vite/client" />

import type { BenchmarkEvent, BenchmarkSnapshot } from './benchmark'

declare global {
  interface Window {
    __MAPPER_BENCHMARK__?: {
      snapshot: BenchmarkSnapshot
      events: BenchmarkEvent[]
    }
  }
}
