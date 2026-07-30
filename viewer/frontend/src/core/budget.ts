export const DEFAULT_POINT_BUDGET = 2_000_000
export const GEOMETRY_POOL_BYTES = 256 * 1024 * 1024
export const MAX_DETAIL_DISTANCE_METRES = 10_000

export function allocatePointBudget(
  artifactIds: readonly string[],
  visible: ReadonlySet<string>,
  total = DEFAULT_POINT_BUDGET,
): Record<string, number> {
  const active = artifactIds.filter(id => visible.has(id))
  if (!active.length) return Object.fromEntries(artifactIds.map(id => [id, 0]))
  const base = Math.floor(total / active.length)
  let remainder = total - base * active.length
  return Object.fromEntries(artifactIds.map(id => {
    if (!visible.has(id)) return [id, 0]
    const allocation = base + (remainder > 0 ? 1 : 0)
    remainder -= 1
    return [id, allocation]
  }))
}

export class BoundedAbortableCache<T> {
  private readonly entries = new Map<string, { value: T; bytes: number }>()
  private bytes = 0

  constructor(readonly maxBytes: number, readonly maxEntries = 32) {}

  get(key: string): T | undefined {
    const entry = this.entries.get(key)
    if (!entry) return undefined
    this.entries.delete(key)
    this.entries.set(key, entry)
    return entry.value
  }

  set(key: string, value: T, bytes: number): void {
    this.delete(key)
    this.entries.set(key, { value, bytes })
    this.bytes += bytes
    while (this.bytes > this.maxBytes || this.entries.size > this.maxEntries) {
      const oldest = this.entries.keys().next().value as string | undefined
      if (!oldest) break
      this.delete(oldest)
    }
  }

  delete(key: string): void {
    const entry = this.entries.get(key)
    if (entry) this.bytes -= entry.bytes
    this.entries.delete(key)
  }

  clear(): void {
    this.entries.clear()
    this.bytes = 0
  }

  get sizeBytes(): number {
    return this.bytes
  }
}
