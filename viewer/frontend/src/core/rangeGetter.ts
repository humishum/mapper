import type { Getter } from 'copc'

/**
 * COPC byte-range getter whose in-flight fetches can be cancelled as a group.
 * Giro3D accepts this instead of a URL, which gives renderer disposal direct
 * ownership of hierarchy, node-geometry, and picked-attribute requests.
 */
export class AbortableRangeGetter {
  private readonly controllers = new Set<AbortController>()
  private aborted = false

  constructor(private readonly url: string) {}

  readonly get: Getter = async (begin, end) => {
    if (this.aborted) throw new DOMException('COPC source disposed', 'AbortError')
    const controller = new AbortController()
    this.controllers.add(controller)
    try {
      const response = await fetch(this.url, {
        signal: controller.signal,
        headers: { Range: `bytes=${begin}-${end - 1}` },
      })
      if (response.status !== 200 && response.status !== 206) {
        throw new Error(`COPC range request failed: ${response.status} ${response.statusText}`)
      }
      const bytes = new Uint8Array(await response.arrayBuffer())
      if (response.status === 206) return bytes
      if (begin === 0 && bytes.byteLength === end) return bytes
      if (bytes.byteLength < end) {
        throw new Error('COPC server ignored Range and returned an undersized response')
      }
      return bytes.slice(begin, end)
    } finally {
      this.controllers.delete(controller)
    }
  }

  abort(): void {
    if (this.aborted) return
    this.aborted = true
    for (const controller of this.controllers) controller.abort()
    this.controllers.clear()
  }
}
