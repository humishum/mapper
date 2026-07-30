import { describe, expect, it } from 'vitest'
import { BoundedAbortableCache, DEFAULT_POINT_BUDGET, allocatePointBudget } from './budget'

describe('detail resource budgets', () => {
  it('shares one global budget across two visible artifacts', () => {
    expect(allocatePointBudget(['a', 'b'], new Set(['a', 'b']))).toEqual({
      a: DEFAULT_POINT_BUDGET / 2,
      b: DEFAULT_POINT_BUDGET / 2,
    })
    expect(allocatePointBudget(['a', 'b'], new Set(['b']))).toEqual({ a: 0, b: DEFAULT_POINT_BUDGET })
  })

  it('evicts oldest node attributes by byte and entry bounds', () => {
    const cache = new BoundedAbortableCache<number>(10, 2)
    cache.set('a', 1, 6)
    cache.set('b', 2, 6)
    expect(cache.get('a')).toBeUndefined()
    expect(cache.get('b')).toBe(2)
    cache.clear()
    expect(cache.sizeBytes).toBe(0)
  })
})
