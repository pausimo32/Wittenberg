/**
 * The queue behind step 4 of `/dades`. Two properties, both of them about six hundred files
 * rather than six: the server is never given more work than it was told it could take, and one
 * unreadable photo in a folder does not cost the rest of the folder.
 */

import { describe, expect, it } from 'vitest'
import { uploadQueue } from './upload'

describe('uploadQueue', () => {
  it('never runs more than `concurrency` at once', async () => {
    let running = 0
    let peak = 0
    const items = Array.from({ length: 20 }, (_, i) => i)

    await uploadQueue(
      items,
      async (n) => {
        running += 1
        peak = Math.max(peak, running)
        await new Promise((r) => setTimeout(r, 1))
        running -= 1
        return n
      },
      { concurrency: 3 },
    )

    expect(peak).toBe(3)
  })

  it('reports every result as it arrives, and keeps going after a failure', async () => {
    const seen: string[] = []

    await uploadQueue(
      ['a', 'b', 'c'],
      async (item) => {
        if (item === 'b') throw new Error('nope')
        return item
      },
      {
        concurrency: 1,
        onResult: (item, result, error) => seen.push(`${item}:${result ?? error?.message}`),
      },
    )

    expect(seen).toEqual(['a:a', 'b:nope', 'c:c'])
  })
})
