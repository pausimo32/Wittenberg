/**
 * A bounded queue for the one place this app uploads in bulk.
 *
 * Ten classes at sixty frames is six hundred requests, each of which costs the server a
 * ~560 ms HEIC decode. Firing them all at once buries the process; firing them one at a time
 * wastes the wait. Three at a time keeps a verdict arriving every few hundred milliseconds —
 * which is what "you find out that the lighting is wrong on photo three" actually requires —
 * while leaving the tab responsive.
 *
 * A failure is a result, not an end: the eleventh photo being unreadable must not stop the
 * other five hundred and eighty-nine.
 */

export interface QueueOptions<T, R> {
  concurrency?: number
  onResult?: (item: T, result: R | null, error: Error | null) => void
  signal?: AbortSignal
}

export async function uploadQueue<T, R>(
  items: readonly T[],
  worker: (item: T) => Promise<R>,
  { concurrency = 3, onResult, signal }: QueueOptions<T, R> = {},
): Promise<void> {
  let next = 0

  async function drain(): Promise<void> {
    while (next < items.length) {
      if (signal?.aborted) return
      // The index was just bounds-checked; `noUncheckedIndexedAccess` cannot see that, and
      // widening the item to `T | undefined` here would push the lie into `worker`'s signature.
      const item = items[next++] as T
      try {
        const result = await worker(item)
        onResult?.(item, result, null)
      } catch (error) {
        onResult?.(item, null, error instanceof Error ? error : new Error(String(error)))
      }
    }
  }

  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, drain))
}
