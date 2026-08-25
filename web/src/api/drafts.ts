/**
 * `/dades`'s calls. Same-origin and relative, like `client.ts` — see its header for why there
 * is no base URL and no retry.
 *
 * The one rule specific to this file: a `File` that arrives here goes into the `FormData`
 * untouched. No canvas, no `createImageBitmap`, no compression "to make the upload quicker".
 * `data/raw/` is defined as untouched, straight off the phone (`docs/02` §4), and a recoded
 * JPEG cannot be told from an original once it is stored. `drafts.test.ts` asserts the bytes.
 */

import { apiFailure } from './client'
import type {
  CalibrationInfo,
  CommitStatus,
  Draft,
  DraftSummary,
  FrameRecord,
  Matrix,
  SplitResult,
} from './types'

async function json<T>(response: Response): Promise<T> {
  if (!response.ok) throw await apiFailure(response)
  return (await response.json()) as T
}

async function nothing(response: Response): Promise<void> {
  if (!response.ok) throw await apiFailure(response)
}

export async function listDrafts(signal?: AbortSignal): Promise<DraftSummary[]> {
  const body = await json<{ drafts: DraftSummary[] }>(await fetch('/api/drafts', { signal }))
  return body.drafts
}

export async function createDraft(session: Record<string, unknown>): Promise<Draft> {
  return json<Draft>(
    await fetch('/api/drafts', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(session),
    }),
  )
}

export async function getDraft(id: string, signal?: AbortSignal): Promise<Draft> {
  return json<Draft>(await fetch(`/api/drafts/${id}`, { signal }))
}

export async function discardDraft(id: string): Promise<void> {
  return nothing(await fetch(`/api/drafts/${id}`, { method: 'DELETE' }))
}

export async function uploadCalibration(
  id: string,
  file: File,
): Promise<{ calibration: CalibrationInfo; preview: string }> {
  const form = new FormData()
  form.append('file', file, file.name)
  return json(await fetch(`/api/drafts/${id}/calibration`, { method: 'POST', body: form }))
}

export async function uploadFrame(id: string, freqHz: number, file: File): Promise<FrameRecord> {
  const form = new FormData()
  form.append('file', file, file.name) // the File itself. Never a re-encoded copy.

  // The class is a query parameter, not a form field: it labels the request, and keeping the
  // body to exactly one part means there is nothing in it that could be mistaken for content.
  return json<FrameRecord>(
    await fetch(`/api/drafts/${id}/frames?freq_hz=${freqHz}`, { method: 'POST', body: form }),
  )
}

export async function deleteFrame(id: string, freqHz: number, name: string): Promise<void> {
  return nothing(await fetch(`/api/drafts/${id}/frames/${freqHz}/${name}`, { method: 'DELETE' }))
}

export async function startCommit(
  id: string,
  replace = false,
  jobs?: number,
): Promise<CommitStatus> {
  const query = new URLSearchParams({ replace: String(replace) })
  if (jobs !== undefined) query.set('jobs', String(jobs))
  return json<CommitStatus>(await fetch(`/api/drafts/${id}/commit?${query}`, { method: 'POST' }))
}

export async function getCommit(id: string, signal?: AbortSignal): Promise<CommitStatus> {
  return json<CommitStatus>(await fetch(`/api/drafts/${id}/commit`, { signal }))
}

export async function runSplit(): Promise<SplitResult> {
  return json<SplitResult>(await fetch('/api/dataset/split', { method: 'POST' }))
}

export async function getMatrix(signal?: AbortSignal): Promise<Matrix> {
  return json<Matrix>(await fetch('/api/dataset/matrix', { signal }))
}

/** The stored preview of one frame. `original` is the capture, `edges` the Canny result. */
export function previewSrc(
  id: string,
  freqHz: number,
  name: string,
  kind: 'original' | 'edges',
): string {
  return `/api/drafts/${id}/preview/${freqHz}/${name}?kind=${kind}`
}
