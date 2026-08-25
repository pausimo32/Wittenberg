/**
 * The only place this app talks to anything. Same-origin, relative URLs, no configurable
 * base: `docs/adr/0004` and `docs/05` §3 make one outbound call to anywhere else a bug, and
 * a base-URL option is how that bug arrives — someone points it at a laptop on the network
 * "just for a demo" and the app is no longer local.
 *
 * There is no retry and no timeout wrapper either. The server is on the same machine; if it
 * does not answer, saying so immediately is more useful than waiting.
 */

import type { Health, Prediction } from './types'

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    /** The machine-readable half of `detail`. Empty when the server sent prose only. */
    readonly code: string = '',
    /** Whatever else the refusal carried — `existing_rows`, for one. */
    readonly info: Record<string, unknown> = {},
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

/** Turn a failed response into an `ApiError`. Exported: `drafts.ts` needs the same parsing,
 *  and two copies of it is exactly the duplication this project keeps refusing. */
export async function apiFailure(response: Response): Promise<ApiError> {
  // FastAPI puts the reason in `detail`; anything else, fall back to the status text so the
  // caller never has to render "undefined".
  let detail = response.statusText
  let code = ''
  let info: Record<string, unknown> = {}
  try {
    const body = (await response.json()) as { detail?: unknown }
    if (typeof body.detail === 'string') {
      detail = body.detail
    } else if (body.detail && typeof body.detail === 'object') {
      // §9's routes answer `{code, message, …}` so the Catalan can be chosen in `strings.ts`
      // rather than written in Python.
      const d = body.detail as Record<string, unknown>
      if (typeof d.message === 'string') detail = d.message
      if (typeof d.code === 'string') code = d.code
      info = d
    }
  } catch {
    /* a non-JSON error body is not itself an error worth reporting */
  }
  return new ApiError(detail, response.status, code, info)
}

export async function getHealth(signal?: AbortSignal): Promise<Health> {
  const response = await fetch('/api/health', { signal })
  if (!response.ok) throw await apiFailure(response)
  return (await response.json()) as Health
}

export interface PredictOptions {
  /** `false` asks for the final edge map alone — see `docs/05` §3. */
  stages?: boolean
  signal?: AbortSignal
}

export async function predict(
  file: File,
  { stages = true, signal }: PredictOptions = {},
): Promise<Prediction> {
  const form = new FormData()
  form.append('file', file, file.name)

  // ONE request returns the prediction and every stage (§3, §8). Never call this twice —
  // once "to show" and once "to predict" — or the viewer displays images that did not
  // produce the answer beside them.
  const response = await fetch(`/api/predict?stages=${stages}`, {
    method: 'POST',
    body: form,
    signal,
  })
  if (!response.ok) throw await apiFailure(response)
  return (await response.json()) as Prediction
}

/** A stage's PNG as a data URL. The bytes never touch the network again. */
export function stageImageSrc(pngBase64: string): string {
  return `data:image/png;base64,${pngBase64}`
}
