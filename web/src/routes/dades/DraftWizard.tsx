/**
 * One draft, from the session form to the moment it becomes rows in `data/metadata.csv`.
 * `docs/05-webapp-spec.md` §9.
 *
 * The draft is loaded from the server by the id in the URL and held here, in one place, with
 * a `reload` the step components call after anything that changes it. That is the whole point
 * of the address `/dades/:draftId` existing at all: the state of the afternoon lives in
 * `drafts/<session_id>/`, so a reload, a second tab or a pasted link all arrive at the same
 * session, and this component is only the window onto it. Nothing about the draft is kept
 * here that the server does not already hold.
 *
 * The four headings below are the shape of the flow, always visible, so the order is legible
 * before it is enforced — and the order *is* enforced, by the server: `POST /frames` answers
 * 409 `not_calibrated` until the four markers have been found. Tasks 15 to 17 hang the step
 * components off `draft` and `reload`.
 */

import { useCallback, useEffect, useState } from 'react'
import { useParams } from 'react-router'
import type { Draft } from '../../api/types'
import { ApiError } from '../../api/client'
import { getDraft } from '../../api/drafts'
import { t } from '../../strings'

/** The four steps, in order, with the state each one is waiting on. */
const STEPS = [
  { key: 'session', label: t.data.steps.session },
  { key: 'calibration', label: t.data.steps.calibration },
  { key: 'frames', label: t.data.steps.frames },
  { key: 'review', label: t.data.steps.review },
] as const

/**
 * Which step a draft is sitting at, read off the draft rather than remembered.
 *
 * Step 1 is done the moment a draft exists — a draft *is* an accepted session log. After that
 * the answer is the server's: no calibration means step 2, no frames means step 3, and frames
 * on disk mean there is something to review.
 */
function currentStep(draft: Draft): number {
  if (!draft.calibrated) return 1
  return draft.total > 0 ? 3 : 2
}

export function DraftWizard() {
  const { draftId } = useParams()
  const [draft, setDraft] = useState<Draft | null>(null)
  const [error, setError] = useState<string | null>(null)

  /** Re-read the draft from the server. The steps call this; they never patch it locally. */
  const reload = useCallback(
    async (signal?: AbortSignal) => {
      if (draftId === undefined) return
      try {
        setDraft(await getDraft(draftId, signal))
        setError(null)
      } catch (failure) {
        if (signal?.aborted) return
        // A 404 here is a reachable, ordinary case: the draft was discarded or committed in
        // another tab and this link is now stale. Reporting it as a network problem would
        // send someone to look at the server for a server that answered perfectly.
        const missing = failure instanceof ApiError && failure.status === 404
        setError(missing ? t.data.error.notFound : t.data.error.load)
      }
    },
    [draftId],
  )

  // Same shape as `App.tsx`'s `/health` fetch: one request, an `AbortController`, aborted in
  // the teardown so a view left before the server answers never settles behind the user.
  useEffect(() => {
    const controller = new AbortController()
    void reload(controller.signal)
    return () => controller.abort()
  }, [reload])

  const step = draft === null ? -1 : currentStep(draft)

  return (
    <div className="mx-auto max-w-3xl px-4 py-12 sm:px-6">
      <h1 className="font-mono text-3xl font-semibold">{draftId}</h1>
      {draft !== null && (
        <p className="mt-2 text-sm text-muted">{t.data.drafts.statuses[draft.status]}</p>
      )}

      {error !== null && (
        <div
          role="alert"
          className="mt-4 rounded-lg border border-warn-border bg-warn-surface p-4 text-warn-ink"
        >
          <p>{error}</p>
          <button
            type="button"
            onClick={() => void reload()}
            className="mt-3 rounded-lg border border-warn-ink px-4 py-2 font-medium"
          >
            {t.common.retry}
          </button>
        </div>
      )}

      {draft === null && error === null && (
        <p role="status" className="mt-6">
          {t.common.loading}
        </p>
      )}

      {draft !== null && draft.status === 'committing' && (
        <p
          role="alert"
          className="mt-4 rounded-lg border border-warn-border bg-warn-surface p-4 text-warn-ink"
        >
          {t.data.drafts.interrupted}
        </p>
      )}

      <ol className="mt-8 space-y-4">
        {STEPS.map((entry, index) => (
          <li
            key={entry.key}
            aria-current={index === step ? 'step' : undefined}
            className={
              index === step
                ? 'rounded-xl border border-primary bg-surface p-5'
                : 'rounded-xl border border-border p-5'
            }
          >
            <h2 className="text-xl font-semibold">{entry.label}</h2>
          </li>
        ))}
      </ol>
    </div>
  )
}
