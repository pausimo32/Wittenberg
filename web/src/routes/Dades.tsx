/**
 * `/dades` — the team's labelling surface, §9. Never shown in a field session.
 *
 * This view is the list of sessions in progress, and it exists because a draft is **server**
 * state: `drafts/<session_id>/` on disk, not React state in a tab. That is what makes a
 * reload, a second tab or a pasted link land back on the same afternoon's work, and it is
 * also what makes an abandoned draft possible — so something has to show them, name where
 * they are, and let one be thrown away deliberately.
 *
 * Three things on each row earn their place:
 *
 * - the **path**, printed in full. An orphan draft in a directory nobody can find is an
 *   afternoon that looks lost even when every byte of it is still there.
 * - the **size**. Three open drafts are a few gigabytes of 12 MP captures, and the person
 *   who has to delete one needs to know which one that should be.
 * - the **counts**, in the three categories the server keeps — accepted, rejected and
 *   duplicated. Duplicates are neither of the other two and collapsing them into "rejected"
 *   would hide a retried batch, which is the one thing that unbalances classes silently.
 *
 * The discard confirmation is an in-page panel and not `window.confirm`. `confirm` cannot be
 * asserted from a test, and — the real reason — it cannot carry the sentence the confirmation
 * exists for: this deletes the uploads, and the dataset is not touched, because nothing of
 * this draft was ever written into it.
 */

import { useEffect, useState } from 'react'
import { Link } from 'react-router'
import type { DraftSummary } from '../api/types'
import { discardDraft, listDrafts } from '../api/drafts'
import { t } from '../strings'

export function Dades() {
  const [drafts, setDrafts] = useState<DraftSummary[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  /** The draft whose confirmation panel is open, if any. One at a time, by construction. */
  const [confirming, setConfirming] = useState<string | null>(null)

  // The `/health` pattern from `App.tsx`: one request, an `AbortController`, aborted in the
  // effect's teardown so a view left before the server answers does not settle afterwards.
  useEffect(() => {
    const controller = new AbortController()
    listDrafts(controller.signal)
      .then(setDrafts)
      .catch(() => {
        if (controller.signal.aborted) return
        setError(t.data.error.list)
        setDrafts([])
      })
    return () => controller.abort()
  }, [])

  async function discard(id: string) {
    try {
      await discardDraft(id)
      // The 204 is the confirmation; the row goes from the list we already hold rather than
      // from a second `GET`, so the answer on screen is the one the server just gave.
      setDrafts((current) => (current ?? []).filter((draft) => draft.draft_id !== id))
      setConfirming(null)
    } catch {
      setError(t.data.error.discard)
    }
  }

  return (
    <div className="mx-auto max-w-3xl px-4 py-12 sm:px-6">
      <h1 className="text-3xl font-semibold">{t.data.title}</h1>
      <p className="mt-4 rounded-lg border border-warn-border bg-warn-surface p-4 text-warn-ink">
        {t.data.warning}
      </p>
      <p className="mt-4 text-muted">{t.data.lead}</p>

      <Link
        to="/dades/nova"
        className="mt-6 inline-block rounded-lg bg-primary px-5 py-2.5 font-medium text-white hover:bg-primary-strong"
      >
        {t.data.newSession}
      </Link>

      <h2 className="mt-10 text-2xl font-semibold">{t.data.drafts.title}</h2>
      <p className="mt-2 text-sm text-muted">{t.data.drafts.untouched}</p>

      {error && (
        <p role="alert" className="mt-4 rounded-lg border border-warn-border bg-warn-surface p-4 text-warn-ink">
          {error}
        </p>
      )}

      {drafts === null && (
        <p role="status" className="mt-6">
          {t.common.loading}
        </p>
      )}

      {drafts !== null && drafts.length === 0 && <p className="mt-6">{t.data.drafts.none}</p>}

      {drafts !== null && drafts.length > 0 && (
        <ul className="mt-6 space-y-4">
          {drafts.map((draft) => (
            <li key={draft.draft_id} className="rounded-xl border border-border bg-white p-5">
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <h3 className="font-mono text-lg font-semibold">{draft.draft_id}</h3>
                <time dateTime={draft.updated} className="text-sm text-muted">
                  {t.data.drafts.updated}: {draft.updated.replace('T', ' ')}
                </time>
              </div>

              <p className="mt-2">
                {t.data.drafts.counts(draft.accepted, draft.rejected, draft.duplicates)}
              </p>
              {draft.pending > 0 && (
                <p className="mt-1 text-sm text-muted">{t.data.drafts.pending(draft.pending)}</p>
              )}
              <p className="mt-1 text-sm text-muted">{t.data.drafts.size(draft.bytes)}</p>

              <p className="mt-3 text-sm text-muted">{t.data.drafts.path}</p>
              <p className="font-mono text-sm break-all">{draft.path}</p>

              {draft.status === 'committing' && (
                <p
                  role="alert"
                  className="mt-3 rounded-lg border border-warn-border bg-warn-surface p-3 text-sm text-warn-ink"
                >
                  {t.data.drafts.interrupted}
                </p>
              )}

              <div className="mt-4 flex flex-wrap items-center gap-3">
                <Link
                  to={`/dades/${draft.draft_id}`}
                  className="rounded-lg bg-primary px-4 py-2 font-medium text-white hover:bg-primary-strong"
                >
                  {t.data.drafts.open}
                </Link>

                {/* The row's own button disappears while its panel is open, so "Descarta"
                    always names exactly one control on the page. */}
                {confirming !== draft.draft_id && (
                  <button
                    type="button"
                    onClick={() => setConfirming(draft.draft_id)}
                    className="rounded-lg border border-border px-4 py-2 font-medium hover:bg-surface"
                  >
                    {t.data.drafts.discard}
                  </button>
                )}
              </div>

              {confirming === draft.draft_id && (
                <div
                  role="alertdialog"
                  aria-label={t.data.drafts.discard}
                  className="mt-4 rounded-lg border border-warn-border bg-warn-surface p-4 text-warn-ink"
                >
                  <p>{t.data.drafts.discardConfirm(draft.draft_id)}</p>
                  <div className="mt-3 flex flex-wrap gap-3">
                    <button
                      type="button"
                      onClick={() => void discard(draft.draft_id)}
                      className="rounded-lg bg-warn-ink px-4 py-2 font-medium text-white"
                    >
                      {t.data.drafts.discard}
                    </button>
                    <button
                      type="button"
                      onClick={() => setConfirming(null)}
                      className="rounded-lg border border-warn-ink px-4 py-2 font-medium"
                    >
                      {t.common.cancel}
                    </button>
                  </div>
                </div>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
