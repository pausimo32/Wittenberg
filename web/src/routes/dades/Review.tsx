/**
 * Steps 5 and 6 — the summary, the commit, and the split it forces.
 * `docs/05-webapp-spec.md` §9.
 *
 * **Nothing has been written yet, and the page says so first.** Everything above this point
 * lives in `drafts/<session_id>/`, outside `data/`; §9's "a half-finished session leaves no
 * trace in `data/`" is a property of that layout, not of anything this component does. What
 * this component does is make the one moment it stops being true a moment somebody chose.
 *
 * **A commit is confirmed, never performed twice by accident.** The server answers 409
 * `session_exists` with the row count when the session already has rows (requirement 4), and
 * the answer here is not a silent retry with `replace=true` — it is the count, in a sentence,
 * and a *different* button. A `mixed_provenance` refusal gets no button at all: §9 says a
 * `SYNTH-*` session cannot be mixed with real ones, so there is nothing to override, and a
 * button that answered 409 forever would be a worse lie than no button.
 *
 * **Progress, not a spinner.** Six hundred captures is minutes of ingest, and a spinner over
 * minutes is indistinguishable from a hang. `GET /commit` is polled with a `setTimeout`
 * chained off each answer — never `setInterval`, which on a slow answer stacks requests
 * behind one another and asks a busy server to also serve the queue that formed while it was
 * busy. The teardown clears the pending timer, so leaving the view ends the polling.
 *
 * **An interrupted commit is resumed, not restarted.** The captures are staged at
 * `data/raw/.incoming-<session_id>/` and no session exists yet; pressing commit again picks
 * them up from there. So the button stays pressable, because the alternative on offer would
 * be discarding the afternoon.
 *
 * **Rejected images are reported.** §9: they "are a reported result, not rubbish". The yield
 * and the reasons with their counts are the summary, alongside the duplicates — the number
 * that says a batch was uploaded twice, which no threshold anywhere else will ever trip.
 */

import { useEffect, useMemo, useState } from 'react'
import { ApiError } from '../../api/client'
import { getCommit, runSplit, startCommit } from '../../api/drafts'
import type { CommitStatus, Draft, FrameRecord, SplitResult } from '../../api/types'
import { t } from '../../strings'
import { Matrix } from './Matrix'

interface Props {
  draft: Draft
}

/** Half a second. Fast enough that the number looks live, slow enough that a commit of six
 *  hundred frames costs a couple of hundred requests to a server on the same machine. */
const POLL_MS = 500

/** `quality` values that mean the gate said no. `pending` is neither — it is stored-not-gated. */
function isRejected(record: FrameRecord): boolean {
  return record.quality !== 'ok' && record.quality !== 'pending'
}

/**
 * The Catalan for a gate reason, or the reason itself when there is none.
 *
 * The same fallback `Frames.tsx` makes, for the same reason: an unrecognised reason is a
 * string that needs adding to `strings.ts` and to `docs/07-glossary-ca-en.md`, and seeing the
 * server's own token on screen is what makes that happen. Inventing a plausible phrase here
 * would hide a new rejection reason behind it.
 */
function reasonText(reason: string): string {
  const reasons: Record<string, string> = t.data.frames.reasons
  return reasons[reason] ?? reason
}

/** What the commit is doing, said as the phase it is in rather than as a spinner. */
function progressText(status: CommitStatus): string {
  if (status.phase === 'stage') return t.data.review.staging
  if (status.phase === 'rename') return t.data.review.renaming
  return t.data.review.committing(status.done, status.total)
}

export function Review({ draft }: Props) {
  const [status, setStatus] = useState<CommitStatus | null>(null)
  const [refusal, setRefusal] = useState<ApiError | null>(null)
  const [busy, setBusy] = useState(false)
  /** Bumped when a commit is started, so the poll below restarts against the new job. */
  const [attempt, setAttempt] = useState(0)

  const [split, setSplit] = useState<SplitResult | null>(null)
  const [splitRefusal, setSplitRefusal] = useState<ApiError | null>(null)
  const [splitting, setSplitting] = useState(false)

  const id = draft.draft_id

  // One request on arrival, then one after each answer for as long as the answer says
  // `running`. The first one is not decoration: a reload during a commit, or a server that
  // was restarted mid-commit, are both found here and nowhere else — `GET /commit` is what
  // reports `interrupted`.
  useEffect(() => {
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout> | undefined

    async function poll(): Promise<void> {
      let answer: CommitStatus
      try {
        answer = await getCommit(id, controller.signal)
      } catch {
        // Leaving the view aborts the request; that is not a failure to report. Anything
        // else means the local server stopped answering, which is its own sentence.
        if (!controller.signal.aborted) setRefusal(new ApiError(t.analyze.error.network, 0))
        return
      }
      if (controller.signal.aborted) return
      setStatus(answer)
      // Chained, never `setInterval`: the next request is only scheduled once this one has
      // been answered, so a slow answer cannot stack a queue behind itself.
      if (answer.status === 'running') timer = setTimeout(() => void poll(), POLL_MS)
    }

    void poll()
    return () => {
      controller.abort()
      if (timer !== undefined) clearTimeout(timer)
    }
  }, [id, attempt])

  const running = status?.status === 'running'
  const committed = status?.status === 'done'

  async function commit(replace: boolean) {
    setRefusal(null)
    setBusy(true)
    try {
      // The 202 already carries a status, so the number on screen starts moving without
      // waiting half a second for the first poll.
      setStatus(await startCommit(id, replace))
      setAttempt((n) => n + 1)
    } catch (failure) {
      setRefusal(
        failure instanceof ApiError ? failure : new ApiError(t.analyze.error.network, 0),
      )
    } finally {
      setBusy(false)
    }
  }

  async function resplit() {
    setSplitRefusal(null)
    setSplitting(true)
    try {
      setSplit(await runSplit())
    } catch (failure) {
      setSplitRefusal(
        failure instanceof ApiError ? failure : new ApiError(t.analyze.error.network, 0),
      )
    } finally {
      setSplitting(false)
    }
  }

  /** The afternoon, counted: what was kept, what was refused and why, and what was a repeat. */
  const summary = useMemo(() => {
    const records = Object.values(draft.frames).flat()
    const byClass = Object.entries(draft.frames)
      .map(([freq, entries]) => ({
        freq: Number(freq),
        accepted: entries.filter((record) => record.quality === 'ok').length,
        total: entries.length,
      }))
      .sort((a, b) => a.freq - b.freq)

    const reasons = new Map<string, number>()
    for (const record of records.filter(isRejected)) {
      const reason = record.reason ?? record.quality
      reasons.set(reason, (reasons.get(reason) ?? 0) + 1)
    }

    const total = records.length
    const accepted = records.filter((record) => record.quality === 'ok').length
    return {
      total,
      accepted,
      // The yield §9 asks for. Zero captures has no yield rather than a yield of zero.
      percent: total === 0 ? null : Math.round((accepted / total) * 100),
      byClass,
      reasons: [...reasons.entries()].sort((a, b) => b[1] - a[1]),
    }
  }, [draft.frames])

  return (
    <div className="mt-4">
      <p className="text-muted">{t.data.review.lead}</p>

      <div className="mt-4 flex flex-wrap gap-8">
        <p className="flex flex-col">
          <span className="text-sm text-muted">{t.data.review.total}</span>
          <span className="font-mono text-2xl">{summary.total}</span>
        </p>
        <p className="flex flex-col">
          <span className="text-sm text-muted">{t.data.review.yield}</span>
          <span className="font-mono text-2xl">
            {summary.accepted} / {summary.total}
            {summary.percent !== null && ` · ${summary.percent} %`}
          </span>
        </p>
      </div>

      {/* The number that says a batch was uploaded twice. Beside the yield, because it is the
          other thing the totals do not explain by themselves. */}
      {draft.duplicates > 0 && (
        <p className="mt-3">{t.data.review.duplicatesNote(draft.duplicates)}</p>
      )}

      {summary.byClass.length > 0 && (
        <div className="mt-6">
          <h3 className="font-semibold">{t.data.review.byClass}</h3>
          <ul className="mt-2 flex flex-wrap gap-6">
            {summary.byClass.map((entry) => (
              <li key={entry.freq} className="flex flex-col">
                <span className="font-mono text-sm text-muted">{entry.freq} Hz</span>
                <span data-testid={`class-${entry.freq}`} className="font-mono text-lg">
                  {t.data.matrix.cell(entry.accepted, entry.total)}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* §9: rejected images "are kept and counted, and the yield appears in the summary.
          They are a reported result, not rubbish." A total that does not say why is a total
          nobody can act on. */}
      {summary.reasons.length > 0 && (
        <div className="mt-6">
          <h3 className="font-semibold">{t.data.frames.rejected}</h3>
          <ul className="mt-2 space-y-1">
            {summary.reasons.map(([reason, count]) => (
              <li key={reason} className="flex items-baseline gap-3">
                <span data-testid={`reason-${reason}`} className="font-mono text-lg">
                  {count}
                </span>
                <span>{reasonText(reason)}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* The staged captures are still there and no session exists yet; committing again
          resumes from them. Said here as well as in the wizard's header, because this is
          where the button that resumes it is. */}
      {status?.status === 'interrupted' && (
        <p
          role="alert"
          className="mt-6 rounded-lg border border-warn-border bg-warn-surface p-4 text-warn-ink"
        >
          {t.data.drafts.interrupted}
        </p>
      )}

      {status?.status === 'failed' && (
        <div
          role="alert"
          className="mt-6 rounded-lg border border-warn-border bg-warn-surface p-4 text-warn-ink"
        >
          <p className="font-medium">{t.data.review.failed}</p>
          {/* The server's own sentence, in English, for the log. Guessing at a Catalan one
              would be inventing a claim about a failure this view did not see. */}
          {status.error !== null && (
            <p className="mt-1 font-mono text-sm">{status.error.message}</p>
          )}
        </div>
      )}

      {refusal !== null && (
        <div
          role="alert"
          className="mt-6 rounded-lg border border-warn-border bg-warn-surface p-4 text-warn-ink"
        >
          {refusal.code === 'session_exists' ? (
            <>
              <p>{t.data.review.exists(draft.session_id, Number(refusal.info.existing_rows))}</p>
              {/* The override, and it is a second, explicit press. Retrying automatically
                  with `replace=true` would make requirement 4 a formality. */}
              <button
                type="button"
                onClick={() => void commit(true)}
                disabled={busy || running}
                className="mt-3 rounded-lg border border-warn-ink px-4 py-2 font-medium disabled:opacity-60"
              >
                {t.data.review.replace}
              </button>
            </>
          ) : refusal.code === 'mixed_provenance' ? (
            // No override button, deliberately: there is no override. §9 refuses a `SYNTH-*`
            // session beside a real one outright, and a button here would answer 409 forever.
            <p>{t.data.review.mixedProvenance}</p>
          ) : (
            <p>{refusal.message}</p>
          )}
        </div>
      )}

      {running && (
        <p role="status" className="mt-6 font-mono text-lg">
          {progressText(status)}
        </p>
      )}

      {!committed && (
        <button
          type="button"
          onClick={() => void commit(false)}
          disabled={busy || running}
          className="mt-6 rounded-lg bg-primary px-5 py-2.5 font-medium text-white hover:bg-primary-strong disabled:opacity-60"
        >
          {t.data.review.commit}
        </button>
      )}

      {committed && (
        <div className="mt-6 space-y-4">
          <p className="text-lg font-medium">{t.data.review.done}</p>

          {status.report !== null && (
            <div>
              <p className="font-mono">
                {t.data.matrix.cell(status.report.accepted, status.report.total)}
              </p>
              {Object.entries(status.report.by_reason).length > 0 && (
                <ul className="mt-2 space-y-1">
                  {Object.entries(status.report.by_reason).map(([reason, count]) => (
                    <li key={reason} className="flex items-baseline gap-3">
                      <span className="font-mono text-lg">{count}</span>
                      <span>{reasonText(reason)}</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}

          {/* §9: "Editing a label after the fact requires re-running the split, and the view
              says so rather than leaving the dataset quietly inconsistent." Before the button,
              because it is the sentence that explains why the button is there at all. */}
          <p className="rounded-lg border border-warn-border bg-warn-surface p-4 text-warn-ink">
            {t.data.review.relabelWarning}
          </p>

          <button
            type="button"
            onClick={() => void resplit()}
            disabled={splitting}
            className="rounded-lg bg-primary px-5 py-2.5 font-medium text-white hover:bg-primary-strong disabled:opacity-60"
          >
            {t.data.review.split}
          </button>

          {/* The split's own numbers, from the split's own answer. Never the draft's totals
              rearranged to look like them: the split is over the whole dataset, not over this
              session, and a plausible number computed here would be wrong every time. */}
          {split !== null && (
            <p role="status" className="font-mono">
              {t.data.review.splitDone(
                split.images.train ?? 0,
                split.images.val ?? 0,
                split.images.test ?? 0,
              )}
            </p>
          )}

          {splitRefusal !== null && (
            <p
              role="alert"
              className="rounded-lg border border-warn-border bg-warn-surface p-4 text-warn-ink"
            >
              {splitRefusal.code === 'split_failed'
                ? t.data.review.splitFailed
                : splitRefusal.message}
            </p>
          )}
        </div>
      )}

      <Matrix />
    </div>
  )
}
