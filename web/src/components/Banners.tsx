/**
 * The two things the app must say out loud, and their reason for being components rather
 * than sentences somewhere in a page.
 *
 * `DoNotReportBanner` — `docs/05` §3, §4. `/api/health` carries the checkpoint's
 * `do_not_report`, written by `src/train.py` into any run built from `SYNTH-*` sessions.
 * Such a run reaches ~100 % accuracy and that number measures how well the network inverted
 * the formula that drew its images. The app will serve it; what it must not do is let the
 * number look like a result. Shown on **every view**, because someone arriving straight at
 * `/analitza` never sees a home page.
 *
 * `GeometryWarning` — `docs/05` §4. Its own visible state, deliberately *not* folded into
 * the confidence tier: a low-confidence answer and a wrongly-rectified one are different
 * failures leading to different next actions. Ablation A10 (`docs/04` §5) will put a number
 * on the cost; until then the text says what happened, not how much it cost.
 */

import type { Prediction } from '../api/types'
import { t } from '../strings'

export function DoNotReportBanner({
  message,
  run,
}: {
  message: string | null
  run?: string
}) {
  if (!message) return null
  return (
    <div
      role="status"
      className="border-b border-warn-border bg-warn-surface text-warn-ink"
    >
      <div className="mx-auto flex max-w-6xl flex-wrap items-baseline gap-x-3 gap-y-1 px-4 py-2.5 sm:px-6">
        <strong className="font-semibold">{t.banner.doNotReport}</strong>
        <span className="text-sm">{message}</span>
        {run && (
          <span className="ml-auto font-mono text-xs opacity-80">
            {t.banner.run} {run}
          </span>
        )}
      </div>
    </div>
  )
}

export function GeometryWarning({ prediction }: { prediction: Prediction }) {
  if (prediction.roi_mode === 'homography') return null
  return (
    <aside
      role="status"
      aria-labelledby="geometry-heading"
      className="rounded-xl border border-warn-border bg-warn-surface p-5 text-warn-ink"
    >
      <h2 id="geometry-heading" className="font-semibold">
        {t.analyze.geometry.title}
      </h2>
      <p className="mt-2 max-w-prose text-[0.95rem]">{t.analyze.geometry.body}</p>
      <p className="mt-2 max-w-prose text-[0.95rem] font-medium">
        {t.analyze.geometry.action}
      </p>
      <p className="mt-3 text-sm opacity-80">
        {t.analyze.geometry.markers(prediction.markers_found)}
      </p>
    </aside>
  )
}
