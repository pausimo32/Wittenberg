/**
 * The result of a prediction, per `docs/05-webapp-spec.md` §4.
 *
 * The tier logic is in `lib/confidence.ts` and unit-tested there; this file is only how it
 * looks. Three things it does deliberately, all of them from §4:
 *
 * - **Every tier shows all three candidates with their probabilities.** The tier changes the
 *   headline, never how much is disclosed. There is no state of this component in which the
 *   reader is shown one number and not told how sure it is.
 * - **The low tier names nobody.** "No ho sé prou" plus what to try — not a greyed-out
 *   guess, which reads as a guess.
 * - **Nothing is rounded up.** `formatProbability` truncates for exactly this reason.
 */

import type { Candidate, Prediction } from '../api/types'
import { formatFrequency, formatProbability, verdictOf } from '../lib/confidence'
import { t } from '../strings'

export function Result({ prediction }: { prediction: Prediction }) {
  const verdict = verdictOf(prediction.top3)

  return (
    <section
      aria-labelledby="result-heading"
      className="card p-6 sm:p-8"
      data-tier={verdict.tier}
    >
      <h2 id="result-heading" className="sr-only">
        Resultat
      </h2>

      {verdict.tier === 'high' && verdict.leader && (
        <p className="text-2xl sm:text-3xl">
          {t.analyze.result.high}{' '}
          <strong className="text-4xl sm:text-5xl font-semibold text-secondary tabular-nums">
            {formatFrequency(verdict.leader.freq_hz)}
          </strong>
        </p>
      )}

      {verdict.tier === 'mid' && verdict.leader && verdict.runnerUp && (
        // Both shown equally — same size, same weight, same colour. The order is the
        // model's, but the presentation must not turn a 0.45/0.41 into a winner.
        <p className="text-2xl sm:text-3xl">
          {t.analyze.result.mid}{' '}
          <strong className="text-4xl sm:text-5xl font-semibold text-secondary tabular-nums">
            {formatFrequency(verdict.leader.freq_hz)}
          </strong>{' '}
          <span className="text-muted">{t.analyze.result.or}</span>{' '}
          <strong className="text-4xl sm:text-5xl font-semibold text-secondary tabular-nums">
            {formatFrequency(verdict.runnerUp.freq_hz)}
          </strong>
        </p>
      )}

      {verdict.tier === 'low' && (
        <div>
          <p className="text-2xl sm:text-3xl font-semibold text-secondary">
            {t.analyze.result.low}
          </p>
          <p className="mt-3 max-w-prose text-text/90">{t.analyze.result.lowHint}</p>
        </div>
      )}

      <h3 className="mt-8 mb-3 text-sm font-semibold uppercase tracking-wide text-muted">
        {t.analyze.result.alternatives}
      </h3>
      <ol className="space-y-2">
        {verdict.candidates.map((candidate, i) => (
          <ProbabilityBar
            key={candidate.class_index}
            candidate={candidate}
            emphasis={verdict.tier !== 'low' && i === 0}
          />
        ))}
      </ol>
      <p className="mt-4 text-sm text-muted">{t.analyze.result.distribution}</p>
    </section>
  )
}

function ProbabilityBar({
  candidate,
  emphasis,
}: {
  candidate: Candidate
  emphasis: boolean
}) {
  const percent = Math.max(0, Math.min(100, candidate.p * 100))
  return (
    <li className="flex items-center gap-3">
      <span className="w-24 shrink-0 text-right tabular-nums font-medium">
        {formatFrequency(candidate.freq_hz)}
      </span>
      <span
        className="relative h-6 flex-1 overflow-hidden rounded bg-surface"
        // The bar is decoration; the number beside it is the accessible value, so the
        // meter is hidden from the accessibility tree rather than duplicated into it.
        aria-hidden="true"
      >
        <span
          className={`absolute inset-y-0 left-0 ${emphasis ? 'bg-primary' : 'bg-muted/50'}`}
          style={{ width: `${percent}%` }}
        />
      </span>
      <span className="w-20 shrink-0 text-right tabular-nums text-sm">
        {formatProbability(candidate.p)}
      </span>
    </li>
  )
}
