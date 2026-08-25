/**
 * The three-tier result presentation of `docs/05-webapp-spec.md` §4 — "still the most
 * important UI decision in the project".
 *
 * Pure functions with no React and no strings baked in, so the thresholds can be tested on
 * fixed inputs without rendering anything. The rules, verbatim from §4:
 *
 * | max p          | behaviour                                                    |
 * |----------------|--------------------------------------------------------------|
 * | >= 0.70        | "Probablement N Hz" + the other two candidates                |
 * | >= 0.40, < 0.70| "Podria ser N Hz o M Hz" — both shown equally                 |
 * | < 0.40         | "No ho sé prou" + guidance                                    |
 *
 * Two prohibitions that are the point of the whole section, and are enforced here rather
 * than left to whoever writes the component: a probability is **never rounded to a nicer
 * number**, and a low confidence is **never hidden**. A tool that always answers a
 * confident number teaches participants that machines are always right, which is the
 * opposite of what this project is for.
 */

import type { Candidate } from '../api/types'

export const TIER_HIGH = 0.7
export const TIER_MID = 0.4

export type Tier = 'high' | 'mid' | 'low'

export interface Verdict {
  tier: Tier
  /** Top-3, always all three, always in descending probability — every tier shows them. */
  candidates: Candidate[]
  /** The one to name in the headline, or null at the low tier where naming one would lie. */
  leader: Candidate | null
  /** The second name at the mid tier, where two candidates are shown as equals. */
  runnerUp: Candidate | null
}

export function tierOf(p: number): Tier {
  if (p >= TIER_HIGH) return 'high'
  if (p >= TIER_MID) return 'mid'
  return 'low'
}

export function verdictOf(top3: readonly Candidate[]): Verdict {
  const candidates = [...top3].sort((a, b) => b.p - a.p)
  const best = candidates[0]

  // No prediction at all is the low tier, not a crash and not an empty confident box.
  if (best === undefined) {
    return { tier: 'low', candidates: [], leader: null, runnerUp: null }
  }

  const tier = tierOf(best.p)
  return {
    tier,
    candidates,
    // At the low tier the headline names nobody. Showing "probably N Hz" under a 0.3 and
    // relying on small print to walk it back is exactly the hidden low confidence §4 bans.
    leader: tier === 'low' ? null : best,
    runnerUp: tier === 'mid' ? (candidates[1] ?? null) : null,
  }
}

/**
 * A probability as a percentage for display. One decimal, truncated toward zero, never
 * rounded up: 0.6999 must not read as "70 %" next to a headline that says the tier is
 * "podria ser". The reader can always see the tier and the number agree.
 */
export function formatProbability(p: number): string {
  const pct = Math.floor(p * 1000) / 10
  return `${pct.toFixed(1)} %`
}

/**
 * A frequency for display. Integers, exactly as the class list stores them — never rounded
 * to a "nicer" number (§4). 1540 Hz is 1540 Hz, not "about 1500".
 */
export function formatFrequency(hz: number): string {
  return `${hz} Hz`
}

/** Milliseconds as the viewer shows them beside a stage. Real measured values only. */
export function formatMs(ms: number | null): string {
  if (ms === null) return '—'
  if (ms < 1) return `${ms.toFixed(2)} ms`
  return `${ms.toFixed(1)} ms`
}
