/**
 * `docs/05-webapp-spec.md` §6: "Unit (vitest): result-tier logic, the reset behaviour, the
 * confidence formatting".
 *
 * The boundaries are asserted exactly at 0.70 and 0.40 and just below them, because §4
 * writes them as `>=` and an off-by-one-hundredth here changes what a participant is told
 * about how much the machine knows.
 */

import { describe, expect, it } from 'vitest'
import type { Candidate } from '../api/types'
import {
  formatFrequency,
  formatMs,
  formatProbability,
  tierOf,
  verdictOf,
} from './confidence'

const c = (freq_hz: number, p: number, class_index = freq_hz): Candidate => ({
  class_index,
  freq_hz,
  p,
})

describe('tierOf — the §4 thresholds, on the boundary', () => {
  it('is high at exactly 0.70 and above', () => {
    expect(tierOf(0.7)).toBe('high')
    expect(tierOf(0.71)).toBe('high')
    expect(tierOf(1)).toBe('high')
  })

  it('drops to mid just below 0.70', () => {
    expect(tierOf(0.6999)).toBe('mid')
  })

  it('is mid at exactly 0.40 and above', () => {
    expect(tierOf(0.4)).toBe('mid')
    expect(tierOf(0.55)).toBe('mid')
  })

  it('drops to low just below 0.40', () => {
    expect(tierOf(0.3999)).toBe('low')
    expect(tierOf(0)).toBe('low')
  })
})

describe('verdictOf', () => {
  it('names one frequency at the high tier and still shows all three', () => {
    const v = verdictOf([c(740, 0.82), c(610, 0.11), c(860, 0.07)])
    expect(v.tier).toBe('high')
    expect(v.leader?.freq_hz).toBe(740)
    expect(v.runnerUp).toBeNull()
    expect(v.candidates).toHaveLength(3)
  })

  it('names two at the mid tier', () => {
    const v = verdictOf([c(740, 0.45), c(610, 0.41), c(860, 0.14)])
    expect(v.tier).toBe('mid')
    expect(v.leader?.freq_hz).toBe(740)
    expect(v.runnerUp?.freq_hz).toBe(610)
  })

  it('names nobody at the low tier — a hedged guess is still a guess', () => {
    const v = verdictOf([c(740, 0.34), c(610, 0.33), c(860, 0.33)])
    expect(v.tier).toBe('low')
    expect(v.leader).toBeNull()
    expect(v.runnerUp).toBeNull()
    // §4: never hide low confidence. The numbers are still all there.
    expect(v.candidates).toHaveLength(3)
  })

  it('sorts by probability, whatever order the server sent', () => {
    const v = verdictOf([c(610, 0.11), c(740, 0.82), c(860, 0.07)])
    expect(v.candidates.map((x) => x.freq_hz)).toEqual([740, 610, 860])
  })

  it('treats no candidates as low rather than throwing', () => {
    const v = verdictOf([])
    expect(v.tier).toBe('low')
    expect(v.leader).toBeNull()
    expect(v.candidates).toEqual([])
  })

  it('does not mutate the array it was given', () => {
    const top3 = [c(610, 0.11), c(740, 0.82), c(860, 0.07)]
    verdictOf(top3)
    expect(top3.map((x) => x.freq_hz)).toEqual([610, 740, 860])
  })
})

describe('formatting — §4 bans rounding to a nicer number', () => {
  it('truncates a probability instead of rounding up', () => {
    // 0.6999 is the mid tier. Reading "70.0 %" beside "Podria ser" would make the headline
    // and the number disagree, which is the whole failure §4 is written to prevent.
    expect(formatProbability(0.6999)).toBe('69.9 %')
    expect(formatProbability(0.8249)).toBe('82.4 %')
    expect(formatProbability(1)).toBe('100.0 %')
    expect(formatProbability(0)).toBe('0.0 %')
  })

  it('shows a frequency exactly as the class list stores it', () => {
    expect(formatFrequency(1540)).toBe('1540 Hz')
    expect(formatFrequency(420)).toBe('420 Hz')
  })

  it('shows sub-millisecond stage times without collapsing them to zero', () => {
    // Canny is ≈ 0.35 ms; "0 ms" would read as "not measured".
    expect(formatMs(0.348)).toBe('0.35 ms')
    expect(formatMs(19.35)).toBe('19.4 ms')
  })

  it('shows an untimed stage as a dash, never as a plausible number', () => {
    expect(formatMs(null)).toBe('—')
  })
})
