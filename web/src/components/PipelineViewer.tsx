/**
 * The thirteen-stage pipeline viewer — `docs/05-webapp-spec.md` §8.
 *
 * **Everything here came from the one request that produced the answer.** Nothing is
 * recomputed for display and nothing is fetched again: the stages, their images, their
 * OpenCV parameters and their measured times are all fields of the `Prediction` already in
 * hand. If this component ever needs to call the API, something has gone wrong upstream.
 *
 * **Animated, not streamed, and the measurement is why.** The whole preprocessing chain is
 * ≈ 1.3 ms and the network ≈ 5–11 ms (measured — `docs/05` §8). Every stage would arrive
 * inside one animation frame, so a live stream would show the last image appearing and
 * nothing else. Instead one response is walked at a human pace, ~1.1 s per step, with
 * play/pause, arrows and a thumbnail rail.
 *
 * **The times shown are real.** Each stage carries its own measured `ms` from the server;
 * a stage nobody timed shows "—" rather than a plausible number. That is the point of
 * showing them at all: it turns "it's instant" from something the visitor has to take on
 * trust into a number on the screen.
 *
 * **The parameters are read, never retyped.** `stage.params` is `config/preprocess.json` as
 * the server resolved it. Change a Canny threshold and this panel says the new number.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import type { Prediction, Stage } from '../api/types'
import { stageImageSrc } from '../api/client'
import { formatFrequency, formatMs, formatProbability } from '../lib/confidence'
import { t } from '../strings'

/** §8: "~1.1 s per step". */
const STEP_MS = 1100

/** Stages 7 and 8 are 128 px; the feature sheets are smaller still. See index.css. */
const PIXELATED_KEYS = new Set(['resized', 'tensor', 'features_1', 'features_2', 'features_3'])

function groupOf(n: number): string {
  if (n <= 8) return t.viewer.groups.preparation
  if (n <= 11) return t.viewer.groups.network
  if (n === 12) return t.viewer.groups.attention
  return t.viewer.groups.decision
}

export function PipelineViewer({ prediction }: { prediction: Prediction }) {
  const stages = prediction.stages
  const [index, setIndex] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [compare, setCompare] = useState(false)
  const railRef = useRef<HTMLDivElement>(null)

  const total = stages.length
  const stage = stages[index]
  const original = stages.find((s) => s.n === 1)

  const step = useCallback(
    (delta: number) => {
      setPlaying(false)
      setIndex((i) => Math.min(total - 1, Math.max(0, i + delta)))
    },
    [total],
  )

  // Playback stops at the last stage rather than looping: the sequence is an argument with
  // an end — this is what the machine did — and looping it turns it into wallpaper.
  useEffect(() => {
    if (!playing) return
    if (index >= total - 1) {
      setPlaying(false)
      return
    }
    const timer = window.setTimeout(() => setIndex((i) => i + 1), STEP_MS)
    return () => window.clearTimeout(timer)
  }, [playing, index, total])

  // §8: "Keyboard: ← → to step, space to play/pause. Participants share one laptop."
  // Bound on the section rather than the window so the shortcuts belong to the viewer and
  // space does not hijack a button elsewhere on the page.
  const onKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === 'ArrowRight') {
      event.preventDefault()
      step(1)
    } else if (event.key === 'ArrowLeft') {
      event.preventDefault()
      step(-1)
    } else if (event.key === ' ' || event.key === 'Spacebar') {
      event.preventDefault()
      setPlaying((p) => !p)
    }
  }

  useEffect(() => {
    railRef.current
      ?.querySelector(`[data-thumb="${index}"]`)
      ?.scrollIntoView({ block: 'nearest', inline: 'center' })
  }, [index])

  if (!stage) return null

  return (
    <section
      aria-labelledby="viewer-heading"
      className="card p-5 sm:p-7"
      tabIndex={0}
      onKeyDown={onKeyDown}
    >
      <h2 id="viewer-heading" className="text-xl font-semibold">
        {t.viewer.title}
      </h2>
      <p className="mt-1 max-w-prose text-sm text-muted">{t.viewer.lead}</p>

      <div className="mt-5 grid gap-6 lg:grid-cols-[minmax(0,1fr)_20rem]">
        <div>
          <div className="flex flex-wrap items-baseline gap-x-3">
            <span className="text-xs font-semibold uppercase tracking-wide text-primary">
              {groupOf(stage.n)}
            </span>
            <h3 className="text-lg font-semibold">
              {stage.n}. {stage.title}
            </h3>
            <span className="ml-auto font-mono text-sm tabular-nums text-muted">
              {formatMs(stage.ms)}
            </span>
          </div>

          <div className="mt-3 grid gap-3" data-compare={compare}>
            <StageFigure stage={stage} />
            {compare && original && stage.n !== 1 && <StageFigure stage={original} muted />}
          </div>

          <StageDetails stage={stage} />
        </div>

        <div className="flex flex-col gap-4">
          <Controls
            index={index}
            total={total}
            playing={playing}
            compare={compare}
            onStep={step}
            onTogglePlay={() => setPlaying((p) => !p)}
            onToggleCompare={() => setCompare((c) => !c)}
          />
          <Rail
            ref={railRef}
            stages={stages}
            index={index}
            onPick={(i) => {
              setPlaying(false)
              setIndex(i)
            }}
          />
        </div>
      </div>
    </section>
  )
}

function StageFigure({ stage, muted = false }: { stage: Stage; muted?: boolean }) {
  if (stage.kind === 'bars') return <Distribution stage={stage} />
  if (!stage.png_base64) return null
  return (
    <figure>
      <img
        src={stageImageSrc(stage.png_base64)}
        alt={stage.title}
        className={`w-full rounded-lg border border-border bg-surface ${
          PIXELATED_KEYS.has(stage.key) ? 'pixelated' : ''
        }`}
      />
      {muted && (
        <figcaption className="mt-1 text-sm text-muted">{t.analyze.original}</figcaption>
      )}
    </figure>
  )
}

/** Stage 13 — a bar per class, sorted. The output is a distribution, not a verdict (§8). */
function Distribution({ stage }: { stage: Stage }) {
  const bars = [...(stage.bars ?? [])].sort((a, b) => b.p - a.p)
  return (
    <ol className="space-y-1.5">
      {bars.map((bar) => (
        <li key={bar.class_index} className="flex items-center gap-3">
          <span className="w-24 shrink-0 text-right tabular-nums text-sm">
            {formatFrequency(bar.freq_hz)}
          </span>
          <span className="relative h-5 flex-1 overflow-hidden rounded bg-surface" aria-hidden>
            <span
              className="absolute inset-y-0 left-0 bg-primary"
              style={{ width: `${Math.max(0, Math.min(100, bar.p * 100))}%` }}
            />
          </span>
          <span className="w-20 shrink-0 text-right tabular-nums text-sm">
            {formatProbability(bar.p)}
          </span>
        </li>
      ))}
    </ol>
  )
}

function StageDetails({ stage }: { stage: Stage }) {
  const [h, w] = [stage.shape[0], stage.shape[1]]
  const display = stage.display_shape
  return (
    <dl className="mt-4 grid gap-x-6 gap-y-2 text-sm sm:grid-cols-[auto_minmax(0,1fr)]">
      <dt className="font-semibold text-muted">{t.viewer.call}</dt>
      <dd className="font-mono break-all">{stage.call}</dd>

      {Object.keys(stage.params).length > 0 && (
        <>
          <dt className="font-semibold text-muted">{t.viewer.params}</dt>
          {/* Straight from config/preprocess.json — §8 forbids retyping these here. */}
          <dd className="font-mono break-all">{JSON.stringify(stage.params)}</dd>
        </>
      )}

      {w !== undefined && h !== undefined && (
        <>
          <dt className="font-semibold text-muted">{t.viewer.size}</dt>
          <dd className="font-mono tabular-nums">
            {display && display[1] !== undefined && display[0] !== undefined ? (
              <>
                {t.viewer.shownAt(display[1], display[0])} ·{' '}
                <span className="text-muted">{t.viewer.originalSize(w, h)}</span>
              </>
            ) : (
              `${w}×${h}`
            )}
          </dd>
        </>
      )}

      {stage.n >= 9 && stage.n <= 11 && (
        <dd className="sm:col-span-2 max-w-prose text-muted">{t.viewer.featureNote}</dd>
      )}
      {stage.n === 12 && (
        <dd className="sm:col-span-2 max-w-prose text-muted">{t.viewer.gradcamNote}</dd>
      )}
    </dl>
  )
}

function Controls({
  index,
  total,
  playing,
  compare,
  onStep,
  onTogglePlay,
  onToggleCompare,
}: {
  index: number
  total: number
  playing: boolean
  compare: boolean
  onStep: (d: number) => void
  onTogglePlay: () => void
  onToggleCompare: () => void
}) {
  return (
    <div>
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => onStep(-1)}
          disabled={index === 0}
          aria-label={t.viewer.previous}
          className="rounded-lg border border-primary px-3 py-2 text-primary disabled:opacity-40"
        >
          ←
        </button>
        <button
          type="button"
          onClick={onTogglePlay}
          className="flex-1 rounded-lg bg-primary px-4 py-2 font-medium text-white hover:bg-primary-strong"
          aria-pressed={playing}
        >
          {playing ? t.viewer.pause : t.viewer.play}
        </button>
        <button
          type="button"
          onClick={() => onStep(1)}
          disabled={index >= total - 1}
          aria-label={t.viewer.next}
          className="rounded-lg border border-primary px-3 py-2 text-primary disabled:opacity-40"
        >
          →
        </button>
      </div>

      <p aria-live="polite" className="mt-2 text-center text-sm text-muted">
        {t.viewer.step(index + 1, total)}
      </p>

      <label className="mt-3 flex items-center gap-2 text-sm">
        <input type="checkbox" checked={compare} onChange={onToggleCompare} />
        {t.viewer.compare}
      </label>
      <p className="mt-2 text-xs text-muted">{t.viewer.keyboard}</p>
    </div>
  )
}

function Rail({
  ref,
  stages,
  index,
  onPick,
}: {
  ref: React.Ref<HTMLDivElement>
  stages: Stage[]
  index: number
  onPick: (i: number) => void
}) {
  return (
    <div
      ref={ref}
      className="flex gap-2 overflow-x-auto pb-2 lg:grid lg:grid-cols-4 lg:overflow-visible"
    >
      {stages.map((stage, i) => (
        <button
          key={stage.n}
          type="button"
          data-thumb={i}
          onClick={() => onPick(i)}
          aria-current={i === index}
          aria-label={`${stage.n}. ${stage.title}`}
          className={`shrink-0 rounded-md border-2 p-0.5 ${
            i === index ? 'border-primary' : 'border-transparent hover:border-border'
          }`}
        >
          {stage.png_base64 ? (
            <img
              src={stageImageSrc(stage.png_base64)}
              alt=""
              className={`h-14 w-14 rounded object-cover ${
                PIXELATED_KEYS.has(stage.key) ? 'pixelated' : ''
              }`}
            />
          ) : (
            <span className="flex h-14 w-14 items-center justify-center rounded bg-surface text-xs font-semibold tabular-nums">
              {stage.n}
            </span>
          )}
        </button>
      ))}
    </div>
  )
}
