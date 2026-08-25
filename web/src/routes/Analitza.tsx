/**
 * `/analitza` — file or drag-drop, then the prediction. `docs/05-webapp-spec.md` §4, §8.
 *
 * One request per image, and it returns everything: the answer, the thirteen stages, the
 * measured times. The viewer below reads that same object. There is no second call "to
 * show" what the first call decided (§3, §8).
 */

import { useRef, useState } from 'react'
import type { Prediction } from '../api/types'
import { ApiError, predict } from '../api/client'
import { GeometryWarning } from '../components/Banners'
import { PipelineViewer } from '../components/PipelineViewer'
import { Result } from '../components/Result'
import { t } from '../strings'

type State =
  | { status: 'idle' }
  | { status: 'working' }
  | { status: 'done'; prediction: Prediction }
  | { status: 'error'; message: string }

function messageFor(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 415) return t.analyze.error.unsupported
    if (error.status === 400) return t.analyze.error.empty
    return error.message
  }
  return t.analyze.error.network
}

export function Analitza() {
  const [state, setState] = useState<State>({ status: 'idle' })
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  async function analyse(file: File) {
    setState({ status: 'working' })
    try {
      setState({ status: 'done', prediction: await predict(file) })
    } catch (error) {
      setState({ status: 'error', message: messageFor(error) })
    }
  }

  function reset() {
    setState({ status: 'idle' })
    // Clear the input too, or picking the same file again fires no change event and the
    // app looks broken to someone retrying after a bad photo.
    if (inputRef.current) inputRef.current.value = ''
  }

  return (
    <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
      <h1 className="text-3xl font-semibold">{t.nav.analyze}</h1>

      {state.status !== 'done' && (
        <div
          onDragOver={(e) => {
            e.preventDefault()
            setDragging(true)
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDragging(false)
            const file = e.dataTransfer.files[0]
            if (file) void analyse(file)
          }}
          className={`mt-6 rounded-xl border-2 border-dashed p-10 text-center ${
            dragging ? 'border-primary bg-surface' : 'border-warn-border/60 bg-surface/60'
          }`}
        >
          <p className="text-lg">{t.analyze.drop}</p>
          <input
            ref={inputRef}
            id="capture"
            type="file"
            accept="image/*,.heic,.heif"
            className="sr-only"
            onChange={(e) => {
              const file = e.target.files?.[0]
              if (file) void analyse(file)
            }}
          />
          <label
            htmlFor="capture"
            className="mt-4 inline-block cursor-pointer rounded-lg bg-primary px-5 py-2.5 font-medium text-white hover:bg-primary-strong"
          >
            {t.analyze.choose}
          </label>
          <p className="mt-4 text-sm text-muted">{t.analyze.privacy}</p>
        </div>
      )}

      {state.status === 'working' && (
        <p role="status" className="mt-6 text-lg">
          {t.analyze.working}
        </p>
      )}

      {state.status === 'error' && (
        <div role="alert" className="mt-6 rounded-xl border border-warn-border bg-warn-surface p-5 text-warn-ink">
          <p>{state.message}</p>
          <button
            type="button"
            onClick={reset}
            className="mt-3 rounded-lg border border-warn-ink px-4 py-2 font-medium"
          >
            {t.common.retry}
          </button>
        </div>
      )}

      {state.status === 'done' && (
        <div className="mt-6 space-y-6">
          {/* Its own state, above the tier and independent of it (§4). */}
          <GeometryWarning prediction={state.prediction} />

          <div className="grid gap-6 lg:grid-cols-2">
            <Result prediction={state.prediction} />
            <SideBySide prediction={state.prediction} />
          </div>

          <PipelineViewer prediction={state.prediction} />

          <div className="flex flex-wrap items-center gap-4">
            <button
              type="button"
              onClick={reset}
              className="rounded-lg bg-primary px-5 py-2.5 font-medium text-white hover:bg-primary-strong"
            >
              {t.analyze.again}
            </button>
            <span className="font-mono text-sm tabular-nums text-muted">
              {t.analyze.elapsed}: {state.prediction.elapsed_ms.toFixed(0)} ms
            </span>
          </div>
        </div>
      )}
    </div>
  )
}

/**
 * §4: "Always show the preprocessed edge map next to the original. Seeing what the model
 * actually sees is half the didactic value and makes failures self-explanatory."
 */
function SideBySide({ prediction }: { prediction: Prediction }) {
  const original = prediction.stages.find((s) => s.n === 1)
  const edges = prediction.stages.find((s) => s.key === 'resized')
  if (!original?.png_base64 || !edges?.png_base64) return null
  return (
    <section className="card grid grid-cols-2 gap-4 p-5">
      <figure>
        <img
          src={`data:image/png;base64,${original.png_base64}`}
          alt={t.analyze.original}
          className="w-full rounded-lg border border-border"
        />
        <figcaption className="mt-2 text-sm text-muted">{t.analyze.original}</figcaption>
      </figure>
      <figure>
        <img
          src={`data:image/png;base64,${edges.png_base64}`}
          alt={t.analyze.edges}
          className="pixelated w-full rounded-lg border border-border bg-surface"
        />
        <figcaption className="mt-2 text-sm text-muted">{t.analyze.edges}</figcaption>
      </figure>
    </section>
  )
}
