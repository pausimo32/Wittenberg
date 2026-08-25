/**
 * Step 2 — the calibration frame. `docs/05-webapp-spec.md` §9, `docs/02-dataset-spec.md` §1.1.
 *
 * One photograph of the marc, once per session, before any pattern. Four `DICT_4X4_50` markers
 * or nothing: fewer than four is a `partial` rejection under §6 and never a quiet fall-back to
 * `center_crop`, so until this step answers yes there is no rectified plate, and without a
 * rectified plate the quality gate has nothing to measure. That is why the photo step below is
 * held shut on `draft.calibrated` — see `DraftWizard`.
 *
 * **The drawing is the server's.** `POST /calibration` returns a PNG with the quad already on
 * it, drawn in Python on the very array the homography was computed from, and this component
 * shows those bytes. `calibration.corners` arrives too, and drawing it here — a canvas, an SVG
 * overlay on the preview — is the tempting thing to do and the wrong one: it would be a second
 * implementation of the geometry, in a second coordinate space, at a second scale. Two
 * implementations agree until they do not, and the day they part the picture is the convincing
 * evidence and the wrong one. `corners` is data about the calibration, not a drawing recipe.
 *
 * **The flags are shown, never swallowed.** They are §1.1's nominal-corner mismatch: ingest
 * picks the calibration point geometrically — of a marker's four corners, the one nearest the
 * centroid of the four — and then compares that with the table for the nominal rig. A
 * disagreement means a marker is taped on rotated or the plate is turned in the frame. Both are
 * legitimate, and the calibration is correct either way because the geometry chose the point,
 * which is precisely what makes the warning so easy to drop. Its entire value is arriving on
 * the day of the session, while the tape is still reachable, instead of three weeks later out
 * of `data/ingest_report.json`. They are English prose from Python and are rendered as they
 * are: paraphrasing a rig report into Catalan here would be inventing the second half of it.
 */

import { useRef, useState } from 'react'
import type { CalibrationInfo, Draft } from '../../api/types'
import { uploadCalibration } from '../../api/drafts'
import { t } from '../../strings'

interface Props {
  draft: Draft
  /** Called after the server has answered, so the wizard re-reads the draft it holds. */
  onCalibrated: () => void
}

/** What the last upload returned: the calibration and the PNG it was drawn on. */
interface Result {
  calibration: CalibrationInfo
  preview: string
}

export function Calibration({ draft, onCalibrated }: Props) {
  const [result, setResult] = useState<Result | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [retrying, setRetrying] = useState(false)
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  // The upload's answer when there is one, the draft's stored calibration otherwise. Both come
  // from the same place — the server — so this is a freshness choice, not two sources of truth.
  const calibration = result?.calibration ?? draft.calibration
  const preview = result?.preview ?? null
  const showDropZone = (calibration === null || retrying) && !busy

  function reset() {
    setRetrying(true)
    setError(null)
    // Clear the input, or picking the same file again fires no change event and the app looks
    // broken to someone retrying after a bad photograph.
    if (inputRef.current) inputRef.current.value = ''
  }

  async function calibrate(file: File) {
    setBusy(true)
    setError(null)
    try {
      const answer = await uploadCalibration(draft.draft_id, file)
      setResult(answer)
      setRetrying(false)
      // Whatever the verdict: the draft on the server has changed, and the wizard's copy of
      // it decides whether the photo step is open. A failed detection changes it too — from
      // "no calibration" to "a calibration that says no".
      onCalibrated()
    } catch (failure) {
      // The server's own sentence. It knows which file it could not decode and why; a
      // friendlier guess written here would be a guess.
      setError(failure instanceof Error ? failure.message : String(failure))
      setRetrying(true)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="mt-4">
      <p className="text-muted">{t.data.calibration.lead}</p>

      {showDropZone && (
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
            if (file) void calibrate(file)
          }}
          className={`mt-4 rounded-xl border-2 border-dashed p-8 text-center ${
            dragging ? 'border-primary bg-surface' : 'border-border bg-surface/60'
          }`}
        >
          <p className="text-lg">{t.data.calibration.drop}</p>
          <input
            ref={inputRef}
            id="calibration-frame"
            type="file"
            accept="image/*,.heic,.heif"
            className="sr-only"
            onChange={(e) => {
              const file = e.target.files?.[0]
              if (file) void calibrate(file)
            }}
          />
          <label
            htmlFor="calibration-frame"
            className="mt-4 inline-block cursor-pointer rounded-lg bg-primary px-5 py-2.5 font-medium text-white hover:bg-primary-strong"
          >
            {t.data.calibration.choose}
          </label>
        </div>
      )}

      {busy && (
        <p role="status" className="mt-4">
          {t.data.calibration.working}
        </p>
      )}

      {error !== null && (
        <p
          role="alert"
          className="mt-4 rounded-lg border border-warn-border bg-warn-surface p-4 text-warn-ink"
        >
          {error}
        </p>
      )}

      {calibration !== null && (
        <div className="mt-4 space-y-3">
          <p>{t.data.calibration.found(calibration.n_markers)}</p>

          {calibration.ok ? (
            <>
              <p>{t.data.calibration.ok}</p>
              {calibration.px_per_mm !== null && (
                <p className="font-mono">{t.data.calibration.pxPerMm(calibration.px_per_mm)}</p>
              )}
            </>
          ) : (
            // Not `role="alert"`: fewer than four markers is an ordinary outcome of a
            // photograph, and the answer to it is another photograph, not a bug report.
            <p className="rounded-lg border border-warn-border bg-warn-surface p-4 text-warn-ink">
              {t.data.calibration.failed}
            </p>
          )}

          {preview !== null && (
            <img
              src={`data:image/png;base64,${preview}`}
              alt={t.data.calibration.title}
              className="max-w-full rounded-lg border border-border"
            />
          )}

          {calibration.flags.length > 0 && (
            <div className="rounded-lg border border-warn-border bg-warn-surface p-4 text-warn-ink">
              <h3 className="font-semibold">{t.data.calibration.flags}</h3>
              <ul className="mt-2 list-disc space-y-1 pl-5 text-sm">
                {calibration.flags.map((flag) => (
                  <li key={flag}>{flag}</li>
                ))}
              </ul>
            </div>
          )}

          {!showDropZone && (
            <button
              type="button"
              onClick={reset}
              className="rounded-lg border border-border px-4 py-2 font-medium hover:bg-surface"
            >
              {t.data.calibration.retry}
            </button>
          )}
        </div>
      )}
    </div>
  )
}
