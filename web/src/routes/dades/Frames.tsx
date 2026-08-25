/**
 * Steps 3 and 4 — the classes, and the gate answering while the folder is still uploading.
 * `docs/05-webapp-spec.md` §9.
 *
 * §9 calls step 4 the one that earns this view: "each image is preprocessed and gated AS IT
 * ARRIVES… You find out that the lighting is wrong on photo three, not after importing four
 * hundred." Everything below follows from that sentence rather than from what is convenient.
 *
 * **One label per class.** There is a single `freq_hz` field, used to open a class, and then a
 * folder of photographs goes into it. §9 forbids the per-image alternative outright: a label
 * typed six hundred times is a label typed wrong, and there is no reason for it — a burst of a
 * mode is one mode. The class travels as the query parameter of every upload (`uploadFrame`),
 * so a photograph carries the label of the box it was dropped into and cannot carry any other.
 *
 * **The verdicts stream, and a refusal is a result.** `uploadQueue` runs the batch three at a
 * time and calls back once per file, and each callback lands in state on its own — so the
 * counters move while the rest are in the air, and the eleventh photo being unreadable stops
 * nothing. The alternative, `Promise.all` over the folder, reads the same in a diff and turns
 * six hundred uploads into one all-or-nothing wait whose first news is at the end.
 *
 * **Three counters, not two.** Accepted, rejected, and *duplicated*. A frame refused for being
 * bytes the draft already holds never reached the gate — see `store_frame` — so filing it under
 * `rejected` would report a retried batch as a lighting problem, and someone would go and adjust
 * the lamp. `src/dataset_api.py::_summary` keeps the same three categories apart for the same
 * reason.
 *
 * **The pictures are the server's, and so are the bytes.** The `File` objects out of the input's
 * `FileList` go straight to `uploadFrame`; nothing here decodes, resizes or re-encodes one, and
 * nothing shows a local preview of a file in flight — the thumbnail worth looking at is the one
 * written from the *stored* bytes, which is the only picture that is evidence about what was
 * stored. `docs/02` §4: `data/raw/` is untouched, straight off the phone.
 *
 * **A `partial` frame gets no edge map.** Fewer than four markers means no rectified plate, so
 * there is no edge map that means anything; `write_previews` does not write one and
 * `GET /preview?kind=edges` answers 404. Asking for it anyway would be displaying a
 * rectification that did not happen.
 */

import { useState } from 'react'
import { ApiError } from '../../api/client'
import { deleteFrame, previewSrc, uploadFrame } from '../../api/drafts'
import type { Draft, FrameRecord } from '../../api/types'
import { uploadQueue } from '../../lib/upload'
import { t } from '../../strings'

interface Props {
  draft: Draft
  /** Called after anything that changed the draft on the server, so the wizard re-reads it. */
  onChange: () => void
}

/** One refusal, kept beside the class it happened in and the file it was about. */
interface Problem {
  key: string
  filename: string
  message: string
}

/**
 * What this view has watched happen to one class since it was rendered.
 *
 * It is not a second copy of the draft: `records` is merged over the server's own list on
 * every render, and its only job is to carry a verdict from the moment it arrives to the
 * moment the wizard has re-read the draft. `duplicates` and `problems` are the exception —
 * a refused upload leaves nothing in `draft.frames` to merge, because nothing was stored.
 */
interface ClassState {
  records: FrameRecord[]
  duplicates: number
  problems: Problem[]
  done: number
  total: number
}

const EMPTY: ClassState = { records: [], duplicates: 0, problems: [], done: 0, total: 0 }

/** `imageio.SUPPORTED_SUFFIXES`, as a hint to the file picker. The gate is the server's. */
const ACCEPT = '.heic,.heif,.jpg,.jpeg,.png,image/*'

/** `quality` values that mean the gate said no. `pending` is neither — it is stored-not-gated. */
function isRejected(record: FrameRecord): boolean {
  return record.quality !== 'ok' && record.quality !== 'pending'
}

/**
 * The Catalan for a gate reason, or the reason itself when there is none.
 *
 * A missing key is reported as the server's own token rather than papered over with an
 * invented sentence: an unrecognised reason is a string that needs adding to `strings.ts` and
 * to `docs/07-glossary-ca-en.md`, and seeing `edge_density` on screen is what makes that
 * happen. Guessing at it here would hide a new rejection reason behind a plausible phrase.
 */
function reasonText(reason: string): string {
  const reasons: Record<string, string> = t.data.frames.reasons
  return reasons[reason] ?? reason
}

/**
 * The Catalan for a refusal, or the server's own sentence when the code has none.
 *
 * The codes come from `src/dataset_api.py`; the sentences live here, because §9's routes answer
 * `{code, message}` precisely so that no Catalan is written in Python. `message` is the fallback
 * and not the default: it is English, meant for the log, and it is shown when showing it is more
 * honest than inventing a claim about a refusal this component did not make.
 */
function refusalText(error: Error): string {
  if (error instanceof ApiError) {
    if (error.code === 'duplicate_frame') return t.data.frames.duplicate
    if (error.code === 'duplicate_content') return t.data.frames.duplicateContent
    if (error.code === 'unsupported_image') return t.data.frames.unsupported
  }
  return error.message
}

export function Frames({ draft, onChange }: Props) {
  const [added, setAdded] = useState<number[]>([])
  const [seen, setSeen] = useState<Record<number, ClassState>>({})
  const [freqField, setFreqField] = useState('')

  // The classes the server knows about, plus the ones opened here and still empty. A class with
  // no photographs in it yet exists nowhere but on this screen — it is a box waiting for a
  // folder, and the server hears about it with the first upload.
  const classes = [...new Set([...draft.classes, ...added])].sort((a, b) => a - b)

  /**
   * The frames of one class: the server's list, with anything this view has watched land since
   * merged over it by name. Both sides are the server's answers, so this is a freshness choice
   * and not a second source of truth.
   */
  function recordsOf(freq: number): FrameRecord[] {
    const byName = new Map<string, FrameRecord>()
    for (const stored of draft.frames[String(freq)] ?? []) byName.set(stored.name, stored)
    for (const arrived of seen[freq]?.records ?? []) byName.set(arrived.name, arrived)
    return [...byName.values()].sort((a, b) => a.name.localeCompare(b.name))
  }

  function update(freq: number, change: (current: ClassState) => ClassState) {
    setSeen((current) => ({ ...current, [freq]: change(current[freq] ?? EMPTY) }))
  }

  function openClass(event: React.FormEvent) {
    event.preventDefault()
    const freq = Number.parseInt(freqField, 10)
    if (!Number.isFinite(freq) || freq <= 0) return
    setAdded((current) => (current.includes(freq) ? current : [...current, freq]))
    setFreqField('')
  }

  async function upload(freq: number, chosen: FileList | null) {
    const files = [...(chosen ?? [])]
    if (files.length === 0) return
    update(freq, (current) => ({ ...current, total: current.total + files.length }))

    // The queue's own concurrency: three at a time, one callback per file. `uploadFrame` is
    // handed the `File` straight out of the `FileList` — see this module's header.
    await uploadQueue(files, (file) => uploadFrame(draft.draft_id, freq, file), {
      onResult: (file, arrived, failure) => {
        update(freq, (current) => {
          const next: ClassState = { ...current, done: current.done + 1 }
          if (arrived !== null) return { ...next, records: [...current.records, arrived] }
          const error = failure ?? new Error(t.data.error.upload)
          return {
            ...next,
            duplicates:
              error instanceof ApiError && error.code === 'duplicate_content'
                ? current.duplicates + 1
                : current.duplicates,
            problems: [
              ...current.problems,
              {
                key: `${freq}-${current.problems.length}-${file.name}`,
                filename: file.name,
                message: refusalText(error),
              },
            ],
          }
        })
      },
    })

    // Once, at the end of the batch. The counters above have been moving all along; this is
    // for the summary the wizard shows and for `draft.frames`, and re-reading the draft after
    // every one of six hundred verdicts would be six hundred requests to say what the last
    // callback already said.
    onChange()
  }

  async function remove(freq: number, name: string) {
    await deleteFrame(draft.draft_id, freq, name)
    update(freq, (current) => ({
      ...current,
      records: current.records.filter((entry) => entry.name !== name),
    }))
    onChange()
  }

  return (
    <div className="mt-4">
      <p className="text-muted">{t.data.frames.lead}</p>

      <form onSubmit={openClass} className="mt-4 flex flex-wrap items-end gap-3">
        <div>
          <label htmlFor="freq_hz" className="font-medium">
            {t.data.frames.freq}
          </label>
          <input
            id="freq_hz"
            name="freq_hz"
            type="number"
            min="1"
            step="1"
            inputMode="numeric"
            value={freqField}
            onChange={(event) => setFreqField(event.target.value)}
            className="mt-1 w-36 rounded-lg border border-border bg-white px-3 py-2 font-mono focus:border-primary"
          />
        </div>
        <button
          type="submit"
          className="rounded-lg bg-primary px-5 py-2.5 font-medium text-white hover:bg-primary-strong"
        >
          {t.data.frames.addClass}
        </button>
      </form>

      <div className="mt-6 space-y-6">
        {classes.map((freq) => (
          <ClassPanel
            key={freq}
            draftId={draft.draft_id}
            freq={freq}
            records={recordsOf(freq)}
            state={seen[freq] ?? EMPTY}
            onFiles={(files) => void upload(freq, files)}
            onRemove={(name) => void remove(freq, name)}
          />
        ))}
      </div>
    </div>
  )
}

interface PanelProps {
  draftId: string
  freq: number
  records: FrameRecord[]
  state: ClassState
  onFiles: (files: FileList | null) => void
  onRemove: (name: string) => void
}

function ClassPanel({ draftId, freq, records, state, onFiles, onRemove }: PanelProps) {
  const [dragging, setDragging] = useState(false)
  const inputId = `frames-${freq}`

  const accepted = records.filter((record) => record.quality === 'ok').length
  const rejected = records.filter(isRejected).length
  const pending = records.filter((record) => record.quality === 'pending').length
  const uploading = state.done < state.total

  return (
    <section className="rounded-xl border border-border p-5">
      <h3 className="font-mono text-xl font-semibold">{freq} Hz</h3>

      <div className="mt-3 flex flex-wrap gap-6">
        <Counter freq={freq} kind="accepted" label={t.data.frames.accepted} value={accepted} />
        <Counter freq={freq} kind="rejected" label={t.data.frames.rejected} value={rejected} />
        {/* Third category, deliberately: see this module's header. It counts what this view
            has watched be refused — the draft keeps its own total, which `/dades` shows in
            the listing and in the review. */}
        <Counter
          freq={freq}
          kind="duplicated"
          label={t.data.frames.duplicated}
          value={state.duplicates}
        />
        {/* Stored but not yet gated. Shown only when there is one, because it is a real state
            and a frame in it must be visible as something rather than as nothing. */}
        {pending > 0 && (
          <Counter freq={freq} kind="pending" label={t.data.frames.pending} value={pending} />
        )}
      </div>

      <div
        onDragOver={(event) => {
          event.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault()
          setDragging(false)
          onFiles(event.dataTransfer.files)
        }}
        className={`mt-4 rounded-xl border-2 border-dashed p-6 text-center ${
          dragging ? 'border-primary bg-surface' : 'border-border bg-surface/60'
        }`}
      >
        <label htmlFor={inputId} className="cursor-pointer text-lg">
          {t.data.frames.drop}
        </label>
        <input
          id={inputId}
          type="file"
          multiple
          accept={ACCEPT}
          className="sr-only"
          onChange={(event) => {
            onFiles(event.target.files)
            // Or picking the same folder again fires no change event and the app looks broken
            // to someone retrying after a batch that half failed.
            event.target.value = ''
          }}
        />
      </div>

      {uploading && (
        <p role="status" className="mt-3 flex flex-wrap gap-2">
          <span className="font-mono">{t.data.frames.uploading(state.done, state.total)}</span>
          <span className="text-muted">{t.data.frames.keepGoing}</span>
        </p>
      )}

      {state.problems.length > 0 && (
        <ul aria-live="polite" className="mt-3 space-y-2">
          {state.problems.map((problem) => (
            <li
              key={problem.key}
              className="rounded-lg border border-warn-border bg-warn-surface p-3 text-warn-ink"
            >
              <span className="font-mono text-sm">{problem.filename}</span>
              <span className="mt-1 block">{problem.message}</span>
            </li>
          ))}
        </ul>
      )}

      {records.some(isRejected) && (
        <ul className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
          {records.filter(isRejected).map((record) => (
            <Rejection
              key={record.name}
              draftId={draftId}
              freq={freq}
              record={record}
              onRemove={() => onRemove(record.name)}
            />
          ))}
        </ul>
      )}

      {records.some((record) => record.quality === 'ok') && (
        <ul className="mt-4 space-y-1">
          {records
            .filter((record) => record.quality === 'ok')
            .map((record) => (
              <li key={record.name} className="flex items-center gap-3 text-sm">
                <span className="font-mono">{record.name}</span>
                <span className="text-muted">{record.original_filename}</span>
                <button
                  type="button"
                  onClick={() => onRemove(record.name)}
                  className="rounded border border-border px-2 py-1 hover:bg-surface"
                >
                  {t.data.frames.remove}
                </button>
              </li>
            ))}
        </ul>
      )}
    </section>
  )
}

function Counter({
  freq,
  kind,
  label,
  value,
}: {
  freq: number
  kind: string
  label: string
  value: number
}) {
  return (
    <p className="flex flex-col">
      <span className="text-sm text-muted">{label}</span>
      <span data-testid={`${kind}-${freq}`} className="font-mono text-2xl">
        {value}
      </span>
    </p>
  )
}

function Rejection({
  draftId,
  freq,
  record,
  onRemove,
}: {
  draftId: string
  freq: number
  record: FrameRecord
  onRemove: () => void
}) {
  // No four markers, no rectified plate, no edge map — see this module's header.
  const hasEdges = record.reason !== 'partial'

  return (
    <li className="rounded-lg border border-warn-border bg-warn-surface p-3 text-warn-ink">
      <p className="font-mono text-sm">{record.name}</p>
      <p className="text-sm">{record.original_filename}</p>
      {record.reason !== null && <p className="mt-1 font-medium">{reasonText(record.reason)}</p>}

      {record.preview && (
        <div className="mt-2 flex gap-2">
          <img
            src={previewSrc(draftId, freq, record.name, 'original')}
            alt={t.data.frames.original}
            className="w-1/2 rounded border border-border"
          />
          {hasEdges && (
            <img
              src={previewSrc(draftId, freq, record.name, 'edges')}
              alt={t.data.frames.edges}
              className="w-1/2 rounded border border-border"
            />
          )}
        </div>
      )}

      <button
        type="button"
        onClick={onRemove}
        className="mt-2 rounded border border-warn-ink px-2 py-1 text-sm"
      >
        {t.data.frames.remove}
      </button>
    </li>
  )
}
