/**
 * Steps 3 and 4 — one label per class, and six hundred verdicts arriving.
 *
 * `docs/05-webapp-spec.md` §9 calls step 4 "the one that earns this view": each photograph is
 * gated *as it arrives*, so the lighting is wrong on photo three rather than on four hundred.
 * Everything asserted here is a consequence of that sentence.
 *
 * - **The label is typed once per class.** One `freq_hz` field, then a folder. A per-image
 *   label field is the thing §9 forbids, so the test counts the fields rather than trusting
 *   the layout.
 * - **The verdicts stream.** Three files go out at once and the counters move on the first
 *   answer, while the other two are still in flight. A view that waited for the batch would
 *   pass a "counts them correctly" test and fail the requirement completely.
 * - **A refusal is a result, not an end.** A 415 in the middle of a folder leaves the rest
 *   uploading — the half of requirement 5 that is easy to get wrong.
 * - **Three counters, not two.** A frame refused for being bytes the draft already holds never
 *   reached the gate. Counting it as a rejection would make a retried batch read as a lighting
 *   problem, which is the one failure this whole category exists to prevent.
 * - **A rejection shows its reason next to the picture it is about**, and a `partial` frame has
 *   no edge map to show: without four markers there is no rectified plate, so an edge map would
 *   be a rectification that did not happen (and `GET /preview?kind=edges` answers 404).
 *
 * The bytes surviving the trip is `api/drafts.test.ts`'s assertion and stays there — one test
 * for one property. What is asserted here is the other half: this component hands `uploadFrame`
 * the `File` out of the input's `FileList` and never a copy of it, so there is no canvas and no
 * `createImageBitmap` anywhere on the path.
 *
 * `CALIBRATED` is declared here rather than imported from `Calibration.test.tsx`: a fixture
 * shared between test files is a fixture that changes under a test that did not ask for it.
 */

import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { Draft, FrameRecord } from '../../api/types'
import { Frames } from './Frames'

afterEach(() => vi.unstubAllGlobals())

const CALIBRATED: Draft = {
  draft_id: '20260830-01',
  session_id: '20260830-01',
  status: 'calibrated',
  path: 'C:\\Users\\PC10\\Desktop\\ide1\\drafts\\20260830-01',
  created: '2026-08-30T10:04:11',
  updated: '2026-08-30T10:41:02',
  calibrated: true,
  classes: [],
  total: 0,
  accepted: 0,
  rejected: 0,
  pending: 0,
  duplicates: 0,
  bytes: 0,
  session: {},
  calibration: {
    ok: true,
    roi_mode: 'homography',
    n_markers: 4,
    marker_ids: [0, 1, 2, 3],
    corners: [
      [0, 0],
      [1, 0],
      [1, 1],
      [0, 1],
    ],
    px_per_mm: 14.5,
    flags: [],
  },
  frames: {},
  existing_rows: 0,
}

function jpeg(n: number): File {
  return new File([new Uint8Array([n])], `IMG_${n}.jpg`, { type: 'image/jpeg' })
}

/** A `FrameRecord` as the server sends it, with the fields a test cares about overridden. */
function record(over: Partial<FrameRecord> = {}): FrameRecord {
  return {
    name: 'IMG_0001.jpg',
    original_filename: 'IMG_1.jpg',
    freq_hz: 440,
    quality: 'ok',
    reason: null,
    metrics: {},
    sha256: '',
    preview: false,
    ...over,
  }
}

function verdict(over: Partial<FrameRecord> = {}): Response {
  return new Response(JSON.stringify(record(over)), { status: 200 })
}

function refusal(code: string, message: string, status = 409): Response {
  return new Response(JSON.stringify({ detail: { code, message } }), { status })
}

/** A promise the test holds the other end of, so "still in flight" is a state it controls. */
function deferred<T>(): { promise: Promise<T>; resolve: (value: T) => void } {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((settle) => {
    resolve = settle
  })
  return { promise, resolve }
}

async function addClass(freq: string) {
  await userEvent.type(screen.getByLabelText('Freqüència (Hz)'), freq)
  await userEvent.click(screen.getByRole('button', { name: 'Afegeix la classe' }))
}

function dropZone(): HTMLElement {
  return screen.getByLabelText(/Arrossega aquí les fotografies/)
}

describe('Frames', () => {
  it('uploads a dropped folder and counts the verdicts as they arrive', async () => {
    let n = 0
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        n += 1
        return verdict({
          name: `IMG_000${n}.jpg`,
          original_filename: `IMG_${n}.jpg`,
          quality: n === 2 ? 'blurred' : 'ok',
          reason: n === 2 ? 'blurred' : null,
          preview: n === 2,
        })
      }),
    )

    render(<Frames draft={CALIBRATED} onChange={() => {}} />)
    await addClass('440')
    await userEvent.upload(dropZone(), [jpeg(1), jpeg(2), jpeg(3)])

    expect(await screen.findByText('borrosa')).toBeInTheDocument()
    await waitFor(() => expect(screen.getByTestId('accepted-440')).toHaveTextContent('2'))
    expect(screen.getByTestId('rejected-440')).toHaveTextContent('1')
    expect(screen.getByTestId('duplicated-440')).toHaveTextContent('0')
  })

  it('moves the counters while the rest of the batch is still in flight', async () => {
    // §9's "you find out that the lighting is wrong on photo three". A view that collected the
    // batch and rendered it at the end would pass every counting test above and fail this.
    const gates = [deferred<Response>(), deferred<Response>(), deferred<Response>()]
    let issued = 0
    vi.stubGlobal(
      'fetch',
      vi.fn(() => gates[issued++]!.promise),
    )

    render(<Frames draft={CALIBRATED} onChange={() => {}} />)
    await addClass('440')
    await userEvent.upload(dropZone(), [jpeg(1), jpeg(2), jpeg(3)])

    // The queue's own default: three at a time, so the second and third are already on the
    // wire when the first one answers.
    await waitFor(() => expect(vi.mocked(fetch)).toHaveBeenCalledTimes(3))

    gates[0]!.resolve(verdict({ name: 'IMG_0001.jpg' }))
    await waitFor(() => expect(screen.getByTestId('accepted-440')).toHaveTextContent('1'))
    expect(screen.getByText('1 de 3')).toBeInTheDocument()
    expect(screen.getByText('Pots continuar treballant mentre es processen.')).toBeInTheDocument()

    gates[1]!.resolve(verdict({ name: 'IMG_0002.jpg', quality: 'blurred', reason: 'blurred' }))
    gates[2]!.resolve(verdict({ name: 'IMG_0003.jpg' }))
    await waitFor(() => expect(screen.getByTestId('accepted-440')).toHaveTextContent('2'))
    expect(screen.getByTestId('rejected-440')).toHaveTextContent('1')
  })

  it('keeps going after one photo is refused', async () => {
    // Requirement 5, the half that is easy to get wrong: one bad file in a folder of sixty
    // must not stop the other fifty-nine. The refused file is a `.jpg` the *server* cannot
    // decode rather than a `notes.txt`, because `userEvent.upload` applies the input's own
    // `accept` filter and a `.txt` would never leave the browser — which is the input doing
    // its job, not the batch surviving a refusal.
    let n = 0
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        n += 1
        return n === 2
          ? refusal('unsupported_image', 'IMG_2.jpg: cannot be decoded', 415)
          : verdict({ name: `IMG_000${n}.jpg`, original_filename: `IMG_${n}.jpg` })
      }),
    )

    render(<Frames draft={CALIBRATED} onChange={() => {}} />)
    await addClass('440')
    await userEvent.upload(dropZone(), [jpeg(1), jpeg(2), jpeg(3)])

    expect(
      await screen.findByText('Format no admès. Han de ser HEIC, JPEG o PNG.'),
    ).toBeInTheDocument()
    await waitFor(() => expect(screen.getByTestId('accepted-440')).toHaveTextContent('2'))
    expect(screen.getByTestId('rejected-440')).toHaveTextContent('0')
  })

  it('counts a repeated capture apart from the rejections', async () => {
    // Requirement 8. Six hundred uploads over a connection that stumbles means someone retries
    // a batch. Those frames never reached the gate: filed under `rejected` they would read as
    // a lighting problem, and someone would go and change the lamp.
    let n = 0
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        n += 1
        return n === 2
          ? refusal('duplicate_content', 'IMG_2.jpg has the same bytes as IMG_0001.jpg')
          : verdict({ name: `IMG_000${n}.jpg`, original_filename: `IMG_${n}.jpg` })
      }),
    )

    render(<Frames draft={CALIBRATED} onChange={() => {}} />)
    await addClass('440')
    await userEvent.upload(dropZone(), [jpeg(1), jpeg(2), jpeg(3)])

    const repeated = /Una fotografia repetida no és una segona observació/
    expect(await screen.findByText(repeated)).toBeInTheDocument()
    await waitFor(() => expect(screen.getByTestId('duplicated-440')).toHaveTextContent('1'))
    expect(screen.getByTestId('rejected-440')).toHaveTextContent('0')
    expect(screen.getByTestId('accepted-440')).toHaveTextContent('2')
  })

  it('says which refusal it was, and falls back to the server when it has no sentence', async () => {
    let n = 0
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        n += 1
        return n === 1
          ? refusal('duplicate_frame', 'IMG_1.jpg is already in the 440 Hz class')
          : refusal('not_calibrated', 'this draft has no validated calibration yet')
      }),
    )

    render(<Frames draft={CALIBRATED} onChange={() => {}} />)
    await addClass('440')
    await userEvent.upload(dropZone(), [jpeg(1), jpeg(2)])

    const droppedTwice = /Sembla que la carpeta s’ha deixat anar dues vegades/
    expect(await screen.findByText(droppedTwice)).toBeInTheDocument()
    // No Catalan for this code, so the server's own English sentence is shown rather than a
    // friendlier one invented here about a refusal this component did not make.
    expect(
      await screen.findByText('this draft has no validated calibration yet'),
    ).toBeInTheDocument()
  })

  it('shows a rejection next to the picture it is about', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        verdict({ name: 'IMG_0001.jpg', quality: 'blurred', reason: 'blurred', preview: true }),
      ),
    )

    render(<Frames draft={CALIBRATED} onChange={() => {}} />)
    await addClass('440')
    await userEvent.upload(dropZone(), [jpeg(1)])

    expect(await screen.findByText('borrosa')).toBeInTheDocument()
    expect(screen.getByAltText('Original')).toHaveAttribute(
      'src',
      '/api/drafts/20260830-01/preview/440/IMG_0001.jpg?kind=original',
    )
    expect(screen.getByAltText('Contorns')).toHaveAttribute(
      'src',
      '/api/drafts/20260830-01/preview/440/IMG_0001.jpg?kind=edges',
    )
  })

  it('has no edge map to show for a frame with no markers', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        verdict({ name: 'IMG_0001.jpg', quality: 'partial', reason: 'partial', preview: true }),
      ),
    )

    render(<Frames draft={CALIBRATED} onChange={() => {}} />)
    await addClass('440')
    await userEvent.upload(dropZone(), [jpeg(1)])

    expect(await screen.findByText('marcadors no trobats')).toBeInTheDocument()
    expect(screen.getByAltText('Original')).toBeInTheDocument()
    // Without four markers there is no rectified plate. An edge map here would be showing a
    // rectification that did not happen — and the server answers 404 for it.
    expect(screen.queryByAltText('Contorns')).not.toBeInTheDocument()
  })

  it('labels the class once and sends that label with every photograph', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => verdict()),
    )

    render(<Frames draft={CALIBRATED} onChange={() => {}} />)
    await addClass('440')
    await userEvent.upload(dropZone(), [jpeg(1), jpeg(2), jpeg(3)])

    await waitFor(() => expect(vi.mocked(fetch)).toHaveBeenCalledTimes(3))
    // §9: "the label is typed once per class, never per image". One field for the class, and
    // no second one anywhere near a photograph.
    expect(screen.getAllByLabelText('Freqüència (Hz)')).toHaveLength(1)
    for (const call of vi.mocked(fetch).mock.calls) {
      expect(call[0]).toBe('/api/drafts/20260830-01/frames?freq_hz=440')
    }
  })

  it('sends the File the input handed it, not a copy of it', async () => {
    // The bytes surviving `uploadFrame` are `api/drafts.test.ts`'s assertion. This one is about
    // the path *into* it: the `File` out of the `FileList`, with nothing in between that could
    // produce a re-encoded blob.
    const bitmap = vi.fn()
    vi.stubGlobal('createImageBitmap', bitmap)
    const sent: FormData[] = []
    vi.stubGlobal(
      'fetch',
      vi.fn(async (_url: string, init: RequestInit) => {
        sent.push(init.body as FormData)
        return verdict()
      }),
    )

    render(<Frames draft={CALIBRATED} onChange={() => {}} />)
    await addClass('440')
    await userEvent.upload(dropZone(), [jpeg(7)])

    await waitFor(() => expect(sent).toHaveLength(1))
    const part = sent[0]?.get('file') as File
    expect(part.name).toBe('IMG_7.jpg')
    expect(part.type).toBe('image/jpeg')
    expect(part.size).toBe(1)
    expect(bitmap).not.toHaveBeenCalled()
    expect(document.querySelector('canvas')).toBeNull()
  })

  it('takes a frame back out and tells the wizard the draft changed', async () => {
    const stored: Draft = {
      ...CALIBRATED,
      classes: [440],
      total: 1,
      accepted: 1,
      frames: { '440': [record()] },
    }
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(null, { status: 204 })),
    )
    const onChange = vi.fn()

    render(<Frames draft={stored} onChange={onChange} />)
    await userEvent.click(screen.getByRole('button', { name: 'Treu-la' }))

    expect(vi.mocked(fetch)).toHaveBeenCalledWith(
      '/api/drafts/20260830-01/frames/440/IMG_0001.jpg',
      { method: 'DELETE' },
    )
    await waitFor(() => expect(onChange).toHaveBeenCalled())
  })

  it('shows the classes the draft already has, without being told about them', () => {
    const stored: Draft = {
      ...CALIBRATED,
      classes: [440, 620],
      total: 2,
      accepted: 1,
      rejected: 1,
      frames: {
        '440': [record()],
        '620': [record({ name: 'IMG_0001.jpg', freq_hz: 620, quality: 'blurred', reason: 'blurred' })],
      },
    }

    render(<Frames draft={stored} onChange={() => {}} />)

    expect(screen.getByTestId('accepted-440')).toHaveTextContent('1')
    expect(screen.getByTestId('rejected-620')).toHaveTextContent('1')
  })
})
