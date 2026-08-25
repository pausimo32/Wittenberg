/**
 * Step 2 — the calibration, and the door it holds shut.
 *
 * Three properties are asserted here and none of them is "the happy path renders".
 *
 * - **The picture is the server's.** The quad is drawn in Python, on the very array the
 *   homography was computed from, and arrives as a PNG. Redrawing it here from
 *   `calibration.corners` would be a second implementation of the geometry in a second
 *   coordinate space, and on the day the two disagree the drawing is the convincing one and
 *   the wrong one. So: an `<img>` with a `data:image/png;base64,…` source, and no canvas.
 * - **The flags are rendered.** They are the ArUco nominal-corner mismatch of `docs/02` §1.1 —
 *   a marker taped on rotated, or the plate turned in the frame. Both are legitimate and the
 *   calibration is correct either way, which is exactly why the warning is easy to swallow and
 *   worthless three weeks later. It arrives as English prose from Python; it is shown as it is.
 * - **A failed detection is a state, not an error.** Fewer than four markers is `partial` under
 *   §6 and an ordinary outcome of a photograph: it says how many were found, says what to do,
 *   shows the frame it looked at, and offers another go.
 *
 * The last test is about `DraftWizard`, not this component: the photo step stays shut while
 * the draft is uncalibrated. The server refuses `POST /frames` with 409 `not_calibrated` too
 * (Task 8) — this is the same rule said twice on purpose, once where it can be seen and once
 * where it cannot be bypassed, and this test covers only the half that can be seen.
 */

import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { CalibrationInfo, Draft } from '../../api/types'
import { Calibration } from './Calibration'
import { DraftWizard } from './DraftWizard'

afterEach(() => vi.unstubAllGlobals())

const BASE: Draft = {
  draft_id: '20260830-01',
  session_id: '20260830-01',
  status: 'new',
  path: 'C:\\Users\\PC10\\Desktop\\ide1\\drafts\\20260830-01',
  created: '2026-08-30T10:04:11',
  updated: '2026-08-30T10:41:02',
  calibrated: false,
  classes: [],
  total: 0,
  accepted: 0,
  rejected: 0,
  pending: 0,
  duplicates: 0,
  bytes: 0,
  session: {},
  calibration: null,
  frames: {},
  existing_rows: 0,
}

const FOUND: CalibrationInfo = {
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
}

/** A one-pixel PNG, base64, standing in for what the server drew the quad on. */
const PNG = 'iVBORw0KGgoAAAANSUhEUg=='

function frame(): File {
  return new File([new Uint8Array([1, 2, 3])], 'CAL_0001.HEIC', { type: 'image/heic' })
}

describe('Calibration', () => {
  it('will not show the photo step until the four markers are found', () => {
    const draft: Draft = {
      ...BASE,
      calibrated: false,
      calibration: { ...FOUND, ok: false, n_markers: 3, marker_ids: [0, 1, 2], corners: null, px_per_mm: null },
    }

    render(<Calibration draft={draft} onCalibrated={() => {}} />)

    expect(screen.getByText('3 marcadors de 4 detectats')).toBeInTheDocument()
    expect(screen.getByText(/No s’han trobat els quatre marcadors/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Repeteix el calibratge' })).toBeInTheDocument()
  })

  it('shows the detected plate once it is calibrated', () => {
    const draft: Draft = { ...BASE, calibrated: true, calibration: FOUND }

    render(<Calibration draft={draft} onCalibrated={() => {}} />)

    expect(screen.getByText(/Calibratge fet/)).toBeInTheDocument()
    expect(screen.getByText('14.5 px/mm')).toBeInTheDocument()
  })

  it('surfaces the rig warnings rather than swallowing them', () => {
    const draft: Draft = {
      ...BASE,
      calibrated: true,
      calibration: {
        ...FOUND,
        flags: ['marker 2: the corner touching the plate is index 1, nominal is 0'],
      },
    }

    render(<Calibration draft={draft} onCalibrated={() => {}} />)

    expect(screen.getByText('Avisos del muntatge')).toBeInTheDocument()
    expect(screen.getByText(/the corner touching the plate/)).toBeInTheDocument()
  })

  it('shows the server’s drawing and never draws a second one of its own', async () => {
    const sent: FormData[] = []
    const file = frame()
    vi.stubGlobal(
      'fetch',
      vi.fn(async (_url: string, init: RequestInit) => {
        sent.push(init.body as FormData)
        return new Response(JSON.stringify({ calibration: FOUND, preview: PNG }), { status: 200 })
      }),
    )
    const onCalibrated = vi.fn()

    render(<Calibration draft={BASE} onCalibrated={onCalibrated} />)
    await userEvent.upload(
      screen.getByLabelText('Tria un fitxer'),
      file,
    )

    const image = await screen.findByRole('img')
    expect(image).toHaveAttribute('src', `data:image/png;base64,${PNG}`)
    // The quad is the server's drawing or it is nothing: no second geometry in a second
    // coordinate space, however tempting `calibration.corners` looks.
    expect(document.querySelector('canvas')).toBeNull()
    expect(document.querySelector('svg')).toBeNull()
    // The picked file goes into the request as it is. That its *bytes* survive the trip is
    // `drafts.test.ts`'s assertion and stays there — one test for one property, not two.
    const uploaded = sent[0]?.get('file')
    expect(uploaded).toBeInstanceOf(File)
    expect((uploaded as File).name).toBe(file.name)
    expect(onCalibrated).toHaveBeenCalled()
  })

  it('treats a failed detection as a state, not as a failure of the app', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(
        async () =>
          new Response(
            JSON.stringify({
              calibration: { ...FOUND, ok: false, n_markers: 2, marker_ids: [0, 3], corners: null, px_per_mm: null },
              preview: PNG,
            }),
            { status: 200 },
          ),
      ),
    )

    render(<Calibration draft={BASE} onCalibrated={() => {}} />)
    await userEvent.upload(
      screen.getByLabelText('Tria un fitxer'),
      frame(),
    )

    expect(await screen.findByText('2 marcadors de 4 detectats')).toBeInTheDocument()
    expect(screen.getByText(/No s’han trobat els quatre marcadors/)).toBeInTheDocument()
    // The frame it looked at is still worth seeing: that is how you notice the marc is cut off.
    expect(screen.getByRole('img')).toHaveAttribute('src', `data:image/png;base64,${PNG}`)
    expect(screen.getByRole('button', { name: 'Repeteix el calibratge' })).toBeInTheDocument()
  })

  it('shows the server’s refusal rather than guessing at it', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(
        async () =>
          new Response(
            JSON.stringify({
              detail: { code: 'unsupported_image', message: 'CAL_0001.HEIC: cannot be decoded' },
            }),
            { status: 415 },
          ),
      ),
    )

    render(<Calibration draft={BASE} onCalibrated={() => {}} />)
    await userEvent.upload(
      screen.getByLabelText('Tria un fitxer'),
      frame(),
    )

    expect(await screen.findByText(/cannot be decoded/)).toBeInTheDocument()
  })
})

describe('DraftWizard, step 3', () => {
  function renderWizard(draft: Draft) {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(JSON.stringify(draft), { status: 200 })),
    )
    return render(
      <MemoryRouter initialEntries={['/dades/20260830-01']}>
        <Routes>
          <Route path="/dades/:draftId" element={<DraftWizard />} />
        </Routes>
      </MemoryRouter>,
    )
  }

  it('holds the photo step shut while the draft is not calibrated', async () => {
    renderWizard({ ...BASE, calibrated: false })

    expect(
      await screen.findByText('Primer el calibratge. Les fotografies de patrons vénen després.'),
    ).toBeInTheDocument()
  })

  it('opens it once the calibration is in', async () => {
    renderWizard({ ...BASE, calibrated: true, status: 'calibrated', calibration: FOUND })

    await screen.findByText(/Calibratge fet/)
    await waitFor(() =>
      expect(
        screen.queryByText('Primer el calibratge. Les fotografies de patrons vénen després.'),
      ).not.toBeInTheDocument(),
    )
  })
})
