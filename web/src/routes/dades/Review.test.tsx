/**
 * Steps 5 and 6 — the confirmation, the progress, and the split.
 *
 * Four properties are asserted here, and none of them is "the button calls the endpoint".
 *
 * - **The commit is confirmed, not performed.** Requirement 4 lives in the server (409
 *   `session_exists`), but the half a person actually meets is this one: they are told the
 *   session already has 240 rows and are asked a second time, with a *different* button. A
 *   `mixed_provenance` refusal gets no such button, because there is no override for it —
 *   offering one that answers 409 forever would be worse than offering none.
 * - **Progress, not a spinner.** Six hundred captures is minutes. A spinner says "wait"; a
 *   number says "wait about this long, and it is moving". The test watches the number change
 *   rather than watching a request be made.
 * - **An interrupted commit is resumable, and the button says so by staying pressable.**
 *   Disabling it would leave discarding the afternoon as the only way forward.
 * - **Rejected images are a reported result** (§9), so the summary carries the yield, the
 *   reasons with their counts, and the duplicates — the number that says a batch was
 *   uploaded twice.
 *
 * The fetch stub routes on the URL rather than on call order: `<Matrix />` renders underneath
 * the review always, so there is a second endpoint in flight and a stub that counted calls
 * would be asserting the order two independent effects happened to run in.
 */

import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { CommitStatus, Draft, FrameRecord, Matrix, SplitResult } from '../../api/types'
import { Review } from './Review'

afterEach(() => vi.unstubAllGlobals())

function frame(
  name: string,
  freq: number,
  quality: FrameRecord['quality'],
  reason: string | null,
): FrameRecord {
  return {
    name,
    original_filename: `IMG_47${name.slice(4, 8)}.HEIC`,
    freq_hz: freq,
    quality,
    reason,
    metrics: {},
    sha256: `${freq}-${name}`,
    preview: quality !== 'ok',
  }
}

/** A draft with everything uploaded: nine accepted, three refused, two bytes-identical. */
const READY: Draft = {
  draft_id: '20260830-01',
  session_id: '20260830-01',
  status: 'calibrated',
  path: 'C:\\Users\\PC10\\Desktop\\ide1\\drafts\\20260830-01',
  created: '2026-08-30T10:04:11',
  updated: '2026-08-30T10:41:02',
  calibrated: true,
  classes: [440, 620],
  total: 12,
  accepted: 9,
  rejected: 3,
  pending: 0,
  duplicates: 2,
  bytes: 24_000_000,
  session: {},
  calibration: null,
  frames: {
    '440': [
      frame('IMG_0001.jpg', 440, 'ok', null),
      frame('IMG_0002.jpg', 440, 'ok', null),
      frame('IMG_0003.jpg', 440, 'ok', null),
      frame('IMG_0004.jpg', 440, 'ok', null),
      frame('IMG_0005.jpg', 440, 'ok', null),
      frame('IMG_0006.jpg', 440, 'blurred', 'blurred'),
    ],
    '620': [
      frame('IMG_0001.jpg', 620, 'ok', null),
      frame('IMG_0002.jpg', 620, 'ok', null),
      frame('IMG_0003.jpg', 620, 'ok', null),
      frame('IMG_0004.jpg', 620, 'ok', null),
      frame('IMG_0005.jpg', 620, 'partial', 'partial'),
      frame('IMG_0006.jpg', 620, 'partial', 'partial'),
    ],
  },
  existing_rows: 0,
}

const IDLE: CommitStatus = {
  status: 'idle',
  phase: null,
  done: 0,
  total: 0,
  report: null,
  error: null,
}

const RUNNING: CommitStatus = {
  status: 'running',
  phase: 'ingest',
  done: 40,
  total: 600,
  report: null,
  error: null,
}

const DONE: CommitStatus = {
  status: 'done',
  phase: 'finished',
  done: 600,
  total: 600,
  error: null,
  report: {
    session_id: '20260830-01',
    total: 600,
    accepted: 588,
    rejected: 12,
    by_reason: { blurred: 12 },
    by_class: {},
    roi_mode: 'homography',
    flags: [],
  },
}

const NO_MATRIX: Matrix = {
  sessions: [],
  classes: [],
  cells: {},
  sessions_per_class: {},
  min_sessions_per_class: 3,
}

const SPLIT: SplitResult = {
  seed: 42,
  moved: 12,
  sessions: { train: ['20260830-01'], val: [], test: [] },
  images: { train: 420, val: 84, test: 96 },
  rejected: 12,
}

function ok(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status })
}

function refusal(code: string, extra: Record<string, unknown> = {}): Response {
  return new Response(
    JSON.stringify({ detail: { code, message: 'the log reads this, in English', ...extra } }),
    { status: 409 },
  )
}

interface Handlers {
  commit?: () => Response
  onCommit?: () => Response
  split?: () => Response
  matrix?: () => Response
}

/**
 * A stub server. `commit` answers the GET, `onCommit` the POST; the matrix is answered for
 * everybody, because `<Matrix />` is underneath the review always.
 */
function server(handlers: Handlers): string[] {
  const posted: string[] = []
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (init?.method === 'POST') posted.push(url)
      if (url.includes('/api/dataset/matrix')) return (handlers.matrix ?? (() => ok(NO_MATRIX)))()
      if (url.includes('/api/dataset/split')) return (handlers.split ?? (() => ok(SPLIT)))()
      if (url.includes('/commit')) {
        if (init?.method === 'POST') return (handlers.onCommit ?? (() => ok(RUNNING, 202)))()
        return (handlers.commit ?? (() => ok(IDLE)))()
      }
      throw new Error(`unexpected request to ${url}`)
    }),
  )
  return posted
}

const COMMIT = 'Incorpora al conjunt de dades'
const REPLACE = 'Substitueix aquella sessió'

/** A commit that has finished, reached by pressing the button rather than by assumption. */
function committed(): Handlers {
  let started = false
  return {
    onCommit: () => {
      started = true
      return ok(RUNNING, 202)
    },
    commit: () => ok(started ? DONE : IDLE),
  }
}

describe('Review', () => {
  it('asks before replacing a session that already has rows', async () => {
    // Requirement 4, on the side the user sees.
    server({ onCommit: () => refusal('session_exists', { existing_rows: 240 }) })

    render(<Review draft={READY} />)
    await userEvent.click(screen.getByRole('button', { name: COMMIT }))

    expect(await screen.findByText(/ja té 240 files/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: REPLACE })).toBeInTheDocument()
  })

  it('sends replace=true only from the second, explicit button', async () => {
    const posted = server({ onCommit: () => refusal('session_exists', { existing_rows: 240 }) })

    render(<Review draft={READY} />)
    await userEvent.click(screen.getByRole('button', { name: COMMIT }))
    await userEvent.click(await screen.findByRole('button', { name: REPLACE }))

    expect(posted[0]).toContain('replace=false')
    expect(posted[1]).toContain('replace=true')
  })

  it('offers no override for a mixed-provenance dataset, because there is not one', async () => {
    server({ onCommit: () => refusal('mixed_provenance') })

    render(<Review draft={READY} />)
    await userEvent.click(screen.getByRole('button', { name: COMMIT }))

    expect(await screen.findByText(/barreja sessions reals i sintètiques/)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: REPLACE })).toBeNull()
  })

  it('shows the commit advancing rather than a spinner', async () => {
    // The POST answers `running`; the first poll answers `running` again and the second
    // answers `done`. What is asserted is a number that moved, not a request that was made.
    let started = false
    let polls = 0
    server({
      onCommit: () => {
        started = true
        return ok(RUNNING, 202)
      },
      commit: () => {
        if (!started) return ok(IDLE)
        polls += 1
        return ok(polls > 1 ? DONE : RUNNING)
      },
    })

    render(<Review draft={READY} />)
    await userEvent.click(screen.getByRole('button', { name: COMMIT }))

    expect(await screen.findByText('Incorporant… 40 de 600 fotografies')).toBeInTheDocument()
    expect(
      await screen.findByText('Sessió incorporada.', undefined, { timeout: 4000 }),
    ).toBeInTheDocument()
  })

  it('names the phase before ingest starts, so the staging minute is not a dead screen', async () => {
    const staging: CommitStatus = { ...RUNNING, phase: 'stage', done: 0, total: 600 }
    server({ onCommit: () => ok(staging, 202), commit: () => ok(staging) })

    render(<Review draft={READY} />)
    await userEvent.click(screen.getByRole('button', { name: COMMIT }))

    expect(await screen.findByText('Desant les captures a part…')).toBeInTheDocument()
  })

  it('leaves the commit button pressable after an interrupted commit', async () => {
    // The captures are staged and no session exists yet. Disabling the button here would
    // leave discarding the afternoon as the only way forward.
    const interrupted: CommitStatus = {
      ...IDLE,
      status: 'interrupted',
      error: { type: 'Interrupted', message: 'the commit of 20260830-01 was interrupted' },
    }
    server({ commit: () => ok(interrupted) })

    render(<Review draft={READY} />)

    expect(await screen.findByText(/es va interrompre/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: COMMIT })).not.toBeDisabled()
  })

  it('reports the yield, the reasons and the duplicates — they are results, not rubbish', async () => {
    server({})

    render(<Review draft={READY} />)

    expect(screen.getByText(/9 \/ 12/)).toBeInTheDocument()
    expect(screen.getByText(/75 %/)).toBeInTheDocument()
    expect(screen.getByTestId('reason-blurred')).toHaveTextContent('1')
    expect(screen.getByTestId('reason-partial')).toHaveTextContent('2')
    expect(screen.getByText('borrosa')).toBeInTheDocument()
    expect(screen.getByText('2 fotografies duplicades no s’han desat.')).toBeInTheDocument()
    expect(screen.getByTestId('class-440')).toHaveTextContent('5/6')
    expect(screen.getByTestId('class-620')).toHaveTextContent('4/6')

    // The summary above is rendered from the draft alone, so nothing was awaited to get it.
    // The matrix underneath is still in flight; letting it land keeps its state update inside
    // the test rather than after it.
    await screen.findByText('Encara no hi ha cap sessió incorporada.')
  })

  it('says a re-labelling needs a new split, then shows the split’s own numbers', async () => {
    server(committed())

    render(<Review draft={READY} />)
    await userEvent.click(screen.getByRole('button', { name: COMMIT }))

    expect(
      await screen.findByText(/cal refer la partició/, undefined, { timeout: 4000 }),
    ).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Refés la partició' }))

    expect(
      await screen.findByText('Partició feta: 420 entrenament · 84 validació · 96 prova.'),
    ).toBeInTheDocument()
  })

  it('says why the split cannot run yet', async () => {
    server({ ...committed(), split: () => refusal('split_failed') })

    render(<Review draft={READY} />)
    await userEvent.click(screen.getByRole('button', { name: COMMIT }))
    await userEvent.click(
      await screen.findByRole('button', { name: 'Refés la partició' }, { timeout: 4000 }),
    )

    expect(await screen.findByText(/Calen almenys tres sessions/)).toBeInTheDocument()
  })
})
