/**
 * The wizard's shell. Two states matter here and neither is the happy one.
 *
 * A draft is server state with a shareable URL, so a stale link is an ordinary thing to hold:
 * the draft was discarded or committed in another tab and this one still has its address.
 * That is a 404, and it has to read as a 404 — telling someone the local server is unreachable
 * sends them to restart a server that answered the question perfectly.
 */

import { render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { DraftWizard } from './DraftWizard'

afterEach(() => vi.unstubAllGlobals())

const DRAFT = {
  draft_id: '20260830-01',
  session_id: '20260830-01',
  status: 'calibrated',
  path: 'C:\\Users\\PC10\\Desktop\\ide1\\drafts\\20260830-01',
  created: '2026-08-30T10:04:11',
  updated: '2026-08-30T10:41:02',
  calibrated: true,
  classes: [440],
  total: 12,
  accepted: 11,
  rejected: 1,
  pending: 0,
  duplicates: 0,
  bytes: 2_400_000_000,
  session: {},
  calibration: null,
  frames: {},
  existing_rows: 0,
}

const EMPTY_MATRIX = {
  sessions: [],
  classes: [],
  cells: {},
  sessions_per_class: {},
  min_sessions_per_class: 3,
}

function renderAt(id: string) {
  return render(
    <MemoryRouter initialEntries={[`/dades/${id}`]}>
      <Routes>
        <Route path="/dades/:draftId" element={<DraftWizard />} />
      </Routes>
    </MemoryRouter>,
  )
}

describe('DraftWizard', () => {
  it('says the draft is gone, not that the server is', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(
        async () =>
          new Response(JSON.stringify({ detail: { code: 'no_draft', message: 'no draft at …' } }), {
            status: 404,
          }),
      ),
    )

    renderAt('20260830-01')

    expect(
      await screen.findByText(/Aquest esborrany ja no existeix/),
    ).toBeInTheDocument()
  })

  it('reports an unreachable server as its own failure', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new TypeError('Failed to fetch')
      }),
    )

    renderAt('20260830-01')

    expect(await screen.findByText(/No s’ha pogut llegir aquest esborrany/)).toBeInTheDocument()
  })

  it('shows the status in Catalan, not as the machine token', async () => {
    // Answer each endpoint with its own shape. A stub that returns a Draft for every URL
    // hands the matrix a Draft, and the component then only stays up because of defaults
    // that exist for the pre-fetch case — a test propping up a component with the wrong
    // data is a test that stops being about what it says it is about.
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string) =>
        url.includes('/api/dataset/matrix')
          ? new Response(JSON.stringify(EMPTY_MATRIX), { status: 200 })
          : new Response(JSON.stringify(DRAFT), { status: 200 }),
      ),
    )

    renderAt('20260830-01')

    expect(await screen.findByText('Calibrada')).toBeInTheDocument()
    expect(screen.queryByText('calibrated')).not.toBeInTheDocument()
  })
})
