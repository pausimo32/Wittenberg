/**
 * `docs/05-webapp-spec.md` §9 — the entry point of the labelling surface.
 *
 * Two of the round's requirements are asserted here and nowhere else on this side:
 * requirement 2, that a draft survives a reload because it lives on the server and its
 * directory is printed where a person can read it; and requirement 9, that the listing
 * carries enough — path, counts, size — for someone with three open drafts and a full disk
 * to know which one to delete.
 *
 * The discard confirmation is a panel rather than `window.confirm` on purpose. `confirm` is
 * unassertable from a test, and it cannot carry the one sentence the confirmation exists to
 * say: the dataset is not touched by this.
 */

import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { Dades } from './Dades'

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
}

function stubDrafts(drafts: unknown[]) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (_url: string, init?: RequestInit) => {
      if (init?.method === 'DELETE') return new Response(null, { status: 204 })
      return new Response(JSON.stringify({ drafts }), { status: 200 })
    }),
  )
}

function show() {
  render(
    <MemoryRouter>
      <Dades />
    </MemoryRouter>,
  )
}

afterEach(() => vi.unstubAllGlobals())

describe('/dades', () => {
  it('lists open drafts with their identifier and their path', async () => {
    // Requirement 2: an orphan draft in a directory nobody can find is an afternoon that
    // looks lost even when it is not.
    stubDrafts([DRAFT])

    show()

    expect(await screen.findByText('20260830-01')).toBeInTheDocument()
    expect(screen.getByText(DRAFT.path)).toBeInTheDocument()
    expect(screen.getByText('11 acceptades · 1 rebutjades')).toBeInTheDocument()
    // Requirement 9: three open drafts are a few gigabytes and one of them has to go.
    expect(screen.getByText('2.4 GB')).toBeInTheDocument()
  })

  it('offers each draft as a link to its own address', async () => {
    // The draft is server state keyed by session id, so its address is pasteable and a
    // reload lands back on the same session.
    stubDrafts([DRAFT])

    show()

    expect(await screen.findByRole('link', { name: 'Continua' })).toHaveAttribute(
      'href',
      '/dades/20260830-01',
    )
  })

  it('says so when there is nothing open', async () => {
    stubDrafts([])

    show()

    expect(await screen.findByText('No hi ha cap esborrany obert.')).toBeInTheDocument()
  })

  it('flags a draft whose commit was interrupted', async () => {
    // `committing` is what a commit that died inside the ingest leaves behind: the captures
    // are staged and there is no session yet, which is recoverable but only if it is said.
    stubDrafts([{ ...DRAFT, status: 'committing' }])

    show()

    expect(await screen.findByText(/La incorporació d’aquest esborrany es va interrompre/))
      .toBeInTheDocument()
  })

  it('asks before discarding, and says data/ will not be touched', async () => {
    stubDrafts([DRAFT])
    show()
    await screen.findByText('20260830-01')

    await userEvent.click(screen.getByRole('button', { name: 'Descarta' }))

    expect(screen.getByText(/El conjunt de dades no es tocarà/)).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Descarta' }))
    await waitFor(() => expect(screen.queryByText('20260830-01')).not.toBeInTheDocument())
  })

  it('leaves the draft alone when the confirmation is cancelled', async () => {
    stubDrafts([DRAFT])
    show()
    await screen.findByText('20260830-01')

    await userEvent.click(screen.getByRole('button', { name: 'Descarta' }))
    await userEvent.click(screen.getByRole('button', { name: 'Cancel·la' }))

    expect(screen.getByText('20260830-01')).toBeInTheDocument()
    expect(screen.queryByText(/El conjunt de dades no es tocarà/)).not.toBeInTheDocument()
  })
})
