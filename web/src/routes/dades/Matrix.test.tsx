/**
 * The class × session matrix. §9: "the empty cells are the ones that break the split".
 *
 * Which is the whole test. A cell with no photographs in it must not render as `0/0`: a zero
 * is a measurement, and this is the absence of one — the class was never captured in that
 * session. Rendered as `0/0` it sits in a grid of numbers and reads as "few", and the person
 * with the rig still set up walks away from the one thing this table exists to show them.
 *
 * The second assertion is the coverage warning, and it is here rather than in the curator
 * because the curator runs before training, which is four days too late to point a camera.
 */

import { render, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { Matrix as MatrixData } from '../../api/types'
import { Matrix } from './Matrix'

afterEach(() => vi.unstubAllGlobals())

/** 440 Hz was captured in all three sessions; 620 Hz only in the first. That hole is the
 *  point: 620 Hz cannot end up in a test set that is not also its training set. */
const HOLED: MatrixData = {
  sessions: ['20260830-01', '20260901-01', '20260903-01'],
  classes: [440, 620],
  cells: {
    '440': {
      '20260830-01': { total: 60, accepted: 57 },
      '20260901-01': { total: 58, accepted: 51 },
      '20260903-01': { total: 61, accepted: 55 },
    },
    '620': { '20260830-01': { total: 60, accepted: 60 } },
  },
  sessions_per_class: { '440': 3, '620': 1 },
  min_sessions_per_class: 3,
}

function serve(body: unknown, status = 200) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => new Response(JSON.stringify(body), { status })),
  )
}

describe('Matrix', () => {
  it('renders a missing cell as visibly empty, never as 0/0', async () => {
    serve(HOLED)

    render(<Matrix />)

    expect(await screen.findByText('57/60')).toBeInTheDocument()
    expect(screen.getByText('60/60')).toBeInTheDocument()
    expect(screen.queryByText('0/0')).toBeNull()

    const row = screen.getByRole('row', { name: /620 Hz/ })
    const empty = within(row).getAllByText('—')
    expect(empty).toHaveLength(2)
    expect(empty[0]).toHaveAttribute('title')
  })

  it('warns about a class fewer sessions cover than the minimum', async () => {
    serve(HOLED)

    render(<Matrix />)

    expect(
      await screen.findByText('620 Hz només apareix en 1 sessió; en calen 3.'),
    ).toBeInTheDocument()
    expect(screen.queryByText(/440 Hz només apareix/)).toBeNull()
  })

  it('scrolls sideways rather than squeezing ten classes into a narrow window', async () => {
    serve(HOLED)

    render(<Matrix />)

    const table = await screen.findByRole('table')
    expect(table.parentElement?.className).toContain('overflow-x-auto')
  })

  it('says so when nothing has been committed yet', async () => {
    serve({
      sessions: [],
      classes: [],
      cells: {},
      sessions_per_class: {},
      min_sessions_per_class: 3,
    } satisfies MatrixData)

    render(<Matrix />)

    expect(await screen.findByText('Encara no hi ha cap sessió incorporada.')).toBeInTheDocument()
    expect(screen.queryByRole('table')).toBeNull()
  })
})
