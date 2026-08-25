/**
 * Step 1 — the session log. `docs/02-dataset-spec.md` §4, `docs/05-webapp-spec.md` §9.
 *
 * What is asserted here is the shape of what leaves the browser, because that is the part
 * nobody can repair afterwards. Five of these keys become columns of `metadata.csv` and none
 * of them can be reconstructed from the photographs: four days after the lab nobody knows
 * what the grain mass was. A number that leaves as `"6"` instead of `6` is not a type error
 * anyone sees — it is a `metadata.csv` column that reads wrong later.
 *
 * The other half is the refusal. The session id's rules live in `metadata.validate_session_id`
 * and nowhere else, so the form does not check them: it sends what was typed and shows what
 * the server answered. That is why the second test looks for the server's own sentence.
 */

import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { NewSession } from './NewSession'

afterEach(() => vi.unstubAllGlobals())

/** The §4 table, in the order it is written there. The form is that table. */
const SESSION_KEYS = [
  'session_id',
  'date',
  'operator',
  'plate',
  'material',
  'grain_mass_g',
  'distance_mm',
  'lighting',
  'dwell_s',
  'notes',
]

function renderForm() {
  return render(
    <MemoryRouter initialEntries={['/dades/nova']}>
      <Routes>
        <Route path="/dades/nova" element={<NewSession />} />
        {/* Where a created draft lands. Standing in for the wizard so the navigation is
            observable without pulling its fetches into this test. */}
        <Route path="/dades/:draftId" element={<p>obert</p>} />
      </Routes>
    </MemoryRouter>,
  )
}

async function fillTheForm() {
  await userEvent.type(screen.getByLabelText('Identificador de la sessió'), '20260830-01')
  await userEvent.type(screen.getByLabelText('Data'), '2026-08-30')
  await userEvent.type(screen.getByLabelText('Qui ha fet la captura'), 'PS')
  await userEvent.type(screen.getByLabelText('Placa'), 'alu-200')
  await userEvent.clear(screen.getByLabelText('Massa de gra (g)'))
  await userEvent.type(screen.getByLabelText('Massa de gra (g)'), '6')
  await userEvent.clear(screen.getByLabelText('Distància de la càmera (mm)'))
  await userEvent.type(screen.getByLabelText('Distància de la càmera (mm)'), '210')
  await userEvent.clear(screen.getByLabelText('Temps d’estabilització (s)'))
  await userEvent.type(screen.getByLabelText('Temps d’estabilització (s)'), '3')
  // `Material granular`, `Il·luminació` and `Notes` keep their defaults: sand, lamp_diffuse
  // and the empty string that means "nothing anomalous".
}

function submit() {
  return userEvent.click(screen.getByRole('button', { name: 'Comença la sessió' }))
}

describe('NewSession', () => {
  it('sends every session.json key and navigates to the draft', async () => {
    const posted: Record<string, unknown>[] = []
    vi.stubGlobal(
      'fetch',
      vi.fn(async (_url: string, init: RequestInit) => {
        posted.push(JSON.parse(String(init.body)) as Record<string, unknown>)
        return new Response(JSON.stringify({ draft_id: '20260830-01' }), { status: 200 })
      }),
    )

    renderForm()
    await fillTheForm()
    await submit()

    expect(posted[0]).toMatchObject({
      session_id: '20260830-01',
      date: '2026-08-30',
      operator: 'PS',
      plate: 'alu-200',
      material: 'sand',
      grain_mass_g: 6,
      distance_mm: 210,
      lighting: 'lamp_diffuse',
      dwell_s: 3,
      notes: '',
    })
    // The whole key set, in §4's order. A key that grows there has to grow here too — the
    // point of this line is that it cannot grow in only one of the two places.
    expect(Object.keys(posted[0] ?? {})).toEqual(SESSION_KEYS)
    expect(await screen.findByText('obert')).toBeInTheDocument()
  })

  it('leaves the nominal distance alone when nobody changes it', async () => {
    const posted: Record<string, unknown>[] = []
    vi.stubGlobal(
      'fetch',
      vi.fn(async (_url: string, init: RequestInit) => {
        posted.push(JSON.parse(String(init.body)) as Record<string, unknown>)
        return new Response(JSON.stringify({ draft_id: '20260830-01' }), { status: 200 })
      }),
    )

    renderForm()
    await userEvent.type(screen.getByLabelText('Identificador de la sessió'), '20260830-01')
    await userEvent.type(screen.getByLabelText('Data'), '2026-08-30')
    await userEvent.type(screen.getByLabelText('Qui ha fet la captura'), 'PS')
    await userEvent.type(screen.getByLabelText('Placa'), 'alu-200')
    await userEvent.type(screen.getByLabelText('Massa de gra (g)'), '6')
    await userEvent.type(screen.getByLabelText('Temps d’estabilització (s)'), '3')
    await submit()

    // `docs/02` §4: "210 unless deliberately varied". Nobody varied it.
    expect(posted[0]?.distance_mm).toBe(210)
  })

  it('sends the empty notes rather than dropping the key', async () => {
    const posted: Record<string, unknown>[] = []
    vi.stubGlobal(
      'fetch',
      vi.fn(async (_url: string, init: RequestInit) => {
        posted.push(JSON.parse(String(init.body)) as Record<string, unknown>)
        return new Response(JSON.stringify({ draft_id: '20260830-01' }), { status: 200 })
      }),
    )

    renderForm()
    await fillTheForm()
    await submit()

    // §4: `notes` is "required to be present; may be empty" — empty means "nothing
    // anomalous", which is itself information, and a missing key is not the same claim.
    expect(posted[0]).toHaveProperty('notes', '')
  })

  it('shows the server’s refusal rather than guessing at it', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(
        async () =>
          new Response(
            JSON.stringify({
              detail: { code: 'draft_exists', message: 'already open at drafts/20260830-01' },
            }),
            { status: 409 },
          ),
      ),
    )

    renderForm()
    await fillTheForm()
    await submit()

    expect(await screen.findByText(/already open/)).toBeInTheDocument()
    // The code is the half a person can search for, and the half the log agrees with.
    expect(screen.getByText('draft_exists')).toBeInTheDocument()
    expect(screen.queryByText('obert')).not.toBeInTheDocument()
  })

  it('does not judge the session id itself', async () => {
    const fetchMock = vi.fn(
      async () =>
        new Response(
          JSON.stringify({
            detail: { code: 'invalid_session_log', message: 'session_id: expected YYYYMMDD-NN' },
          }),
          { status: 422 },
        ),
    )
    vi.stubGlobal('fetch', fetchMock)

    renderForm()
    await userEvent.type(screen.getByLabelText('Identificador de la sessió'), 'la sessió del dijous')
    await userEvent.type(screen.getByLabelText('Data'), '2026-08-30')
    await userEvent.type(screen.getByLabelText('Qui ha fet la captura'), 'PS')
    await userEvent.type(screen.getByLabelText('Placa'), 'alu-200')
    await userEvent.type(screen.getByLabelText('Massa de gra (g)'), '6')
    await userEvent.type(screen.getByLabelText('Temps d’estabilització (s)'), '3')
    await submit()

    // A malformed id is refused by `metadata.validate_session_id`, whose message says what
    // the shape is. A regexp here would be a second home for `docs/02` §2, and the two would
    // drift — so the request has to be made, not withheld.
    expect(fetchMock).toHaveBeenCalled()
    expect(await screen.findByText(/expected YYYYMMDD-NN/)).toBeInTheDocument()
  })
})
