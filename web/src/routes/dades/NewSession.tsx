/**
 * Step 1 — the session log. `docs/02-dataset-spec.md` §4, `docs/05-webapp-spec.md` §9.
 *
 * This form is the `session.json` table of `docs/02` §4, control for control, in that order.
 * It is filled in before the first capture and not afterwards, because five of these keys
 * become columns of `metadata.csv` and none of them can be reconstructed from the images:
 * four days after the lab nobody knows what the grain mass was. `tasks.py ingest` refuses a
 * session without a complete log — it does not warn — so the ten seconds spent here is what
 * stands between the afternoon and being unusable.
 *
 * **Numbers leave as numbers.** `grain_mass_g` and `dwell_s` are floats and `distance_mm` is
 * an int in `metadata.csv`; a `"6"` that travelled as a string is not an error anybody sees,
 * it is a column that reads wrong later. The `<input type="number">` gives a string, so the
 * conversion happens here, once, where the body is built.
 *
 * **The session id is not validated here.** `metadata.validate_session_id` refuses a
 * malformed one at `POST /api/drafts`, and its message says what the shape is. A regular
 * expression on this side would be a second home for `docs/02` §2, kept in step by hand, and
 * the day the two disagree the form is the one that is wrong while looking right. So the
 * request is made and the server's sentence is what appears — `error.message`, verbatim,
 * beside its `code`. Inventing a friendlier one would be inventing a claim about a refusal
 * this component did not make.
 *
 * `notes` may be left empty, and that is the point of it: empty means "nothing anomalous",
 * which is itself information (§4). It is submitted as `''` rather than omitted — a missing
 * key is a different statement, and the one ingest refuses.
 */

import { useState } from 'react'
import { useNavigate } from 'react-router'
import { ApiError } from '../../api/client'
import { createDraft } from '../../api/drafts'
import { t } from '../../strings'

/** Everything an `<input>` hands back is a string; §4's types are applied on submit. */
interface FormState {
  session_id: string
  date: string
  operator: string
  plate: string
  material: string
  grain_mass_g: string
  distance_mm: string
  lighting: string
  dwell_s: string
  notes: string
}

const EMPTY: FormState = {
  session_id: '',
  date: '',
  operator: '',
  plate: '',
  // The defaults are the rig's own: `docs/02` §1 shoots at 210 mm, and §4 says the column is
  // "210 unless deliberately varied". Prefilling the usual value is not the same as assuming
  // it — the field is there, and varying the distance means changing a number, not
  // remembering that a number exists.
  material: 'sand',
  grain_mass_g: '',
  distance_mm: '210',
  lighting: 'lamp_diffuse',
  dwell_s: '',
  notes: '',
}

const MATERIALS = ['sand', 'semolina', 'salt'] as const
const LIGHTINGS = ['lamp_diffuse', 'lamp_grazing', 'ambient'] as const

const FIELD =
  'mt-1 w-full rounded-lg border border-border bg-white px-3 py-2 focus:border-primary'

export function NewSession() {
  const navigate = useNavigate()
  const [form, setForm] = useState<FormState>(EMPTY)
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<{ message: string; code: string } | null>(null)

  function set<K extends keyof FormState>(key: K, value: string) {
    setForm((current) => ({ ...current, [key]: value }))
  }

  async function submit(event: React.FormEvent) {
    event.preventDefault()
    setSending(true)
    setError(null)
    try {
      // §4's keys, in §4's order, with §4's types. The identifiers are trimmed because a
      // pasted id carries whatever came with it; `notes` is not, because it is prose and
      // what someone typed is what they meant.
      const draft = await createDraft({
        session_id: form.session_id.trim(),
        date: form.date,
        operator: form.operator.trim(),
        plate: form.plate.trim(),
        material: form.material,
        grain_mass_g: Number(form.grain_mass_g),
        distance_mm: Number(form.distance_mm),
        lighting: form.lighting,
        dwell_s: Number(form.dwell_s),
        notes: form.notes,
      })
      navigate(`/dades/${draft.draft_id}`)
    } catch (failure) {
      setError(
        failure instanceof ApiError
          ? { message: failure.message, code: failure.code }
          : { message: t.data.error.create, code: '' },
      )
      setSending(false)
    }
  }

  return (
    <div className="mx-auto max-w-3xl px-4 py-12 sm:px-6">
      <p className="text-sm text-muted">{t.data.steps.session}</p>
      <h1 className="mt-1 text-3xl font-semibold">{t.data.session.title}</h1>
      <p className="mt-4 text-muted">{t.data.session.lead}</p>

      {error !== null && (
        <div
          role="alert"
          className="mt-6 rounded-lg border border-warn-border bg-warn-surface p-4 text-warn-ink"
        >
          <p>{error.message}</p>
          {error.code !== '' && <p className="mt-1 font-mono text-sm">{error.code}</p>}
        </div>
      )}

      <form onSubmit={(event) => void submit(event)} className="mt-8 space-y-6">
        <div>
          <label htmlFor="session_id" className="font-medium">
            {t.data.session.id}
          </label>
          <input
            id="session_id"
            name="session_id"
            aria-describedby="session_id-hint"
            required
            value={form.session_id}
            onChange={(event) => set('session_id', event.target.value)}
            className={`${FIELD} font-mono`}
          />
          <p id="session_id-hint" className="mt-1 text-sm text-muted">
            {t.data.session.idHint}
          </p>
        </div>

        <div>
          <label htmlFor="date" className="font-medium">
            {t.data.session.date}
          </label>
          <input
            id="date"
            name="date"
            type="date"
            required
            value={form.date}
            onChange={(event) => set('date', event.target.value)}
            className={FIELD}
          />
        </div>

        <div>
          <label htmlFor="operator" className="font-medium">
            {t.data.session.operator}
          </label>
          <input
            id="operator"
            name="operator"
            required
            value={form.operator}
            onChange={(event) => set('operator', event.target.value)}
            className={FIELD}
          />
        </div>

        <div>
          <label htmlFor="plate" className="font-medium">
            {t.data.session.plate}
          </label>
          <input
            id="plate"
            name="plate"
            required
            value={form.plate}
            onChange={(event) => set('plate', event.target.value)}
            className={FIELD}
          />
        </div>

        <div>
          <label htmlFor="material" className="font-medium">
            {t.data.session.material}
          </label>
          <select
            id="material"
            name="material"
            value={form.material}
            onChange={(event) => set('material', event.target.value)}
            className={FIELD}
          >
            {MATERIALS.map((value) => (
              <option key={value} value={value}>
                {t.data.session.materials[value]}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label htmlFor="grain_mass_g" className="font-medium">
            {t.data.session.grainMass}
          </label>
          <input
            id="grain_mass_g"
            name="grain_mass_g"
            type="number"
            step="0.1"
            min="0"
            required
            value={form.grain_mass_g}
            onChange={(event) => set('grain_mass_g', event.target.value)}
            className={FIELD}
          />
        </div>

        <div>
          <label htmlFor="distance_mm" className="font-medium">
            {t.data.session.distance}
          </label>
          <input
            id="distance_mm"
            name="distance_mm"
            type="number"
            step="1"
            min="0"
            required
            value={form.distance_mm}
            onChange={(event) => set('distance_mm', event.target.value)}
            className={FIELD}
          />
        </div>

        <div>
          <label htmlFor="lighting" className="font-medium">
            {t.data.session.lighting}
          </label>
          <select
            id="lighting"
            name="lighting"
            value={form.lighting}
            onChange={(event) => set('lighting', event.target.value)}
            className={FIELD}
          >
            {LIGHTINGS.map((value) => (
              <option key={value} value={value}>
                {t.data.session.lightings[value]}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label htmlFor="dwell_s" className="font-medium">
            {t.data.session.dwell}
          </label>
          <input
            id="dwell_s"
            name="dwell_s"
            type="number"
            step="0.1"
            min="0"
            required
            value={form.dwell_s}
            onChange={(event) => set('dwell_s', event.target.value)}
            className={FIELD}
          />
        </div>

        <div>
          <label htmlFor="notes" className="font-medium">
            {t.data.session.notes}
          </label>
          {/* No `required`: §4 makes the empty value meaningful, so demanding a word here
              would only get one invented. */}
          <textarea
            id="notes"
            name="notes"
            rows={3}
            aria-describedby="notes-hint"
            value={form.notes}
            onChange={(event) => set('notes', event.target.value)}
            className={FIELD}
          />
          <p id="notes-hint" className="mt-1 text-sm text-muted">
            {t.data.session.notesHint}
          </p>
        </div>

        <button
          type="submit"
          disabled={sending}
          aria-busy={sending}
          className="rounded-lg bg-primary px-5 py-2.5 font-medium text-white hover:bg-primary-strong disabled:opacity-60"
        >
          {t.data.session.submit}
        </button>
      </form>
    </div>
  )
}
