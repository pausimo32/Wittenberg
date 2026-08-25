/**
 * Images per class per session. `docs/05-webapp-spec.md` §9.
 *
 * §9 lists five things the dataset's dashboard should show and this is the only one in this
 * round, because it is the one step 5 cannot do without: "images per class per session, as a
 * matrix — **the empty cells are the ones that break the split**". The split is by session
 * (hard rule 3), so a class captured in a single afternoon cannot have a test frame that is
 * not also a train frame, and the honest thing to do with it is not to report it at all.
 *
 * **An empty cell is drawn empty.** Not `0/0`. A zero is a measurement — sixty photographs
 * were taken and none survived the gate — and a blank is the absence of one: nobody pointed
 * a camera at that class that day. They lead to opposite actions, and `0/0` sitting in a grid
 * of numbers reads as "few" rather than as "none". The em dash carries a `title` so the
 * distinction is available to a reader who is not sure what the blank means.
 *
 * **It is here rather than only in `wittenberg-dataset-curator`** for the reason §9 gives:
 * the curator runs before training, and by then the rig has been packed away for four days.
 * This table is the same numbers, on the screen where there is still something to be done
 * about them.
 */

import { useEffect, useState } from 'react'
import { getMatrix } from '../../api/drafts'
import type { Matrix as MatrixData } from '../../api/types'
import { t } from '../../strings'

export function Matrix() {
  const [matrix, setMatrix] = useState<MatrixData | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    getMatrix(controller.signal)
      .then((answer) => setMatrix(answer))
      .catch(() => {
        // An aborted request is this component being left, not a failure to report.
        if (!controller.signal.aborted) setError(t.analyze.error.network)
      })
    return () => controller.abort()
  }, [])

  // Read out with defaults rather than behind `matrix !== null` in six places: the answer is
  // either absent or whole, and giving each field its empty value says that once.
  const classes = matrix?.classes ?? []
  const sessions = matrix?.sessions ?? []
  const cells = matrix?.cells ?? {}
  const perClass = matrix?.sessions_per_class ?? {}
  const needed = matrix?.min_sessions_per_class ?? 0

  /** The classes too few sessions cover. A class in one session is a class whose test
   *  accuracy is a fact about one afternoon. */
  const short = classes.filter((freq) => (perClass[String(freq)] ?? 0) < needed)

  return (
    <section className="mt-10 border-t border-border pt-8">
      <h2 className="text-xl font-semibold">{t.data.matrix.title}</h2>
      <p className="mt-1 text-muted">{t.data.matrix.lead}</p>

      {error !== null && (
        <p
          role="alert"
          className="mt-4 rounded-lg border border-warn-border bg-warn-surface p-4 text-warn-ink"
        >
          {error}
        </p>
      )}

      {matrix !== null && classes.length === 0 && <p className="mt-4">{t.data.matrix.empty}</p>}

      {classes.length > 0 && (
        // Ten classes by six sessions does not fit a narrow window, and a table that overflows
        // its container silently is a table whose last two sessions nobody knows about.
        <div className="mt-4 overflow-x-auto">
          <table className="min-w-full border-collapse text-sm">
            <thead>
              <tr>
                <th scope="col" className="border-b border-border px-3 py-2 text-left" />
                {sessions.map((session) => (
                  <th
                    key={session}
                    scope="col"
                    className="border-b border-border px-3 py-2 text-left font-mono font-medium whitespace-nowrap"
                  >
                    {session}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {classes.map((freq) => {
                const row = cells[String(freq)] ?? {}
                return (
                  <tr key={freq}>
                    <th
                      scope="row"
                      className="border-b border-border px-3 py-2 text-left font-mono font-medium whitespace-nowrap"
                    >
                      {freq} Hz
                    </th>
                    {sessions.map((session) => {
                      const cell = row[session]
                      return (
                        <td key={session} className="border-b border-border px-3 py-2 font-mono">
                          {cell === undefined ? (
                            <span title={t.data.matrix.emptyCell} className="text-muted">
                              —
                            </span>
                          ) : (
                            t.data.matrix.cell(cell.accepted, cell.total)
                          )}
                        </td>
                      )
                    })}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      {short.length > 0 && (
        <ul className="mt-4 space-y-2">
          {short.map((freq) => (
            <li
              key={freq}
              className="rounded-lg border border-warn-border bg-warn-surface p-3 text-warn-ink"
            >
              {t.data.matrix.sessionsShort(freq, perClass[String(freq)] ?? 0, needed)}
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
