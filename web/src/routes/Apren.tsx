/** `/apren` — shell only. F7 builds this (docs/00-master-plan.md). */
import { t } from '../strings'

export function Apren() {
  return (
    <div className="mx-auto max-w-3xl px-4 py-12 sm:px-6">
      <h1 className="text-3xl font-semibold">{t.nav.learn}</h1>
      <p className="mt-4 text-muted">{t.learn.soon}</p>
    </div>
  )
}
