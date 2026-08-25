/** `/galeria` — shell only. F7 builds this (docs/00-master-plan.md). */
import { t } from '../strings'

export function Galeria() {
  return (
    <div className="mx-auto max-w-3xl px-4 py-12 sm:px-6">
      <h1 className="text-3xl font-semibold">{t.nav.gallery}</h1>
      <p className="mt-4 text-muted">{t.gallery.soon}</p>
    </div>
  )
}
