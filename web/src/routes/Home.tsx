/** `/` — §4: "one sentence, one button". Resisting the urge to add a third thing is the design. */
import { Link } from 'react-router'
import { t } from '../strings'

export function Home() {
  return (
    <div className="mx-auto flex max-w-3xl flex-col items-start px-4 py-16 sm:px-6 sm:py-24">
      <h1 className="text-4xl font-semibold text-secondary sm:text-5xl">{t.app.name}</h1>
      <p className="mt-6 max-w-prose text-xl sm:text-2xl">{t.home.sentence}</p>
      <Link
        to="/analitza"
        className="mt-10 rounded-lg bg-primary px-6 py-3 text-lg font-medium text-white hover:bg-primary-strong"
      >
        {t.home.cta}
      </Link>
      <p className="mt-8 text-sm text-muted">{t.analyze.privacy}</p>
    </div>
  )
}
