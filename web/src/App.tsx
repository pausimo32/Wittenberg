/**
 * Routes and the shell. `docs/05-webapp-spec.md` §4 — Catalan slugs, English code.
 *
 * `/health` is fetched once, here, for one reason: the checkpoint's `do_not_report` has to
 * be visible on **every** view, and someone who opens `/analitza` directly never passes
 * through a home page. Fetching it per route would be three requests for one fact.
 *
 * No `localStorage`, no cookies, no analytics — anywhere, ever (§5). Progress and state
 * live in React and die with the tab, which is what makes the reset between participants
 * real rather than a promise.
 */

import { useEffect, useState } from 'react'
import { NavLink, Route, Routes } from 'react-router'
import type { Health } from './api/types'
import { getHealth } from './api/client'
import { DoNotReportBanner } from './components/Banners'
import { Analitza } from './routes/Analitza'
import { Apren } from './routes/Apren'
import { Dades } from './routes/Dades'
import { DraftWizard } from './routes/dades/DraftWizard'
import { Galeria } from './routes/Galeria'
import { Home } from './routes/Home'
import { Sobre } from './routes/Sobre'
import { t } from './strings'

const NAV = [
  { to: '/', label: t.nav.home, end: true },
  { to: '/analitza', label: t.nav.analyze },
  { to: '/apren', label: t.nav.learn },
  { to: '/galeria', label: t.nav.gallery },
  { to: '/dades', label: t.nav.data },
  { to: '/sobre', label: t.nav.about },
]

export function App() {
  const [health, setHealth] = useState<Health | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    getHealth(controller.signal)
      .then(setHealth)
      .catch(() => setHealth(null))
    return () => controller.abort()
  }, [])

  return (
    <div className="flex min-h-screen flex-col">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:m-2 focus:rounded focus:bg-primary focus:px-3 focus:py-2 focus:text-white"
      >
        {t.app.skipToContent}
      </a>

      <DoNotReportBanner
        message={health?.do_not_report ?? null}
        run={health?.model_run}
      />

      <header className="border-b border-border bg-white">
        <nav
          aria-label={t.nav.home}
          className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3 sm:px-6"
        >
          <span className="text-lg font-semibold text-secondary">{t.app.name}</span>
          <ul className="flex flex-wrap gap-x-5 gap-y-1">
            {NAV.map((item) => (
              <li key={item.to}>
                <NavLink
                  to={item.to}
                  end={item.end}
                  className={({ isActive }) =>
                    isActive
                      ? 'font-medium text-primary underline underline-offset-4'
                      : 'text-text hover:text-primary'
                  }
                >
                  {item.label}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>
      </header>

      <main id="main" className="flex-1">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/analitza" element={<Analitza />} />
          <Route path="/apren" element={<Apren />} />
          <Route path="/galeria" element={<Galeria />} />
          <Route path="/dades" element={<Dades />} />
          {/* `/dades/nova` — the session form — is not routed yet: it arrives with its
              component in Task 14. The link on `/dades` points at it already and dangles
              until then. It is a static segment, which the router ranks above `:draftId`
              wherever it is declared, so adding it cannot shadow this route. */}
          <Route path="/dades/:draftId" element={<DraftWizard />} />
          <Route path="/sobre" element={<Sobre />} />
          <Route path="*" element={<Home />} />
        </Routes>
      </main>

      <footer className="border-t border-border px-4 py-4 text-center text-sm text-muted sm:px-6">
        {health ? (
          <span className="font-mono">
            {health.model_run} · {health.model} · {health.device} · preprocess{' '}
            {health.preprocess_version}
          </span>
        ) : (
          <span>{t.analyze.error.model}</span>
        )}
      </footer>
    </div>
  )
}
