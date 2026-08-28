# ADR 0007 — The static multipage site is the frontend; French stays the default

- **Date:** 2026-08-28
- **Status:** accepted. **Amends `docs/05-webapp-spec.md` §4 and `CLAUDE.md` rule 6.**
  Nothing in ADR 0004 or 0005 is touched: the backend, the single preprocessing
  implementation and the CPU-only, localhost-only architecture are unchanged.
- **Trigger:** the repository contained two complete frontends, and the team identified the
  older one — not the one the spec described — as the product they intend to keep.

## Context

Two frontends existed side by side:

| | `web/` | the repo root |
|---|---|---|
| Stack | Vite + React + TypeScript + Tailwind | static multipage HTML, CSS, browser JS |
| Language | Catalan | French, with a `ca`/`es`/`fr` switcher |
| Routes | `/`, `/analitza`, `/apren`, `/galeria`, `/dades`, `/sobre` | `index`, `analyseur`, `experience`, `donnees`, `apropos` |
| State | implements `docs/05` §4, §8, §9, with tests | wired to an `/api/v1` backend deleted from this repo |

`docs/05` §4 described the first. The team asked for the second, and it is the one that
carries the visual identity, the trilingual copy and the camera capture flow. Its analyser
could not work at all: it called `/api/v1/predict` and `/api/v1/model-info`, routes that
vanished with the backend ADR 0004 replaced.

A third fact shaped the decision more than either: **the classes are not chosen yet.**
There is no real `data/modes.csv`, so no reportable checkpoint can exist, and whichever
frontend survives has to accept the real training data later without being edited.

## Decision

**1. The static multipage site is the app's frontend.** FastAPI serves it from `site/`,
same origin, so the CORS question disappears instead of being answered with a wildcard.
`web/` is ported from and then removed; it is not deleted yet, because it is still the only
implementation of the pipeline viewer (§8) and the `/dades` wizard (§9).

**2. French remains the default UI language, and the French filenames are kept.** This
contradicts `CLAUDE.md` rule 6 ("UI strings are Catalan") and §4's Catalan slugs. It is an
explicit team decision, recorded here precisely because the surrounding documents say the
opposite: without this ADR the next person to read them would "fix" the French back to
Catalan believing it a slip. `js/i18n.js` already carries all three languages, so Catalan is
one switch away and no translation was lost.

**3. The class list is generated, never written.** `tasks.py modes` writes
`site/js/classes.generated.js` from `config/classes.json` in the same command that freezes
the class index. Editorial material — descriptions, complexity, mode indices — stays in
`site/js/frequencies.js`, keyed by frequency.

## Consequences

**What this costs.** TypeScript's types and the `vitest` suite go with `web/`; the
replacement is `node --test` over the pure mapping functions and pytest over the API, which
is less coverage of the view layer than React Testing Library gave. The UI is now in a
language `docs/07-glossary-ca-en.md` does not cover, so French physics copy has no glossary
to check it against. And §8's stage titles arrive from the server in Catalan by design,
which will read as Catalan inside a French page until the pipeline viewer is ported.

**What it does not cost.** The honesty requirement survived the reversal and is implemented:
the three-tier result presentation, the geometry warning, and `do_not_report` visible on
every page. Those were the parts of §4 that mattered, and they are now in the static site
rather than only in the spec.

**What made it safe to reverse.** The API is the contract. The frontend adapts to it in one
file, `site/js/adapter.js`; no compatibility shim was added on the server, because a second
response contract maintained alongside `docs/05` §3 is the duplicate this project exists to
avoid. Reversing again — to React, or to anything else — means writing one adapter, not
renegotiating the backend.

## Rejected alternatives

**An `/api/v1` shim in FastAPI.** It would have let the old frontend run untouched. It also
would have created a second, versioned response contract with no owner, drifting from §3 the
first time either side changed. Rejected for the same reason ADR 0004 rejected two
preprocessing implementations.

**Keeping both frontends.** Two implementations of the same six views, both claiming to be
the app, is the condition that produced this ADR in the first place.

**Porting the React app's design into the static site instead of the reverse.** That is not
what the team asked for: the static site's design *is* the thing being preserved.
