# Reconnecting the static site to the real pipeline (A+B)

Status: approved 2026-08-28. Scope: sub-projects **A** (analyser reconnected) and **B**
(serving and shell). Sub-projects C (pipeline viewer), D (`/dades` wizard) and E
(`/apren`, `/galeria`, `/sobre`) get their own specs and are explicitly out of scope here.

## 1. Why

Two complete frontends exist. `web/` is a Vite + React implementation of
`docs/05-webapp-spec.md`; the static multipage site at the repo root is the one the team
considers the real product, and it is the one that must be adapted. It is currently wired
to a `/api/v1` backend that was deleted from this repo, so its analyser cannot work at all.

The team has decided: **the static site becomes the only frontend.** `web/` is ported from,
then removed — see §9 for the sequencing, which is not "delete it now".

## 2. The constraint that shapes everything: the classes are not chosen yet

There is no real `data/modes.csv`. The CNN cannot be trained on anything but
`tools/synth_capture.py` output, so every number the app can display today is meaningless
(`docs/00-master-plan.md` F2.5). The work must therefore leave the repo in a state where
introducing the real training data touches **no frontend code at all** — see §7.

This is the acceptance criterion for A+B, and §8 makes it testable today, before the data
exists.

## 3. Where the adaptation lives

In `site/js/predictionService.js`, which is already exactly that layer. The rejected
alternative was a `/api/v1` compatibility shim inside FastAPI: that would create a second
response contract to maintain alongside `docs/05` §3, which is the duplicate the project
forbids. **The API is the source of truth and the frontend adapts to it.**

Consequence: the rendering code in `analyseur.html` keeps receiving the object shape it
already renders, so the diff there stays confined to the result panel.

| Old expectation | Real API |
|---|---|
| `POST /api/v1/predict`, field `image` | `POST /api/predict?stages=false`, field `file` |
| `GET /api/v1/model-info` | `GET /api/health` |
| `body.prediction.{frequency_hz,confidence,reliable,reason,margin}` | derived from `top3[0]` |
| `body.top_k[].{frequency_hz,probability}` | `top3[].{freq_hz,p}` |
| `body.model.{name,version}` | `model`, `model_run` |
| `body.quality.warnings[]` | `quality` — new, see §5 |
| `entry.label` | synthesised from `freq_hz` — the API sends no labels |

## 4. Serving: same origin, and only the site

FastAPI serves the static site. Same origin removes the CORS problem entirely rather than
answering it with a wildcard, which `src/api.py` refuses by design.

**The site moves to `site/`** (`git mv`). This is a correction to the approved design, made
when the mount was made concrete: the site currently lives at the repo root, and mounting
the repo root at `/` would publish `data/raw/`, `data/processed/`, `runs/`, `src/`,
`.venv/` and `.git/` over HTTP. Binding to `127.0.0.1` is not a reason to serve them.
Filenames and URLs are unchanged — `/analyseur.html` stays `/analyseur.html`.

`_SinglePageFiles` goes away with `web/dist`. It falls back to `index.html` for
extensionless paths, which is right for a history-API router and wrong for a multipage site
where `/analyseur.html` is a real file and a missing page must stay a 404. The replacement
is a plain `StaticFiles(html=True)` mount. The existing rule that nothing under `/api` may
be answered with HTML is preserved.

`site/js/config.js` becomes same-origin: an empty API base and no version prefix.

## 5. Backend changes

Two, and neither touches `src/preprocess.py`.

1. **The static mount** (§4).
2. **`quality` on `/api/predict`.** The old result panel has image-quality warnings and the
   API returns nothing to feed them. `src/quality.py::assess` — *the same function*
   `tasks.py ingest` calls, not a second one — produces `{quality, reason, metrics}`.
   The critical difference from ingest: here it **warns and never rejects**. A blurred photo
   is still predicted and still reported as blurred. Rejecting it would invent a policy the
   analyser never had.

`?stages=false` is used for the edge map: A needs only the final stage (≈29 ms rather than
≈82–129 ms). The full thirteen-stage viewer is sub-project C.

## 6. The result panel

The single-number headline is replaced by the three tiers of `docs/05` §4 — the decision
that section calls "still the most important UI decision in the project" — plus two states
the old panel never had:

- **`do_not_report` banner**, on every page, not just the analyser. Hung on the API field,
  so it vanishes by itself when a real checkpoint arrives.
- **Geometry warning** (`docs/05` §4): a `roi_mode` other than `homography` means the ArUco
  frame was not in the photo and the image was centre-cropped into a geometry the model
  never trained on. Its own visible state, above the confidence tiers and independent of
  them.

Probabilities are truncated toward zero, never rounded up: 0.699 must not read as "70 %"
under a headline that says "podria ser".

Kept unchanged: camera capture, drag-and-drop, the `ca`/`es`/`fr` switcher with French as
the default, alternative hypotheses, editorial descriptions, the low-confidence and
ambiguity warnings, the model badge, and all eight error codes.

## 7. Class metadata is generated, never hand-written

The old site assumes the class list in three places, and all three break when the real
classes are chosen:

| Where | Assumption | Failure |
|---|---|---|
| `js/frequencies.js` | 13 hand-written frequencies, 330–2640 Hz | none match; no description, no mode |
| `donnees.html` | `MODES`, 13 pairs indexed **by position** | 14 classes reads past the end, the whole grid stops drawing |
| `index.html`, `experience.html` | "660 Hz", "3 910 Hz" in prose | stale examples |

The positional coupling is the dangerous one: it survives today only because there are 6
classes and the array has 13 entries.

The fix is to split the file by who writes it:

- **`site/js/classes.generated.js`** — written by `tasks.py modes`, in the same command that
  already writes `config/classes.json`, so the two cannot drift. Mechanical fields only:
  frequency, label, nodal-line count, symmetry, width. Carries the same "do not hand-edit"
  header `classes.json` carries.
- **`site/js/frequencies.js`** — the editorial layer, keyed **by frequency rather than by
  position**: trilingual description, complexity, mode indices where known, colour. Written
  by people. Orphan entries are ignored; absent ones degrade without lying, which
  `hasLocalMetadata` already does.

`donnees.html` iterates the generated list. The positional `MODES` array is deleted. The
simulated thumbnail is drawn only when the editorial layer supplies the mode indices;
without them the card renders without a canvas instead of breaking the page. Real images
are sub-project E.

## 8. Testing

- **pytest**: the mount serves `analyseur.html` and `css/style.css`; a missing `/api/*` route
  stays a JSON 404 and never becomes HTML; `data/`, `runs/` and `src/` are **not** reachable
  over HTTP; `/api/predict` carries `quality`, and a blurred image gets a reason while still
  being predicted.
- **Consistency** (`docs/05` §6): `/api/predict` on a `tests/fixtures/` image returns the
  class the Python pipeline predicts directly. Exact.
- **The forward-compatibility test** (§2): a fabricated `/api/health` with **12** classes at
  frequencies unrelated to today's, asserting the analyser and the data table still render.
  Without it, "ready for the real data" is a promise; with it, it is a property.
- **JS**: the mapping is extracted as a pure function and tested with `node --test`. No new
  toolchain — `vitest` lives in `web/`, which is going away.
- Playwright (upload to prediction) is noted for the plan, not promised in A+B.

## 9. `web/` is not deleted yet

The team chose the static site as the only frontend, and that is the destination. But
deleting `web/` now would destroy the only reference implementation of the pipeline viewer
and the `/dades` wizard — precisely the things sub-projects C and D have to port. So:
`tasks.py app` stops building and serving it in B; the directory is removed at the end of E.

## 10. Documentation

ADR **0007** records the reversal, because `docs/05` §4 and `CLAUDE.md` rule 6 currently say
the opposite and a future session would "fix" this back believing it a slip:

- a static multipage vanilla-JS site replaces Vite + React + TypeScript;
- French stays the default UI language and the French filenames are kept, against rule 6's
  "UI strings are Catalan" and §4's Catalan slugs. This was an explicit team decision.

`docs/05` §3 is amended for the `quality` field and §4 for the frontend stack.

## 11. Known limits

Until the classes are chosen, the analyser can only be exercised against the synthetic
checkpoint. The chain is demonstrated end to end, but nothing on screen means anything about
a plate — which is why the banners in §6 are not decoration.

Stage titles arrive from the server in Catalan by `docs/05` §8's design, which makes them
didactic surface. In a French-default UI they will read as Catalan. A affects one label;
C affects thirteen, and decides it.
