# Local app specification

**Local-only.** One process on the team's machine. FastAPI on `127.0.0.1:8000` serving a built
Vite + React frontend and a small JSON API. See `docs/adr/0004`.

Not deployed. Not reachable from another device. `--host 0.0.0.0` is not a feature request.

## 1. Why it still has to be good

The app is not a demo. It is the instrument for research questions **P2** (does the tool make
wave concepts intuitive to non-physicists?) and **S3** (which group improves most?). At a field
session it runs on the team's laptop and participants use it **in turns**; their own devices are
used only for the Google Forms questionnaires.

Two consequences that a "it's only local" mindset would miss:

- **The didactic module must survive a stranger using it unattended for ten minutes.** No
  console errors, no dead ends, no jargon before it has been taught.
- **State must reset cleanly between participants.** One participant must never see the
  previous one's answers or progress. A visible "Comença de nou" that returns to a genuinely
  empty state, and an automatic reset when the last step finishes.

## 2. Processes and commands

| | |
|---|---|
| Production (field sessions) | `python tasks.py app` — builds the frontend if stale, starts uvicorn, opens the browser at `127.0.0.1:8000` |
| Development | `python tasks.py api` (uvicorn, reload) + `npm run dev` in `web/` (Vite on 5173, proxying `/api` to 8000) |

Run field sessions from the built app, never from the Vite dev server: hot-reload artefacts and
dev overlays in front of a participant invalidate the session.

## 3. Backend

```
FastAPI
├── startup   load config/preprocess.json, config/classes.json, the checkpoint
│             named by runs/latest.txt (or --run), model.to('cuda').eval()
│             assert checkpoint.preprocess_hash == hash(config/preprocess.json)   ← refuses to start
├── GET  /api/health   → {model_run, preprocess_version, device, classes}
├── POST /api/predict?stages=true|false   → multipart image
│                      → src.preprocess.preprocess_stages(...)   ← THE SAME FUNCTION AS TRAINING
│                      → model forward, softmax
│                      → {top3:[{freq_hz, p}], stages:[…], elapsed_ms, timings_ms}
└── static  the Vite build (web/dist)
```

Rules:

- `/api/predict` calls `src.preprocess.preprocess`. It does **not** reimplement, wrap, inline or
  "optimise" any step. If a transform needs changing, it changes in `src/preprocess.py` and the
  model is retrained.
- The startup hash assertion is the one guard replacing the old parity gate
  (`docs/adr/0004`). Never make it a warning.
- The response includes the pipeline stages as base64 PNGs — see §8. `?stages=false` returns
  only the final edge map, for the rare call where the intermediates are not wanted.
- **One call produces both the stages and the prediction.** Never run the pipeline twice, once
  "to show" and once "to predict": the user would be shown images that did not produce the
  answer on screen.
- Bind to `127.0.0.1` explicitly. No CORS wildcard. No upload directory: images are processed in
  memory and never written to disk.
- Single worker. The GPU model is loaded once; concurrency is one person at a laptop.

## 4. Frontend — Vite + React + TypeScript

No SSR, no file-based routing framework. `react-router` for four views. Tailwind for styling.

| View | Purpose |
|---|---|
| `/` | one sentence, one button |
| `/analitza` | file or drag-drop → prediction |
| `/apren` | the 7-concept didactic sequence (`docs/01-domain-physics.md` §4) |
| `/galeria` | the project's own modes, ordered by frequency |
| `/sobre` | method, limits, team, TDR reference |

Catalan slugs, Catalan UI copy, English code (`CLAUDE.md`).

### Result presentation — the honesty requirement

Unchanged from the original design, and still the most important UI decision in the project.
Show top-3 with probabilities, never a bare single number.

| Max probability | Behaviour |
|---|---|
| ≥ 0.70 | "Probablement **N Hz**" + the other two candidates |
| ≥ 0.40 and < 0.70 | "Podria ser **N Hz** o **M Hz**" — both shown equally |
| < 0.40 | "No ho sé prou" + guidance: more sand, wait longer, check the frequency is on resonance |

Always show the preprocessed edge map next to the original. Seeing what the model actually sees
is half the didactic value and makes failures self-explanatory.

Never round a prediction to a "nicer" number. Never hide low confidence. A tool that always
answers a confident number teaches participants that machines are always right, which is the
opposite of what a research project should demonstrate.

## 5. `/apren` — the didactic module

- Seven steps in the order of `docs/01-domain-physics.md` §4. Linear, no skipping ahead.
- One sentence, one interactive element, one check-question per step.
- **WebAudio tone generator** — sine, 100–5000 Hz, prominent volume control, −20 dB default,
  **muted until the participant presses play**, with a warning before the first tone.
- **Sweep animation** — pre-rendered from the project's own captures, never simulated.
- **No storage of anything personal.** Progress in React state only. No `localStorage`, no
  cookies, no analytics. Questionnaires are Google Forms, separate, with random participant
  codes.
- Accessibility: keyboard navigable, `prefers-reduced-motion` respected, contrast ≥ 4.5:1,
  legible at a distance if the laptop is shared or projected. Participants include older adults.
- **Reset between participants**, per §1.

## 6. Testing

- Unit (`vitest`): result-tier logic, the reset behaviour, the confidence formatting
- API (`pytest` + `httpx`): `/api/predict` on a `tests/fixtures/` image returns the class the Python
  pipeline predicts directly — this is a *consistency* check, not a parity check, and it should
  be exact
- Startup guard: a test that mutates `config/preprocess.json` and asserts the server refuses
  to start
- E2E (Playwright): upload → prediction; the full seven-step didactic flow; reset leaves no
  residue
- Visual regression on `/apren`, because a broken didactic step invalidates a field session
  that cannot be recollected

## 7. What this spec deliberately does not contain

No performance budget for bundle size, no lazy-loading strategy, no service worker, no
Lighthouse target, no CDN. Those were consequences of shipping to strangers over the internet.
The only performance requirement left is that a prediction feels instant on the local GPU —
if `elapsed_ms` exceeds 200 ms, something is wrong (the model is on the CPU, most likely).

## 8. The pipeline viewer

The app shows the user every step the machine took to turn their photo into what the network
sees. This is not a debug panel bolted on: it is the most directly didactic surface in the whole
project, and it is why `preprocess_stages()` exists.

### Why it is animated, not streamed

The obvious reading of "show the steps as they happen" is a live stream — server-sent events,
one stage pushed as it finishes. Do not build that. Measured on a 1400×1400 capture, the whole
eight-step pipeline takes **≈ 24 ms on CPU**, and the slowest single step (Canny) takes 13 ms.
Every stage would arrive inside one animation frame; the user would see the last image appear and
nothing else. Streaming would add an endpoint, an ordering concern and a failure mode in exchange
for a worse result.

So: **one request returns all eight stages, and the UI walks through them at a human pace** —
~1.1 s per step, with play/pause, step-by-step arrows, a clickable thumbnail rail, and a
"compare with the original" toggle. Show the *real measured* time of each step next to it
(`timings_ms`). That is honest — the user sees both that the machine is fast and what it did —
and it turns "it's instant" from a thing they have to take on trust into a number on screen.

### What each stage shows

| # | Stage | What the user should notice |
|---|---|---|
| 1 | Capture | perspective and uneven room light — the raw problem |
| 2 | ROI | the plate becomes a square, always at the same scale |
| 3 | Grayscale | colour was never the signal |
| 4 | CLAHE | the dark corner brightens; lines read equally everywhere |
| 5 | Blur | sensor grain goes, lines stay |
| 6 | **Canny** | the payoff: only the nodal lines survive — plate, grain, scratches and shadow are gone |
| 7 | Resize | greys come back; each pixel is a density of nodal line |
| 8 | Tensor | literally what the network sees |

Step 6 is the "aha" and should be the one the animation lingers on. It is also the step that
makes concept 6 of `docs/01-domain-physics.md` §4 — *nodal lines are where the sand collects* —
visible rather than asserted.

### Rules

- Stages come from the same `preprocess_stages()` call that produced the prediction. Never
  recompute for display.
- Show the exact OpenCV call and the parameters **read from `config/preprocess.json`**, not
  retyped into the UI. If someone changes a threshold, the viewer must say the new number.
- Steps 7 and 8 render with `image-rendering: pixelated`. At 128×128 the pixel grid *is* the
  point; smoothing it hides that the network sees something coarse.
- Payload: eight PNGs is roughly 1 MB. On localhost that is nothing. Do not add compression
  complexity for it.
- Keyboard: ← → to step, space to play/pause. Participants share one laptop.

### Also use it for annex 7

`python tasks.py annexes` renders the same eight stages for one representative image per class
into `annexes/A7_codi_preprocessament.pdf`. The TDR then shows the pipeline rather than only
listing the code.
