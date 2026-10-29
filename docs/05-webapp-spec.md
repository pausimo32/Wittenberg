# Local app specification

**Local, portable, CPU-only.** One process, on whichever computer the team chooses. FastAPI on
`127.0.0.1:8000` serving a built Vite + React frontend and a small JSON API.
See `docs/adr/0004` and `docs/adr/0005`.

Not deployed publicly and not reachable from another device — but **not tied to one machine
either**. `git clone` + `python tasks.py setup` + `python tasks.py app` must give a working
system on any computer with Python, with **no GPU, no API key and no internet**. The checkpoint
is 1.17 MB and travels with the repository.

Inference runs on the CPU in about 12 ms (`docs/adr/0005`). CUDA is used only if it happens to
be present; it is never required and never assumed.

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
│             named by runs/latest.txt (or --run), model.to(device).eval()
│             device = cuda if available else cpu — cpu is the supported default
│             assert checkpoint.preprocess_hash == hash(config/preprocess.json)   ← refuses to start
├── GET  /api/health   → {model_run, preprocess_version, device, classes, do_not_report}
├── POST /api/predict?stages=true|false   → multipart image
│                      → src.preprocess.preprocess_stages(...)   ← THE SAME FUNCTION AS TRAINING
│                      → model forward, softmax
│                      → forward hooks: feature maps + Grad-CAM (§8)
│                      → {top3:[{freq_hz, p}], stages:[…13], elapsed_ms, timings_ms,
│                         roi_mode, markers_found, quality}  — each stage carries its ms
├── POST /api/dataset/session   → open a draft session (nothing written yet)
├── POST /api/dataset/upload    → images + one label; runs the SAME gate as tasks.py ingest
├── POST /api/dataset/commit    → write rows to metadata.csv; re-run the split
├── GET  /api/dataset/summary   → per-class/per-session matrix, yield, interval widths
└── static  the Vite build (web/dist)
```

Rules:

- `/api/predict` calls `src.preprocess.preprocess_stages` — the same function training calls,
  which is why the diagram above says so. It does **not** reimplement, wrap, inline or
  "optimise" any step. If a transform needs changing, it changes in `src/preprocess.py` and the
  model is retrained.
- The startup hash assertion is the one guard replacing the old parity gate
  (`docs/adr/0004`). Never make it a warning.
- **`/api/health` carries the checkpoint's `do_not_report`.** `src/train.py` writes that field
  into the `metrics.json` of any run built from `SYNTH-*` sessions, saying that its accuracy
  measures how well the network inverted the formula that drew the images. The app will serve
  such a checkpoint; what it must not do is let the number look like a result, so the field is
  exposed here and §4 hangs a visible banner on it. `null` means the run's numbers may be
  quoted.
- **`quality` is the ingest gate, run on the upload for information.** `src/quality.py::assess`
  — the same function `tasks.py ingest` calls, never a second copy — returns
  `{quality, reason, metrics}`. On an upload it **warns and never rejects**: a blurred photo
  still gets an answer and is still told it is blurred. The gate decides what enters the
  dataset, not what a participant may ask. It is handed the stages the answer came from, so
  the "never run the pipeline twice" rule below still holds.
- The response includes the pipeline stages as base64 PNGs — see §8. `?stages=false` returns
  only the final edge map, for the rare call where the intermediates are not wanted.
- **One call produces both the stages and the prediction.** Never run the pipeline twice, once
  "to show" and once "to predict": the user would be shown images that did not produce the
  answer on screen.
- Bind to `127.0.0.1` explicitly. No CORS wildcard. No upload directory: images are processed in
  memory and never written to disk.
- Single worker. The model is loaded once; concurrency is one person at a laptop.
- **No outbound network call, ever.** Not for inference, not for telemetry, not for fonts. A
  Playwright test asserts zero outbound requests. The app must work with the machine offline.

## 4. Frontend — a static multipage site

**Amended by `docs/adr/0007`.** This section described Vite + React + TypeScript. The app's
frontend is the static multipage site in `site/`: plain HTML, CSS and browser JavaScript,
served by FastAPI from the same origin, with no build step. Read that ADR before reversing
this — the reversal was a team decision, not an accident.

| Page | Purpose |
|---|---|
| `index.html` | one sentence, one button |
| `analyseur.html` | file, drag-drop or camera → prediction |
| `experience.html` | the didactic sequence (`docs/01-domain-physics.md` §4) |
| `donnees.html` | the class table and the dataset, read from `js/classes.generated.js` |
| `apropos.html` | method, limits, team, TDR reference |
| `acquisition.html` | the capture station: tone, wait, check, capture, store — §10. Team only |

French UI copy by default, with a `ca`/`es`/`fr` switcher in `js/i18n.js`; English code
(`CLAUDE.md`). The French default and the French filenames are ADR 0007's second decision and
are a deliberate exception to rule 6.

**The class list is never written by hand.** `tasks.py modes` generates
`site/js/classes.generated.js` from `config/classes.json`, in the command that freezes the
class index. Editorial material lives in `js/frequencies.js`, keyed by frequency. The classes
are not chosen yet, and no page may have to be edited on the day they are.

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

### The geometry warning — `roi_mode`

`/api/predict` returns `roi_mode` and `markers_found`. When `roi_mode` is not `homography` the
result needs **its own visible state**, above the three-tier confidence presentation and
independent of it: the four ArUco markers were not in the frame, so the photo was centre-cropped
instead of rectified, and the model is reading a geometry it never saw during training.

This is the ordinary case, not an edge case. Every training frame went through the session
calibration; every photo taken by someone who is not running a capture session — at `/apren`,
from a phone, an old picture — will not. The confidence numbers are computed the same way and
look exactly as convincing, which is precisely the problem: the failure is invisible to the
person reading the screen, so the interface has to be the thing that says it.

Say what happened and what it costs, in one sentence, and do not bury it in the confidence tier —
a low-confidence answer and a wrongly-rectified one are different failures and lead to different
next actions (take a better photo vs. take the photo again with the frame). Ablation A10
(`docs/04-model-spec.md` §5) measures the accuracy drop; once it has a number, quote it here.

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
The only performance requirement left is that a prediction feels instant. If `elapsed_ms`
exceeds **150 ms**, something is wrong — most likely the model is being reloaded per request
instead of once at startup.

Measured end to end on an ordinary CPU, over a real socket, with all thirteen stages: **82–129
ms**, of which ≈ 8 ms decode, ≈ 12 ms ArUco detection, **≈ 1.3 ms preprocessing** (steps 2–8),
≈ 5–11 ms inference, ≈ 16 ms Grad-CAM and **≈ 43 ms PNG-encoding the stages** — the largest
single phase by far. `?stages=false` is ≈ 29 ms. The budget holds with room to spare; if it ever
stops holding, the encode phase is where to look, not the pipeline.

*(The earlier figure here, "≈ 24 ms preprocessing", was an estimate and was wrong by about 18×.
It has been replaced by a measurement — and, more usefully, by `timings_ms`: every response now
carries the real numbers for that request, so no figure in this document is load-bearing.)*

## 8. The pipeline viewer

The app shows the user every step the machine took to turn their photo into what the network
sees. This is not a debug panel bolted on: it is the most directly didactic surface in the whole
project, and it is why `preprocess_stages()` exists.

### Why it is animated, not streamed

The obvious reading of "show the steps as they happen" is a live stream — server-sent events,
one stage pushed as it finishes. Do not build that. Measured with
`preprocess_stages(collect_timings=True)`, the whole seven-step pipeline takes **≈ 1.3 ms on
CPU** and the network adds **≈ 5–11 ms**. The slowest preprocessing step is the ROI warp at
≈ 0.55 ms; Canny is ≈ 0.35 ms.

The cost is **flat in the size of the capture** — 800×800, 1400×1400 and 3024×3024 all land at
≈ 1.3 ms — because step 2 resamples to a fixed 512² ROI and every later step works on that. Only
the warp reads the full-resolution array.

Every stage would arrive inside one animation frame, several times over; the user would see the
last image appear and nothing else. Streaming would add an endpoint, an ordering concern and a
failure mode in exchange for a worse result. The measurement makes that argument stronger than
the estimate it replaces did.

So: **one request returns all thirteen stages, and the UI walks through them at a human pace** —
~1.1 s per step, with play/pause, step-by-step arrows, a clickable thumbnail rail, and a
"compare with the original" toggle. Show the *real measured* time of each step next to it
(`timings_ms`). That is honest — the user sees both that the machine is fast and what it did —
and it turns "it's instant" from a thing they have to take on trust into a number on screen.

### The viewer covers the WHOLE chain, not just preprocessing

"The steps the AI follows" means all of them. The eight preprocessing stages are the first half;
the second half is what happens inside the network, which is the part a visitor assumes is
unknowable. On the server, in Python, it is four forward hooks.

| Part | Stages | What it answers |
|---|---|---|
| Preparation | 1–8, from capture to tensor | what was done to my photo |
| **Inside the network** | 9–11, one per convolutional block | what the network extracts |
| **Where it looked** | 12, Grad-CAM over the input | why it answered that |
| Decision | 13, the probabilities | how sure it is |

**Stages 9–11: feature maps.** After each block, show a small grid of channels (8 of the 32, 64
and 128, chosen once and fixed so the display is stable across images). Say plainly what the
reader is looking at: the first block responds to simple changes of brightness, the later ones to
combinations of them. Do **not** claim a given channel "detects nodal lines" unless you have
checked that specific channel — it is a tempting sentence and usually false.

**Stage 12: Grad-CAM.** A heat map over the original photo showing which regions moved the
answer. This is the stage that turns "the machine says 740 Hz" into "the machine says 740 Hz
because of *these* lines", and it is the single most didactic image in the project. It is also
the honest one: when the model is wrong, the heat map usually shows why.

**Stage 13: the probabilities.** A bar per class, sorted. The reader sees that the output is a
distribution, not a verdict — which is what makes the three-tier confidence presentation in §4
make sense rather than look like hedging.

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
| 9–11 | Feature maps | what each block of the network extracts |
| 12 | **Grad-CAM** | where it looked — the answer to "why that frequency?" |
| 13 | Probabilities | the output is a distribution, not a verdict |

Step 6 is the "aha" and should be the one the animation lingers on. It is also the step that
makes concept 6 of `docs/01-domain-physics.md` §4 — *nodal lines are where the sand collects* —
visible rather than asserted.

### Rules

- Every stage comes from the same call that produced the prediction — the preprocessing stages
  from `preprocess_stages()`, the feature maps and Grad-CAM from hooks on that same forward pass.
  Never recompute anything for display.
- Show the exact OpenCV call and the parameters **read from `config/preprocess.json`**, not
  retyped into the UI. If someone changes a threshold, the viewer must say the new number.
- Steps 7 and 8 render with `image-rendering: pixelated`. At 128×128 the pixel grid *is* the
  point; smoothing it hides that the network sees something coarse.
- **Stage 1 is capped for display at 768 px on the long side**, and the response reports both
  the original `shape` and the encoded `display_shape` so the UI can say "mostrada a 768×576 ·
  original 4032×3024". The pipeline still receives the full-resolution array — only the picture
  sent to be looked at is smaller. This is not cosmetic: measured on a 4032×3024 capture,
  encoding stage 1 uncapped costs **459 ms and 24 MB**, a 3× breach of the §7 budget on the
  first real phone photo. At 768 it is 19 ms and 590 KB. 1024 would be 32 ms and 1 MB, for
  detail nobody looks at in a card beside twelve others.
- Payload: the thirteen stages are twelve PNGs, ≈ 1.8 MB, and cost ≈ 43 ms to encode — the
  largest phase of the request. Measured breakdown: stage 1 ≈ 19 ms, stage 2 (the 512² RGBA
  ROI) ≈ 10 ms, stage 12 ≈ 8 ms, stages 3–5 ≈ 3 ms each, the rest negligible. On localhost that
  is nothing and needs no attention.

  **If the field laptop misses the §7 budget** (measure it there — see F7 in
  `docs/00-master-plan.md`), `timings_ms` will say whether encode is the cause, and there are
  exactly two cheap levers, in this order: drop `DISPLAY_MAX_SIDE` to 512 (−11 ms), and encode
  stage 2 as RGB instead of RGBA, whose alpha channel is a constant 255 (−3 ms). Do not add
  compression complexity, and do not touch the pipeline: none of this is the model's fault.
- Keyboard: ← → to step, space to play/pause. Participants share one laptop.

### Also use it for annex 7

`python tasks.py annexes` renders the same eight stages for one representative image per class
into `annexes/A7_codi_preprocessament.pdf`. The TDR then shows the pipeline rather than only
listing the code.

## 9. `/dades` — supplying the labelled database

The last step of the whole project is handing the system a labelled dataset, and it must not
require copying folders by hand or editing a CSV. This view is where class + label goes in.

**Route:** `/dades`. Not part of the participant flow — it is the team's working surface, and it
is never shown during a field session.

### The flow

```
1  New session   →  session id, date, material, grain mass, distance, lighting, dwell
2  Calibration   →  drop the ArUco frame; the 4 markers are detected and previewed
                     with the plate quad drawn on top (docs/02-dataset-spec.md §1.1)
3  Per class     →  enter the frequency in Hz, drop that class's photos
4  Live gate     →  each image is preprocessed and gated AS IT ARRIVES:
                     accepted, or rejected with the reason and a thumbnail
5  Review        →  the session summary; accept it into the dataset or discard it
6  Split         →  re-runs the session-wise split and shows the resulting counts
```

Step 4 is the one that earns this view. A rejected image tells you *why* — blurred, markers
missing, luminance out of range, interior edge density too low — and shows you the Canny result
next to the original. You find out that the lighting is wrong on photo three, not after
importing four hundred.

### Where a session in progress lives

A session being labelled is a **draft**: `drafts/<session_id>/`, at the repo root, outside `data/`
and ignored by git. It holds the uploads exactly as they arrived, the calibration, and one gate
verdict per frame.

That location is the reason "a half-finished session leaves no trace in `data/`" is a property of
the layout rather than a cleanup that has to work: there is no code path that could leave a trace
there, because nothing is written there until the commit. It is also why discarding a draft is safe
to offer as a button.

It is the repo root and not the system temp directory on purpose — an orphan afternoon in `%TEMP%`
that nobody can name is indistinguishable from a lost one. And it is beside `data/` on purpose: the
commit *moves* the captures rather than copying them, so six hundred 12 MP frames change directory
in milliseconds, and on Windows a cross-volume `os.rename` fails outright.

Draft state is on the **server**, keyed by the session id, not in the page. Reloading, opening a
second tab or being handed the link finds the session where it was — which is what makes the
`BrowserRouter` fallback worth having. `GET /api/drafts` lists what is open with the identifier,
the full path and the size on disk, so a draft nobody remembers starting can be found and finished
or discarded, and so the person who has to delete one of three can tell which.

### Committing: either the draft exists or the session exists

The commit does not move the captures to their final name and then ingest them. It moves them to
`data/raw/.incoming-<session_id>/`, runs `ingest.ingest_session` **there**, and renames the
directory to `<session_id>` only once that function has returned. `data/calib/<session_id>.json` is
written at that same last step, because `data/calib/` is committed to git and an orphan calibration
for a session whose ingest failed would be a file in the repository describing nothing.

So a disk that fills up or one corrupt capture leaves a staged draft that says how to resume, and
leaves nothing in `data/` answering to the session id. Committing again resumes from the staged
captures; discarding the draft takes them with it.

One window is not closed and is not pretended away: `ingest_session` writes `metadata.csv` as its
last act, so a crash between that write and the rename leaves rows whose captures are still staged.
Re-running the commit converges — `_clear_session_processed` and `replace_session_rows` both key on
the session id.

The commit runs on a thread and reports `(done, total)`; the view polls it. Polling and not a
stream, because a reload drops an `EventSource` exactly as it drops React state, and requirement 2
is that a reload finds the session where it was. At ≈ 0.56 s per 12 MP frame a ten-class session of
sixty frames is minutes, not seconds, and nothing about it blocks.

### Three more rules the build settled

**A session id that already has rows is a question, not a replacement.** `ingest_session` is
idempotent, which is right for a command line where re-running is what you meant. From a browser it
is not: an id typed one digit wrong lands on a session someone else captured, and idempotent then
means the old rows are gone without anyone being asked. The commit refuses with the row count and
offers the replacement explicitly. A refused commit leaves the session it refused to replace
byte-identical.

**Calibration comes first, and the server enforces the order.** Uploading a pattern before the four
markers have been found is refused before the bytes are stored: the 512 px ROI the gate measures
does not exist yet. The view holds the step shut as well — the same rule said twice, once where it
can be seen and once where it cannot be bypassed.

**The raw bytes arrive raw, and the same bytes arrive once.** The upload is stored byte for byte —
no client-side resize, no re-encode, no EXIF strip — and every preview is generated from the file
*after* it is stored. `docs/02` §4 defines `data/raw/` as untouched, straight off the phone, and a
recoded JPEG survives Canny as spurious contours that cannot be spotted in the file afterwards. The
SHA-256 that proves it also refuses a frame the draft already holds, under any name, in any class:
a retried batch over a connection that stumbles is the likeliest way this view corrupts a dataset,
and a repeated frame is not a second observation. Duplicates are counted as their own category,
neither accepted nor rejected — the gate never saw them, and folding them into the rejections would
make a retried upload read as a lighting problem.

### Rules that are not negotiable

- **The label is typed once per class, never per image.** A folder of photos of the same mode is
  one label. Per-image labelling invites typos and there is no reason for it.
- **The gate is the same gate.** `/dades` calls the same ingest code as
  `python tasks.py ingest` — same thresholds, same interior margin, same reasons. It is a
  different interface to one implementation, not a second one. A discrepancy here would let the
  UI accept images the pipeline rejects.
- **Nothing is written until step 5.** Uploading is not committing. A half-finished session
  leaves no trace in `data/`.
- **Re-ingesting a session replaces its rows.** Never duplicates them.
- **Rejected images are kept and counted**, and the yield appears in the summary. They are a
  reported result (`docs/06-research-protocol.md`), not rubbish.
- **`SYNTH-*` sessions cannot be mixed with real ones.** The view refuses and says why.
- **Editing a label after the fact requires re-running the split**, and the view says so rather
  than leaving the dataset quietly inconsistent.

### What it shows about the dataset as a whole

The same view, without uploading anything, is the dataset's dashboard — because the numbers that
matter are the ones nobody checks until it is too late:

- images per class per session, as a matrix — the empty cells are the ones that break the split
- how many sessions cover each class (fewer than 3 is a warning)
- test-set size per class, with the resulting Wilson interval width next to it
- yield by rejection reason, by lighting, by material
- a flag when the train/test split correlates with a condition rather than with the physics

This is the `wittenberg-dataset-curator` audit rendered instead of printed. The subagent stays —
it is what runs before training — but seeing the matrix while you still have the rig set up is
what stops you from discovering a missing class four days later.

### Testing

The four this section has always named, each now with the test that holds it:

| Property | Test |
|---|---|
| The gate applied by `/dades` and by `tasks.py ingest` returns identical verdicts | `test_dataset_api.py::test_the_live_gate_is_the_ingest_gate` — every verdict re-derived through `ingest.gate_frame` and compared |
| Abandoning a session mid-upload leaves `data/` byte-identical | `test_drafts.py::test_abandoning_a_draft_leaves_data_byte_identical` |
| Re-ingesting a session twice produces one set of rows | `test_dataset_api.py::test_re_ingesting_a_session_twice_produces_one_set_of_rows` |
| A `SYNTH-*` session alongside a real one is refused | `test_dataset_api.py::test_a_synthetic_session_alongside_a_real_one_is_refused` — at the door, before any upload |

And the five the build added:

| Property | Test |
|---|---|
| The bytes stored are the bytes uploaded | `test_drafts.py::test_stored_bytes_are_the_uploaded_bytes` (SHA-256 on disk) **and** `web/src/api/drafts.test.ts` (the bytes leaving the browser) |
| A draft survives a reload | `test_dataset_api.py::test_draft_survives_a_new_client` — a second app over the same roots, which is what a reload is |
| Frames are refused before the calibration | `test_dataset_api.py::test_frames_refused_before_calibration` |
| A failed ingest leaves no session, and can be resumed | `test_dataset_api.py::test_a_failed_commit_leaves_no_session_and_can_be_resumed` |
| The commit's progress moves | `test_dataset_api.py::test_commit_reports_progress`, `test_ingest.py::test_progress_is_called_once_per_frame` |

Two of these are worth knowing are *load-bearing* rather than decorative, because both were checked
by breaking the thing they guard: `test_concurrent_stores_do_not_lose_each_other` fails five times
out of five with the frames-log lock removed, and `test_the_quad_is_drawn_where_the_markers_are`
fails when the preview's scale factor is forced to 1.

## 10. `acquisition.html` — the capture station

**Decided in `docs/adr/0008`, status *proposed* until its feasibility probe has run.** Read the
ADR for every *why*; this section is the *what*.

The station automates a lab session end to end: play a class's frequency through the plate's
driver, wait for the pattern, check it, photograph it, store it, move on. It runs on a MacBook
with the iPhone 13 as a Continuity Camera, but nothing in the page is specific to macOS.

Like `/dades` (§9), it is the team's working surface, **never shown during a field session**.
It has a tab in the navigation, which `js/config.js` can hide.

### It is a client, not a second pipeline

The page adds no storage path and no gate. It drives the `/api/drafts` endpoints of §9 —
create the draft, calibrate, upload frames with `freq_hz`, commit — so every rule of §9 holds
unchanged: calibration first, raw bytes stored once, the gate is the ingest gate, nothing in
`data/` until the commit. The classes come from `js/classes.generated.js`; **the station
never takes a frequency typed by hand.** A class that is not in `config/classes.json` cannot be
captured.

### The flow

```
1  Session      →  the session.json fields of docs/02 §4, plus camera, dwell_s, rounds,
                   re-seed interval and the order seed
2  Pre-flight   →  camera found, stream resolution shown, macOS video effects off (reminder),
                   test tone through the amplifier, Wake Lock held
3  Calibration  →  one frame of the ArUco plate through POST /calibration; no way past it
                   without four markers
4  First frame  →  one pattern frame through the gate before the run starts — the
                   check_capture step of docs/02 §6, on the day, not on day four
5  Run          →  rounds × classes, round-robin, each frame through the checks below
6  Review       →  yield by class and by rejection reason, skipped slots, pauses; commit
                   through the existing endpoint, or discard the draft
```

### One capture

```
tone on (gain ramp)  →  dwell_s  →  wait until the pattern settles  →  grab a stream frame as PNG
→  POST /frames?freq_hz=…  →  markers where the calibration put them?  →  ingest gate verdict
→  accepted: next slot  ·  rejected: retry up to N  ·  still rejected: log, next slot
```

The checks, their order and what each failure does are ADR 0008 decision 6. Two rules from it
belong here, because they are the easiest to "improve" away:

- **The classifier is never a check.** No prediction is run on a station frame before it is
  stored.
- **The order is round-robin**, shuffled per round from the recorded seed. Never sixty frames of
  one class, then sixty of the next.

### Rules

- **The tone stops whenever the run is not capturing**: pause, error, re-seed, review, page
  hidden. Ramps on every start and stop; a volume control that is visible at all times.
- **A pause is resumable.** Draft state is on the server (§9), so a reload or a crash resumes
  at the next empty slot, not at the start.
- **A moved rig stops the run.** A failed marker-position check pauses the session and says
  why. It does not retry: the calibration no longer describes the frame, and retrying would
  only collect more frames that warp to the wrong place.
- **Every skipped or failed slot is logged and counted**, like a rejected frame. A class that
  ends the session short of its target is shown as short, not padded.

### Testing — to be written with the page

| Property | How |
|---|---|
| The round-robin order is a pure function of the seed, and every class appears once per round | `node --test` over the scheduler |
| A frame the station uploads gets the verdict `tasks.py ingest` would give it | already held by `test_dataset_api.py::test_the_live_gate_is_the_ingest_gate`; the station adds no gate to test |
| The marker-position check rejects a frame shifted after the calibration | pytest, on a synthetic calibration and a translated copy of its frame |
| The tone is stopped on pause, error and page hide | `node --test` over the run state machine |
