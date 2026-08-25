# Master plan — Wittenberg

Eight phases, F0 → F7. Each has an exit gate. Do not start a phase until the previous gate is
green: every gate exists because failing it silently costs more than failing it loudly.

`/clear` between phases.

| Phase | Outcome | Gate | Model |
|---|---|---|---|
| F0 | Repo + Claude Code configured | `tasks.py setup` green, GPU seen, 7 agents listed | sonnet |
| F1 | Modes found, classes fixed | `data/modes.csv` ≥ 10 modes | lab + haiku |
| F2 | Python preprocessing | unit tests green on `tests/fixtures/` | opus → sonnet |
| F3 | ≥ 4 ingested sessions | curator says READY TO TRAIN | sonnet |
| F4 | Trained + evaluated model | acceptance gates in model-spec §6 | opus |
| F5 | Local serving API | `check-model` green, API == direct call | sonnet |
| F6 | Vite + React app | analyse flow works end to end | sonnet |
| F7 | Didactic module + field prep | E2E green, reset clean, `v1.0-field` tagged | sonnet |

The critical path is **F2 → F4** — the science. Nothing after F4 can make the results better;
it can only present them. (Before `docs/adr/0004` the critical path ran through preprocessing
parity; making the app local-only removed that risk entirely.)

---

## F0 — Foundations

Repo skeleton, `CLAUDE.md`, `docs/`, `.claude/agents/`, `.claude/skills/`, `tasks.py`, Python and
Node toolchains, `config/preprocess.json`, `config/versions.json` with `opencv-python` pinned.

**Prompt**

```
## Context
Phase F0. Empty repo. The docs/ and .claude/ trees from the Wittenberg starter kit are in place.
Read: CLAUDE.md only.
Do NOT read: docs/.

## Task
Finish the toolchain so `python tasks.py setup` produces a working Python and Node environment
on native Windows with CUDA.

## Constraints
- ENVIRONMENT: native Windows, NVIDIA GPU, no WSL. No `make`, no symlinks. tasks.py is the
  runner and already exists — extend it, do not replace it, and never add a Makefile.
- Python 3.11. requirements.txt is written; do not swap PyTorch for TensorFlow (ADR 0003) and
  do not drop the --index-url line, or pip installs the CPU-only wheel silently.
- Node 20, Vite + React + TypeScript + Tailwind under web/. No Next.js, no SSR, no static-export
  config — the app is local-only (ADR 0004).
- `tasks.py check-versions` must fail loudly when opencv-python is not the pinned version
  (reproducibility across three machines). `tasks.py check-gpu` must report the CUDA device.
- Unimplemented targets print "NOT IMPLEMENTED (Fn)" and exit 1 — never exit 0.
- .gitignore: data/raw, data/processed, runs/, *.pt, *.pth, web/dist, node_modules, .venv.
  There is no served model artefact: the API loads runs/<id>/model.pt directly (ADR 0004).
- Windows specifics: use pathlib everywhere; UTF-8 explicitly on every file read/write
  (Windows still defaults to cp1252 in places); no shell-only utilities (grep, sed) in any
  script that the team is expected to run.

## Definition of done
- [ ] `python tasks.py setup` exits 0 on a clean clone
- [ ] `python tasks.py check-gpu` prints "cuda available: True" and the device name
- [ ] `python tasks.py check-versions` passes, and fails when the versions are made to disagree
- [ ] `npm run build` in web/ produces `web/dist/`
- [ ] no Makefile, no os.symlink anywhere in the repo

## Model
sonnet — mechanical scaffolding.
```

**Do externally**

1. Install Python 3.11, Node 20 and the current NVIDIA driver. Claude Code as the VSCode
   extension is enough for the whole project; the CLI is only needed for headless runs.
2. Paste the six `/plugin install` lines from `SETUP.md` §2 into the Claude Code chat. They are
   interactive slash commands — no agent can run them for you, and they are the one manual step.
3. Import the four external subagents (`SETUP.md` §3), then check each one's `name`, `model`
   and `permissionMode` per `SETUP.md` §4.
4. `/agents` — confirm 7 agents: the three `wittenberg-*` plus the four imported.
5. `/context` — record the baseline cost of CLAUDE.md + skill descriptions.
6. `git init && git commit` the skeleton before anything else.

---

## F1 — Find the modes (lab, not code)

The reference study's 14 frequencies belong to its plate, not ours. Ours must be measured.

**Do externally**

1. Assemble the rig. Camera at d = 210 mm, `H = 180.7 mm`. Lock AE/AF, HDR off, Live Photo off.
2. Tape four ArUco markers at the plate corners. Photograph once → `data/calib/<session>.json`.
3. Sweep 100 → 5000 Hz in 5 Hz steps, 3 s dwell. Note every frequency where the pattern is
   stationary, symmetric, and **reforms after re-seeding the sand**. The re-seeding test is what
   separates a real mode from a transient.
4. For each candidate, record the resonance width (the range over which the figure holds).
5. Discard modes whose nearest neighbour is < 40 Hz away.
6. Fill `data/modes.csv`: `freq_hz, width_hz, n_nodal_lines, symmetry, quality, notes`.

Expect 10–18 usable modes. Fewer than 10 fails the phase gate: it means the drive amplitude is
too low or the plate is over-constrained at the centre — fix the rig and sweep again before
continuing.

**Prompt** (after the lab work)

```
## Context
Phase F1 complete in the lab. data/modes.csv is filled by hand.
Read: docs/02-dataset-spec.md §3.

## Task
Validate data/modes.csv and generate config/classes.json.

## Constraints
- Reject any mode whose nearest neighbour is < 40 Hz away; list what was rejected and why.
- classes.json is an ordered list of integer Hz; the index is the model's class index and is
  frozen from here on. Changing the order later silently invalidates every trained model.

## Definition of done
- [ ] config/classes.json exists, ≥ 10 classes, ascending
- [ ] a printed table of accepted vs rejected modes with reasons

## Model
haiku — validation and file generation, no reasoning.
```

---

## F2 — Python preprocessing (critical path)

The reference implementation. Everything downstream is measured against it.

**Prompt**

```
## Context
Phase F2. config/preprocess.json and config/classes.json exist. No image code yet.
Read: docs/03-preprocessing-spec.md.
Do NOT read: docs/01, 04, 05, 06, 07.

## Task
Implement src/preprocess.py implementing §2 steps 1–8 exactly.

## Constraints
- Every parameter read from config/preprocess.json. Zero numeric literals in the transform path.
- Public API: `preprocess(image_rgba: np.ndarray, calib: dict | None) -> np.ndarray` returning
  float32 (128,128,1) in [0,1]; plus `preprocess_stages(...)` returning the intermediate array
  after each of steps 1-8 — used by the pipeline viewer, by tests, by debugging a bad
  prediction, and by annex 7.
- Pure function. No file I/O, no globals, no logging inside the transform.
- HEIC/EXIF/P3 handling lives in the shared src/imageio.py helper, not here (§5.1).
- pytest covering: each step's output shape and dtype; determinism (same input → identical
  bytes over 10 runs); both roi_modes; a 4-marker-missing failure path.

## Definition of done
- [ ] pytest green
- [ ] `grep -nE '[^.0-9](0\.|[0-9]{2,})' src/preprocess.py` finds no transform constants
- [ ] preprocess_stages returns 8 arrays (steps 1-8) with documented shapes

## Model
opus for the design of preprocess_stages and the calibration/ROI logic; sonnet for the rest.
```

### F2 follow-up — read the two ROI flags from the config (spec 1.1.0)

`INTER_LINEAR` and `BORDER_REPLICATE` lived only in the prose of §2 and were implemented as named
`cv2` constants. Spec 1.1.0 moves them into `config/preprocess.json` as `roi.interpolation` and
`roi.border_mode`, so a reader can rebuild the pipeline from the config alone. The code has to
follow, or `config/preprocess.json` is no longer the single source of truth it claims to be.

Do this **before F3**: it changes the config hash, so once a checkpoint exists it costs a retrain.

```
## Context
Spec 1.1.0 adds roi.interpolation and roi.border_mode to config/preprocess.json.
src/preprocess.py still uses named cv2 constants for both.
Read: docs/03-preprocessing-spec.md §1, §2 step 2.

## Task
Make step 2 read both flags from the config instead of hard-coding them.

## Constraints
- Map the string names to cv2 attributes through an explicit allow-list dict.
  Never getattr(cv2, value) on an unvalidated string.
- An unknown value raises at load time with the offending key and value, not at
  transform time.
- Pixels must not change: the current values ARE INTER_LINEAR and
  BORDER_REPLICATE. Add a test asserting the new path is byte-identical to the
  old constants on a fixture.
- config/preprocess.json is already at 1.1.0. Do not bump it again.

## Definition of done
- [ ] pytest green, including a byte-identity test against the previous output
- [ ] a test that an unknown border_mode raises at config-load time
- [ ] no cv2.INTER_* or cv2.BORDER_* literal in the transform path
- [ ] the mutation "swap INTER_LINEAR for INTER_NEAREST in the config" is caught

## Model
sonnet — small, but the allow-list and the failure mode need care.
```

---

## F3 — Data collection and ingest

**Do externally — the phase that takes real calendar time**

Four sessions minimum, on **different days**, each covering every class. Per session:

1. Fresh sand, fixed mass (record it). Same material within a session.
2. Lock exposure and focus. Photograph the ArUco calibration frame first.
3. Per frequency: set it, wait for the dwell time, capture ≥ 60 frames.
4. Log material, grain mass, distance, lighting, dwell, anything anomalous.
5. Vary lighting **between** sessions, never within one.
6. Session 4 uses a different granular material, for ablation A8.

Do not shortcut this by taking 240 frames in one sitting. Four sittings of 60 is a dataset;
one sitting of 240 is one observation photographed 240 times.

**Prompt** (per session)

```
## Context
Phase F3. Session <YYYYMMDD-NN> is in data/raw/. src/preprocess.py passes its tests.
Read: docs/02-dataset-spec.md.
Do NOT read: docs/01, 03, 04, 05, 06, 07.

## Task
Ingest session <id> and report its yield.

## Delegate
- Use the dataset-ingest skill.
- Then run wittenberg-dataset-curator for the full-dataset audit.

## Constraints
- Idempotent: re-running replaces this session's rows, never duplicates.
- Quality gate per §6. Rejects stay in raw/ and are counted, never deleted.
- Splits by session only (`python tasks.py split`). Never hand-edit the split column.

## Definition of done
- [ ] metadata.csv rows == accepted images, image_id unique
- [ ] ingest_report.json with rejection rate by reason
- [ ] curator verdict printed; if BLOCKED, what to capture next

## Model
sonnet.
```

---

## F4 — Train and evaluate

**Prompt**

```
## Context
Phase F4. Curator says READY TO TRAIN. ≥ 4 sessions ingested.
Read: docs/04-model-spec.md.
Do NOT read: docs/01, 02, 03, 05, 06, 07.

## Task
Implement src/model.py + src/train.py + src/evaluate.py; train wittenberg_v1 and
baseline_paper; produce the full evaluation set.

## Constraints
- Split by session (GroupShuffleSplit on session_id). Never random. ADR 0002 is not
  negotiable.
- Seed 42 for random/numpy/torch + torch.use_deterministic_algorithms(True)
  and CUBLAS_WORKSPACE_CONFIG=:4096:8.
- runs/<ts>/ contains config.json (with the preprocessing version), metrics.json, history.csv,
  model.pt, report/.
- Evaluation must produce all six outputs of §4, including Grad-CAM and the
  frequency-distance error analysis. A run without Grad-CAM is not evaluated.
- baseline_paper is a faithful replication — do not improve it.

## Definition of done
- [ ] both models trained, metrics.json written
- [ ] session-wise accuracy ≥ 80 %, worst-class recall ≥ 60 % (or a written explanation)
- [ ] confusion matrix as PNG and CSV
- [ ] Grad-CAM for 3 correct + 3 incorrect per class
- [ ] accuracy reported with a 95 % Wilson interval

## Model
opus — the split logic and the evaluation design are where this phase goes wrong.
```

**Then the ablations** — one prompt each, `docs/04-model-spec.md` §5, 3 seeds, mean ± std.
A4 (random vs session split) first: its result determines how every other number is read.

---

## F5 — Local serving API

FastAPI on `127.0.0.1`, PyTorch model on the GPU, calling `src/preprocess.py`. Prompt: the
worked example in `PROMPTS.md`.

**Gate** — `python tasks.py check-model` green; `/api/predict` on a reference image returns exactly
what calling `preprocess` + the model directly returns; mutating `config/preprocess.json` makes
startup fail; `elapsed_ms` < 200 on the GPU.

Two rules that matter:

1. `/api/predict` calls `src.preprocess.preprocess_stages`. Not a copy, not a wrapper that "does
   the same thing". The moment someone inlines a step for speed, the property this whole
   architecture was reorganised to get is gone, and nothing will tell you.
2. **One call returns both the stages and the prediction** (`docs/05-webapp-spec.md` §3, §8).
   Never run the pipeline twice — once to display, once to predict — or the user is shown images
   that did not produce the answer next to them.

---

## F6 — Vite + React app

```
## Context
Phase F6. The API from F5 works and its tests pass. No frontend yet.
Read: docs/05-webapp-spec.md.
Do NOT read: docs/01, 02, 03, 04, 06, 07.

## Task
Build the Vite + React frontend: routes, the /analitza flow including the
pipeline viewer, and the shell for /apren, /galeria and /sobre.

## Constraints
- Vite + React + TypeScript + Tailwind, react-router. No Next.js, no SSR.
- Built output is served by FastAPI; `npm run dev` proxies /api to :8000.
- Result presentation follows §4 exactly: top-3 with probabilities, the three
  confidence tiers, and the edge map beside the original. Never a bare number,
  never a rounded "nicer" one, never a hidden low confidence.
- Pipeline viewer per §8: eight stages from ONE request, animated at ~1.1 s per
  step with play/pause, arrows and a thumbnail rail. Do NOT stream the stages —
  the pipeline finishes in ~24 ms, so there is nothing to watch live. Show each
  step's real measured time. Steps 7 and 8 render pixelated.
- The OpenCV call and parameters shown in the viewer are read from
  config/preprocess.json, never retyped into the UI.
- Catalan UI strings via docs/07-glossary-ca-en.md. English code.
- No localStorage, no cookies, no analytics — anywhere, ever.

## Definition of done
- [ ] upload -> prediction works against the real API
- [ ] the three confidence tiers render correctly (unit-tested on fixed inputs)
- [ ] the viewer walks all 8 stages; arrows and space work; 7-8 are pixelated
- [ ] tsc --noEmit clean; npm run build succeeds
- [ ] `python tasks.py app` serves the built app at 127.0.0.1:8000

## Model
sonnet.
```

---

## F7 — Didactic module and field preparation

The phase that decides whether research questions P2 and S3 get answered. Three prompts, then
manual work.

1. **`/apren`** — the seven concepts in order (`docs/01-domain-physics.md` §4), WebAudio muted
   until pressed, sweep animation from the project's own captures, **clean reset between
   participants** (`docs/05-webapp-spec.md` §1).
2. **`/galeria` + `/sobre`** — generated from `data/metadata.csv` at build time, never
   hand-maintained.
3. **Hardening** — Playwright E2E over the full seven-step flow and the reset; accessibility
   pass; visual regression on every didactic step.

**Gate** — E2E green, reset leaves no residue, and `git tag v1.0-field`.

**Do externally**

- Freeze `v1.0-field` before the first session and do not patch it mid-study. A change makes
  groups non-comparable.
- Rehearse a whole session on the actual laptop, with someone who has not seen the app, and
  watch where they get stuck. Run it from the built app, never the dev server.
- Prepare the two Google Forms with random participant codes so pre and post can be paired
  without names. Participants use their own devices for the forms only.
- Prepare informed consent, and guardian consent for minors.
- Check the laptop: battery, screen brightness for the room, sound level, and that
  `python tasks.py app` starts with no network available.

---

## Cross-cutting

**Weekly** — `wittenberg-dataset-curator` (catches drift a single session hides).
**Before any number enters the TDR** — `wittenberg-repro-auditor`, without exception.
**Every preprocessing change** — bump the version, retrain, and expect `check-model` to reject
old checkpoints. That rejection is the guard working, not an obstacle.
**Every reversed decision** — an ADR, before the reversal. This plan has already reversed twice
(0003, 0004); both times the ADR made the blast radius obvious in minutes.
