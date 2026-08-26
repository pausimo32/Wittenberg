# Master plan — Wittenberg

Eight phases, F0 → F7. Each has an exit gate. Do not start a phase until the previous gate is
green: every gate exists because failing it silently costs more than failing it loudly.

`/clear` between phases.

| Phase | Outcome | Gate | Model |
|---|---|---|---|
| F0 | Repo + Claude Code configured | `tasks.py setup` green, device reported, 7 agents listed | sonnet |
| F1 | Modes found, classes fixed | `data/modes.csv` ≥ 10 modes | lab + haiku |
| F2 | Python preprocessing | unit tests green on `tests/fixtures/` | opus → sonnet |
| F3a | Ingest pipeline built | synthetic sessions ingest + split; curator READY | sonnet |
| F3.5 | Documents and code reconciled | no doc↔code contradiction left open | sonnet |
| F4a | Training bench green on synthetic | `train` and `eval` write `runs/<ts>/`, `check-model` green — never an accuracy figure | sonnet |
| F3b | ≥ 4 real sessions ingested | curator says READY TO TRAIN | sonnet |
| F4b | Trained + evaluated model | acceptance gates in model-spec §6, on real captures, **both models** | opus |
| F5 | Local serving API (CPU) | `check-model` green, API == direct call, ≤ 150 ms | sonnet |
| F6 | Vite + React app + `/dades` | analyse flow end to end; a session can be labelled and committed | sonnet |
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
on native Windows, CPU-only. No CUDA anywhere in the app path (ADR 0005).

## Constraints
- ENVIRONMENT: native Windows, no WSL. The APP must run CPU-only on any
  computer; training uses the team's RTX 5070 and is separate. No `make`, no
  symlinks. tasks.py is the
  runner and already exists — extend it, do not replace it, and never add a Makefile.
- Python 3.11. requirements.txt is written and installs **CPU PyTorch on purpose** (ADR 0005).
  Do not add CUDA wheels: the app does not need them and they weigh hundreds of MB.
  GPU wheels belong in requirements-train.txt, installed only on the team's own RTX 5070
  (ADR 0005). There is no cloud notebook.
- Node 20, Vite + React + TypeScript + Tailwind under web/. No Next.js, no SSR, no static-export
  config — the app is local-only (ADR 0004).
- `tasks.py check-versions` must fail loudly when opencv-contrib-python is not the pinned
  version. `tasks.py check-device` reports the device; **CPU is a PASS, not a warning** — it is
  the supported configuration (ADR 0005). Never make CUDA a requirement anywhere.
- Unimplemented targets print "NOT IMPLEMENTED (Fn)" and exit 1 — never exit 0.
- .gitignore: data/raw, data/processed, runs/, *.pt, *.pth, web/dist, node_modules, .venv.
  There is no served model artefact: the API loads runs/<id>/model.pt directly (ADR 0004).
- Windows specifics: use pathlib everywhere; UTF-8 explicitly on every file read/write
  (Windows still defaults to cp1252 in places); no shell-only utilities (grep, sed) in any
  script that the team is expected to run.

## Definition of done
- [ ] `python tasks.py setup` exits 0 on a clean clone
- [ ] `python tasks.py check-device` runs on a machine with no GPU and exits 0
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

## F2.5 — Dry run on synthetic captures (before any lab work)

The labelled database cannot be produced by software: it needs four capture sessions on four
different days with the physical rig. What software **can** do is make sure that on the day the
photos exist, nothing is unknown.

`tools/synth_capture.py` generates fake sessions in `data/raw/SYNTH-*` with real ArUco markers,
per-session lighting and material, and camera jitter. They exercise the entire chain: marker
detection, homography, the quality gate, metadata rows, session-wise splitting and the curator
audit. Build and green the whole F3 ingest against them, then delete them.

```bash
python tools/synth_capture.py --sessions 4 --frames 20 --classes 6
python tasks.py ingest --session SYNTH-20260101-01
python tasks.py split
# curator audit -> READY TO TRAIN
rm -rf data/raw/SYNTH-*        # before the first real ingest
```

**Never train a reported model on synthetic data.** The images come from a closed-form mode
shape; a CNN reaches ~100 % on them and the number means nothing about a real plate. Ingest must
refuse to mix `SYNTH-*` sessions with real ones in a single dataset, and the curator must report
any synthetic session as a blocking finding once real data exists.

---

## F3a — Build the ingest pipeline (code, NOT blocked by the lab)

**This phase does not wait for a single photograph.** Its inputs are the synthetic sessions from
F2.5. Conflating "the dataset needs captures" with "the code that ingests it needs captures"
idles weeks of work waiting on lab availability — they are different things.

What F3a builds, in dependency order:

| # | Piece | Why it is here |
|---|---|---|
| 1 | `tasks.py modes` | validates `data/modes.csv` → `config/classes.json`; everything downstream needs the class list |
| 2 | **`src/imageio.py`** | decode, EXIF orientation, Display P3 → sRGB, HEIC **and** PNG. `docs/03-preprocessing-spec.md` §5.1 delegates step 1 here and no phase claimed it — a real gap |
| 3 | `tasks.py ingest` | ArUco calibration (§1.1), the quality gate (§6, interior margin), `metadata.csv` rows, `ingest_report.json` |
| 4 | `tasks.py split` | `GroupShuffleSplit` on `session_id` |

Gate: the four synthetic sessions ingest, split, and the curator returns READY TO TRAIN. Then
`rm -rf data/raw/SYNTH-*` and the pipeline waits for real photographs.

`src/imageio.py` is shared by ingest and the API (`docs/05-webapp-spec.md` §3). One
implementation, two callers — the same discipline as `src/preprocess.py`.

---

## F3.5 — Reconcile the documents and the code

Numbered 3.5 because it settles what F3a built. Executed **immediately after F3a and before
F1**, because every one of its six items is something the lab would hit on the first day: F1
assembles the rig, tapes the markers and photographs the calibration frame, so the first time
`cv2.aruco` touches a real photograph is F1, not F3b. A contradiction between a spec and its
implementation costs nothing while it is on paper and costs a session once there are
photographs.

What it settles, and which side won:

| # | Contradiction | Resolution |
|---|---|---|
| 1 | `SYNTH-NN` in code vs `SYNTH-YYYYMMDD-NN` in the plan | the document. A synthetic id must be a real id with a prefix, or the dry run never exercises the parsing of a real one |
| 2 | `check-gpu` asks `is_available()` | the document. `check-device`, running a real matmul and `synchronize()` — the only check that catches a cu124 wheel on a Blackwell card |
| 3 | `check-versions` compares `cv2.__version__` | the document. It must assert the *package*: `cv2.aruco` is contrib-only, and the version string is identical either way |
| 4 | `tools/check_capture.py` in §6, absent from the repo | written, reading `config/quality_gate.json` and calling the same gate as ingest |
| 5 | `[F3]` labels after F3 split | `[F3a]` |
| 6 | geometric corner selection vs the §1.1 table | **the code.** Geometry decides; the table becomes an audit of the rig, and a divergence is a warning in `ingest_report.json` |

Plus: `config/quality_gate.json` stays and is documented; `session.json` becomes a hard
requirement at ingest; the F1 ≥ 10 modes gate becomes an error for real sweeps and a warning
for synthetic ones; and `docs/adr/0006` records what the OpenCV pin is actually worth, in
pixels.

**The rule this phase encodes.** When a document and the code disagree, decide which is right
and change the other — in one commit, with the reason written down. Neither "the code is the
truth" nor "the spec is the truth" is a policy; a repository where the two disagree and nobody
has decided is the only outcome that is always wrong.

---

## F4a — The training bench (code, NOT blocked by the lab)

**Numbered after F3b in the old plan and executed before it, on purpose.** In six months that
order will look like a mistake, so: F4a *trains*, but it does not depend on the laboratory. F1
and F3b consume weeks of calendar time; the code that trains does not have to wait for them,
any more than the code that ingests had to wait in F3a. The synthetic sessions from F2.5 are
enough to exercise every path — the loader, the loss, the early stopping, the run directory,
the six evaluation outputs and the checkpoint gate — and the day the real captures land, the
only unknown left is the physics.

The same argument as F2.5 and F3a, and the same payoff: what F4a caught this time was a
Grad-CAM failure that only appears with `baseline_paper`, which would otherwise have surfaced
at the end of the first real training run.

What F4a builds is `src/model.py`, `src/dataset.py`, `src/train.py`, `src/evaluate.py`,
`src/runs.py`, and the `train`, `eval` and `check-model` targets — greened against the four
synthetic sessions.

**The gate is "it runs end to end and writes the artefacts". It is never an accuracy figure.**
On synthetic captures the network reaches ~100 %, and that number measures how well it
inverted the closed-form mode shape that drew the images. `metrics.json` from a synthetic run
carries a `do_not_report` field saying so, in the file itself, because the person who opens it
in six months will not remember where the data came from.

The corollary matters more than the gate: **do not tune a single hyperparameter by looking at
that number.** §3 of `docs/04-model-spec.md` fixes all of them and §5 is where they move — in
an ablation, on real data, with the result written down. A learning rate chosen because it
improved a score on rendered images is a learning rate chosen at random.

Two things to check rather than assume:

- the head reads `n_classes` from `config/classes.json` (6 with the synthetic sessions, 10–18
  with real ones), never from a constant. The checkpoint stores a class *index*, so a head
  sized for the wrong list answers with a different frequency at full confidence;
- `python tasks.py check-model` is green against the checkpoint the bench produced, and red
  when `config/preprocess.json` is mutated. That is the F5 gate, built here because it is what
  makes F4a verifiable at all.

Then `rm -rf data/raw/SYNTH-*` and delete the synthetic runs, before the first real training
finds them and someone has to read timestamps to tell which is which. The committed trace —
`metadata.csv`, `ingest_report.json`, `modes.csv`, `classes.json`, the calibrations — moves to
`docs/annexes/f3a-dry-run/`, which is where it stays: evidence that the pipeline was proved
before any real data existed, out of every path the pipeline reads.

**`config/classes.json` goes with it, and the live path stays empty until F1 writes it.** Six
invented frequencies sitting in the live config while F1 measures the real ones is a trap: the
class index is frozen and is what a checkpoint stores instead of a frequency, so training
against a stale list produces a model that answers the wrong frequency at full confidence.

---

## F3b — Data collection (lab, the phase that takes calendar time)

**Do externally**

Four sessions minimum, on **different days**, each covering every class. Per session:

1. Fresh sand, fixed mass (record it). Same material within a session.
2. Lock exposure and focus. Photograph the ArUco calibration frame first.
3. Per frequency: set it, wait for the dwell time, capture ≥ 60 frames.
4. Log material, grain mass, distance, lighting, dwell, anything anomalous.
5. Lighting is constant **within** a session and varies **between** sessions — but every
   lighting value must appear in **at least two** sessions. See the warning below.
6. Material likewise: do not give one single session a different granular material.

Do not shortcut this by taking 240 frames in one sitting. Four sittings of 60 is a dataset;
one sitting of 240 is one observation photographed 240 times.

### The trap this plan used to contain — read before booking the lab

An earlier version of this plan said "vary lighting between sessions" and "session 4 uses a
different material". **Combined with a session-wise split, that makes the condition predict the
split perfectly.** Every test image would then be the only one shot under its lighting, and the
reported accuracy would measure generalisation to an unseen lamp angle rather than to an unseen
pattern. The number would be wrong and nothing in the pipeline would say so.

This was caught on synthetic data — the F2.5 dry run reproduced it exactly, and the curator's
condition-confound check flagged it before a single real photograph existed. That is what F2.5 is
for.

**Assign conditions so each value straddles the split.** With four sessions and a 2/1/1 split:

| Session | Lighting | Material |
|---|---|---|
| 1 | `lamp_diffuse` | sand |
| 2 | `lamp_grazing` | sand |
| 3 | `lamp_diffuse` | semolina |
| 4 | `lamp_grazing` | semolina |

Two lighting values × two materials, each in two sessions. Any 2/1/1 split then puts both
lighting values and both materials on both sides. **Two conditions balanced beats three
conditions confounded** — a third lighting value across four sessions cannot be balanced, and
buys less than it costs.

Ablation A8 (material) still works: it compares materials *within* a balanced design instead of
comparing session 4 against the rest, which was never a material comparison at all.

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

## F4b — Train and evaluate on real captures (the team's RTX 5070)

Training is the only step that wants a GPU. It runs on the team's own card, a handful of times,
and the app never touches it (`docs/adr/0005`). Nothing is uploaded anywhere.

**The card is not the constraint.** Training peak is ≈ 474 MB of the 12 GB — 4 %. Two things
follow, and both are easy to get wrong:

- **Preload the whole dataset into VRAM.** 4 000 images as uint8 is 65 MB. A `DataLoader`
  reading files every epoch would leave the GPU idling on Python; indexing a resident tensor with
  a random permutation removes that entirely. Seconds per epoch, all 27 runs in under an hour.
- **Do not let 64 GB of RAM justify anything.** Bigger batches, bigger images and a bigger model
  need a reason from the ablations, never from the hardware.

**Before the first run, verify the GPU actually executes.** RTX 50-series is Blackwell, sm_120;
`cu121`/`cu124` PyTorch builds have no kernels for it, report `cuda: True`, and then die on the
first operation. `python tasks.py check-device` runs a real matmul precisely to catch this. Use
`requirements-train.txt` (cu128 or newer) on the training machine only.



**Prompt**

```
## Context
Phase F4b. Curator says READY TO TRAIN. ≥ 4 real sessions ingested.
The training bench from F4a exists and is green on synthetic data.
Read: docs/04-model-spec.md.
Do NOT read: docs/01, 02, 03, 05, 06, 07.

## Task
Train wittenberg_v1 and baseline_paper on the real captures and produce the full
evaluation set. The bench is built; this phase produces the numbers.

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
- [ ] EVERY acceptance run covers BOTH models — train, eval and check-model, baseline_paper
      as well as wittenberg_v1. Not "the one that ships, plus the baseline when convenient".
- [ ] session-wise accuracy ≥ 80 %, worst-class recall ≥ 60 % (or a written explanation)
- [ ] confusion matrix as PNG and CSV
- [ ] Grad-CAM for 3 correct + 3 incorrect per class, for both models
- [ ] accuracy reported with a 95 % Wilson interval

## Model
opus — the split logic and the evaluation design are where this phase goes wrong.
```

**Every acceptance run goes through both models.** `baseline_paper` is not decoration waiting
for A7: it is a second, differently-shaped path through the same code, and it fails where
`wittenberg_v1` does not. F4a found exactly that — Grad-CAM's backward hook makes the hooked
layer's output a view, and `baseline_paper` puts its `ReLU` directly after that layer while
`wittenberg_v1` has a `BatchNorm` in between. Running only the model that ships would have put
that discovery at the end of the first real training run, on the GPU machine, with the lab work
already spent.

**Then the ablations** — one prompt each, `docs/04-model-spec.md` §5, 3 seeds, mean ± std.
A4 (random vs session split) first: its result determines how every other number is read.

---

## F5 — Local serving API

FastAPI on `127.0.0.1`, PyTorch model **on CPU**, calling `src/preprocess.py`. Prompt: the
worked example in `PROMPTS.md`.

**Gate** — `python tasks.py check-model` green; `/api/predict` on a reference image returns exactly
what calling `preprocess` + the model directly returns; mutating `config/preprocess.json` makes
startup fail; `elapsed_ms` < 150 ms on an ordinary CPU.

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
full 13-stage pipeline viewer, the /dades labelling view, and the shell for
/apren, /galeria and /sobre.

## Constraints
- Vite + React + TypeScript + Tailwind, react-router. No Next.js, no SSR.
- Built output is served by FastAPI; `npm run dev` proxies /api to :8000.
- Result presentation follows §4 exactly: top-3 with probabilities, the three
  confidence tiers, and the edge map beside the original. Never a bare number,
  never a rounded "nicer" one, never a hidden low confidence.
- Pipeline viewer per §8: THIRTEEN stages from ONE request — 8 preprocessing,
  3 feature-map, 1 Grad-CAM, 1 probabilities — animated at ~1.1 s per step with
  play/pause, arrows and a thumbnail rail. Do NOT stream them: preprocessing is
  ~1.3 ms and the network ~5-11 ms (measured, docs/05 §8), so there is nothing
  to watch live. Show each step's real measured time — every stage in the
  response carries its own `ms`. Steps 7-8 render pixelated.
- /dades per §9: draft session, calibration frame, one label per class, the
  gate applied AS IMAGES ARRIVE with the reason shown for each rejection, and
  nothing written until commit. It must call the SAME ingest code as
  `tasks.py ingest` — a second implementation of the gate is forbidden.
- The OpenCV call and parameters shown in the viewer are read from
  config/preprocess.json, never retyped into the UI.
- Catalan UI strings via docs/07-glossary-ca-en.md. English code.
- No localStorage, no cookies, no analytics — anywhere, ever.

## Definition of done
- [ ] upload -> prediction works against the real API
- [ ] the three confidence tiers render correctly (unit-tested on fixed inputs)
- [ ] the viewer walks all 13 stages; arrows and space work; 7-8 are pixelated
- [ ] /dades: a session can be labelled, gated and committed; abandoning it
      leaves data/ byte-identical; the gate agrees with tasks.py ingest on
      tests/fixtures/
- [ ] the whole app starts and predicts on a machine with NO GPU and NO network
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
- **Measure `elapsed_ms` on that laptop**, from the built app, on a real capture, with all
  thirteen stages — not with `?stages=false`, and not on a development machine. The 150 ms gate
  of `docs/05` §7 is a claim about the computer the session runs on. Measured during F5 on a
  development machine: 82–129 ms with every stage, of which ≈ 43 ms is PNG-encoding them.
  A slower laptop can fail the gate on the encode phase alone while the model is untouched, and
  `timings_ms` in any response says immediately whether that is what happened. If it fails,
  `docs/05` §8 names the two cheap levers; do it before `v1.0-field` is frozen, never during
  the study.

---

## Cross-cutting

**Weekly** — `wittenberg-dataset-curator` (catches drift a single session hides).
**Before any number enters the TDR** — `wittenberg-repro-auditor`, without exception.
**Every preprocessing change** — bump the version, retrain, and expect `check-model` to reject
old checkpoints. That rejection is the guard working, not an obstacle.
**Every reversed decision** — an ADR, before the reversal. This plan has already reversed twice
(0003, 0004); both times the ADR made the blast radius obvious in minutes.
