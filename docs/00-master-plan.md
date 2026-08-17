# Master plan — Wittenberg

Eight phases, F0 → F7. Each has an exit gate. Do not start a phase until the previous gate is
green: every gate exists because failing it silently costs more than failing it loudly.

`/clear` between phases.

| Phase | Outcome | Gate | Model |
|---|---|---|---|
| F0 | Repo + Claude Code configured | `make setup` green, 7 agents listed | sonnet |
| F1 | Modes found, classes fixed | `data/modes.csv` ≥ 10 modes | lab + haiku |
| F2 | Python preprocessing | unit tests green on golden set | opus → sonnet |
| F3 | ≥ 4 ingested sessions | curator says READY TO TRAIN | sonnet |
| F4 | Trained + evaluated model | acceptance gates in model-spec §6 | opus |
| F5 | ONNX export | quantised ≈ Keras, ≤ 4 MB | sonnet |
| F6 | Browser preprocessing | **`npm run parity` green** | sonnet + parity-auditor |
| F7 | Web app + didactic module | Lighthouse, E2E, offline, frozen tag | sonnet |

The critical path is **F2 → F6**. Everything else can slip; a preprocessing divergence
invalidates the project's central claim.

---

## F0 — Foundations

Repo skeleton, `CLAUDE.md`, `docs/`, `.claude/agents/`, `.claude/skills/`, Makefile, Python and
Node toolchains, `config/preprocess.json`, `config/versions.json` with OpenCV pinned on both sides.

**Prompt**

```
## Context
Phase F0. Empty repo. The docs/ and .claude/ trees from the Wittenberg starter kit are in place.
Read: CLAUDE.md only.
Do NOT read: docs/.

## Task
Set up the toolchain so `make setup` produces a working Python and Node environment.

## Constraints
- Python 3.11, pinned requirements.txt: opencv-python, tensorflow, scikit-learn, pandas,
  numpy, matplotlib, tf2onnx, onnxruntime, pillow-heif, pytest.
- Node 20, Next.js App Router + TypeScript + Tailwind, output: 'export'.
- config/versions.json records the opencv-python version AND the opencv.js build to fetch.
  They must be the same minor version. Assert it in `make setup`.
- Makefile targets: setup, ingest, split, train, eval, export, annexes, parity, clean.
  Targets not yet implementable print "NOT IMPLEMENTED (Fn)" and exit 1 — never exit 0.
- .gitignore: data/raw, data/processed, runs/, *.h5, *.keras, node_modules, .venv,
  and `*.onnx` WITH the exception `!web/public/model/*.onnx` — the served model is a build
  input for a static export, so a fresh clone must be able to build a working app.
- `make train` updates a `runs/latest` symlink to the newest run directory.

## Definition of done
- [ ] `make setup` exits 0 on a clean clone
- [ ] `python -c "import cv2, tensorflow"` works
- [ ] `npm run build` produces a static export
- [ ] version assertion fails loudly if the two OpenCV versions are made to disagree

## Model
sonnet — mechanical scaffolding.
```

**Do externally**

1. Install Claude Code (`npm i -g @anthropic-ai/claude-code`), `claude` in the repo root, `/login`.
2. `/plugin marketplace add anthropics/skills` then `/plugin install document-skills@anthropic-agent-skills`
3. `/plugin install superpowers@claude-plugins-official`
4. `/plugin install pyright-lsp@claude-plugins-official`, `typescript-lsp@claude-plugins-official`,
   `code-review@claude-plugins-official`, `frontend-design@claude-plugins-official`
5. Import the four external subagents (see `SETUP.md` §3)
6. `/agents` — confirm 7 agents: the three `wittenberg-*` plus the four imported
6. `/context` — record the baseline cost of CLAUDE.md + skill descriptions
7. `git init && git commit` the skeleton before anything else

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
  after each of steps 3-8 — the parity harness and the auditor both need this.
- Pure function. No file I/O, no globals, no logging inside the transform.
- HEIC/EXIF/P3 handling lives in the ingest layer, not here (§5.4).
- pytest covering: each step's output shape and dtype; determinism (same input → identical
  bytes over 10 runs); both roi_modes; a 4-marker-missing failure path.

## Definition of done
- [ ] pytest green
- [ ] `grep -nE '[^.0-9](0\.|[0-9]{2,})' src/preprocess.py` finds no transform constants
- [ ] preprocess_stages returns 6 arrays with documented shapes

## Model
opus for the design of preprocess_stages and the calibration/ROI logic; sonnet for the rest.
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
- Splits by session only (`make split`). Never hand-edit the split column.

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
- Split by session (GroupShuffleSplit on session_id). Never random. §2 of the ADR is not
  negotiable.
- Seed 42 + tf.config.experimental.enable_op_determinism().
- runs/<ts>/ contains config.json (with the preprocessing version), metrics.json, history.csv,
  model.keras, report/.
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

## F5 — Export

```
## Context
Phase F5. wittenberg_v1 is trained and passes its gates.
Read: docs/04-model-spec.md §7.

## Task
Implement `make export`: Keras → ONNX (tf2onnx, opset 17) → dynamic INT8 → web/public/model/.

## Constraints
- The bundle carries the preprocessing version from config/preprocess.json.
- Verify quantised vs float on the full test set before writing the file.

## Definition of done
- [ ] argmax agreement 100 %, max softmax diff ≤ 1e-2
- [ ] file ≤ 4 MB
- [ ] classes.json + preprocess.json copied alongside

## Model
sonnet.
```

---

## F6 — Browser preprocessing (critical path)

The phase the whole architecture exists to make possible. Prompt: the worked example in
`PROMPTS.md`.

Sequence: implement → `npm run parity` → on failure invoke the `preprocessing-parity` skill →
if the cause is not in §5, escalate to `wittenberg-parity-auditor` on opus at `xhigh`.

Do not proceed to F7 until the gate is green. A red gate here means every accuracy figure in
the TDR describes a system the user cannot actually run.

---

## F7 — Web app and didactic module

Four prompts, in order, one per surface. `docs/05-webapp-spec.md`.

1. **Shell + `/analitza`** — worker, lazy loading, version assertion, three-tier confidence
   presentation, original-vs-edge-map side by side.
2. **`/apren`** — the seven concepts in order (`docs/01-domain-physics.md` §4), WebAudio muted
   by default, real captured sweep, offline, no storage of anything personal.
3. **`/galeria` + `/sobre`** — generated from `metadata.csv` at build time.
4. **Hardening** — Lighthouse ≥ 90/95, Playwright E2E including the zero-network assertion,
   accessibility pass, then `git tag v1.0-field`.

**Do externally**

- Freeze `v1.0-field` before the first field session and never patch it mid-study.
- Test on the actual devices the schools will use, on the actual Wi-Fi, offline.
- Consent forms and Google Forms ready; random participant codes, no names anywhere.

---

## Cross-cutting

**Weekly** — `wittenberg-dataset-curator` (catches drift a single session hides).
**Every preprocessing touch** — `wittenberg-parity-auditor`, without exception.
**Every reported number** — traceable to `runs/`, per `docs/06-research-protocol.md` §4.
**Every reversed decision** — an ADR, before the reversal.
