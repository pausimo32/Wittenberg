# Wittenberg

Web tool that predicts the excitation frequency of a Chladni plate from a photo of its
nodal pattern, and explains wave physics to people with no physics background.
Research artefact for a TDR (Catalan high-school research project).

## Hard rules

1. **`docs/03-preprocessing-spec.md` is the single source of truth.** Python and browser
   preprocessing MUST produce identical tensors. Never change one side alone. Any edit to
   preprocessing requires re-running `npm run parity` and updating the spec in the same commit.
2. **Never use `cv2.imshow`, `plt.show()` or any blocking call.** Write artefacts to `runs/`.
3. **Never split the dataset randomly.** Split by `session_id` (see `docs/02-dataset-spec.md`).
   A random split leaks near-duplicate frames and inflates accuracy — this is a reported
   finding of the project, not a bug to work around.
4. **Never commit** `data/raw/`, `data/processed/`, `runs/`, `*.h5`, `*.onnx` > 5 MB. Use `dvc`
   or leave them local; commit only `data/metadata.csv` and `runs/*/metrics.json`.
5. **Deterministic by default.** Seed 42 everywhere (`np.random`, `tf.random`, `random`).
   Report the seed in every `metrics.json`.
6. UI strings are **Catalan**. Code, comments, commit messages and docs are **English**.
   Use `docs/07-glossary-ca-en.md` for physics terms — do not invent translations.

## Stack

- **Training** — Python 3.11, `opencv-python` 4.x, TensorFlow/Keras 2.x, scikit-learn, pandas
- **Inference** — ONNX Runtime Web (WASM) in the browser; no server, no upload
- **Preprocessing (both sides)** — OpenCV. Python `cv2`, browser `opencv.js` (same C++ core)
- **Web** — Next.js (App Router), TypeScript, Tailwind, static export

## Commands

| Command | What it does |
|---|---|
| `make ingest` | raw captures → `data/processed/` + `data/metadata.csv` |
| `make train` | trains the CNN, writes `runs/<ts>/` |
| `make eval` | confusion matrix + per-class report → `runs/<ts>/report/` |
| `make export` | Keras → ONNX + quantised, into `web/public/model/` |
| `npm run parity` | **gate.** Python vs browser preprocessing equivalence |
| `npm run dev` | Next.js dev server |

## Documentation index — read on demand, never all at once

Load a doc **only** when the task touches it. Each is self-contained.

| Doc | Read it when you are… |
|---|---|
| `docs/01-domain-physics.md` | writing didactic copy, labelling patterns, choosing frequencies |
| `docs/02-dataset-spec.md` | capturing, ingesting, naming, splitting or auditing data |
| `docs/03-preprocessing-spec.md` | touching **any** image transform, either language |
| `docs/04-model-spec.md` | changing architecture, training, metrics or acceptance gates |
| `docs/05-webapp-spec.md` | building UI, routes, inference glue or the didactic mode |
| `docs/06-research-protocol.md` | producing anything that goes into the TDR document |
| `docs/07-glossary-ca-en.md` | writing user-facing Catalan strings |
| `docs/adr/*.md` | about to reverse a past decision — check it was not already litigated |

If a task needs more than **two** of these, stop and split the task.

## Delegation

Prefer subagents for anything that reads many files: `wittenberg-parity-auditor` for
preprocessing equivalence, `wittenberg-dataset-curator` for data audits,
`wittenberg-tdr-writer` for Catalan prose. They return summaries, not file dumps.
