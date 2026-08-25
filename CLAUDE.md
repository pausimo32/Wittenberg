# Wittenberg

Web tool that predicts the excitation frequency of a Chladni plate from a photo of its
nodal pattern, and explains wave physics to people with no physics background.
Research artefact for a TDR (Catalan high-school research project).

## Hard rules

1. **`src/preprocess.py` is called by BOTH training and the API — never fork it.** If the API
   needs a transform changed, change it there and retrain. Any edit requires updating
   `docs/03-preprocessing-spec.md` in the same commit and bumping the version.
2. **Never use `cv2.imshow`, `plt.show()` or any blocking call.** Write artefacts to `runs/`.
3. **Never split the dataset randomly.** Split by `session_id` (see `docs/02-dataset-spec.md`).
   A random split leaks near-duplicate frames and inflates accuracy — this is a reported
   finding of the project, not a bug to work around.
4. **Never commit** `data/raw/`, `data/processed/`, `runs/`, `*.pt`, `*.pth`, `web/dist/`.
   Commit `data/metadata.csv`, `runs/*/metrics.json` and `runs/*/config.json`.
5. **Deterministic by default.** Seed 42 everywhere (`random`, `np.random`, `torch`), plus
   `torch.use_deterministic_algorithms(True)`. Report the seed in every `metrics.json`.
6. UI strings are **Catalan**. Code, comments, commit messages and docs are **English**.
   Use `docs/07-glossary-ca-en.md` for physics terms — do not invent translations.

## Environment

**Native Windows, no GPU, no WSL.** The app must run on ANY computer the team picks — CPU only,
no API key, no internet (`docs/adr/0005`). Training runs on the team's RTX 5070, offline, and is
not part of the app. Three consequences that bite if forgotten:

- There is **no `make`**. The task runner is `python tasks.py <target>` — pure Python, no
  install, same on every OS. Never write a Makefile or tell the user to run `make`.
- There are **no symlinks**. `runs/latest` is a pointer file, `runs/latest.txt`, containing the
  run directory name. Use `latest_run()` in `tasks.py`; never `os.symlink`.

- **Never require CUDA.** `device = "cuda" if torch.cuda.is_available() else "cpu"`, and CPU is
  the supported configuration, not a degraded one: inference is ≈ 12 ms there.

Prefer `pathlib` over string paths, and never hard-code `/` separators.

## Stack

**Local, portable, CPU-only app** (`docs/adr/0004`, `0005`). One process, on whichever computer
the team picks. Never deployed, never bound to `0.0.0.0`, never calls out to the network.

- **Training** — Python 3.11, PyTorch, `opencv-contrib-python` 4.12 (contrib: `cv2.aruco`),
  scikit-learn, pandas. Runs on the team's **RTX 5070** (peak ≈ 474 MB of 12 GB). Use
  `requirements-train.txt` there — **cu128 or newer**, or Blackwell has no kernels
- **Serving** — FastAPI on `127.0.0.1`, PyTorch model **on CPU**, calling `src/preprocess.py`.
  1.17 MB checkpoint, travels with the repo
- **Preprocessing** — one implementation, `cv2`. No second language, no parity gate.
- **Frontend** — Vite + React + TypeScript + Tailwind, built static, served by FastAPI

## Commands

| Command | What it does |
|---|---|
| `python tasks.py --list` | every target and what it does |
| `python tasks.py setup` | venv, deps, version and device checks |
| `python tasks.py ingest` | raw captures → `data/processed/` + `data/metadata.csv` |
| `python tasks.py train` | trains the CNN, writes `runs/<ts>/` (cloud GPU, or slow on CPU) |
| `python tasks.py eval` | confusion matrix + per-class report → `runs/<ts>/report/` |
| `python tasks.py check-model` | **gate.** checkpoint's preprocessing hash == current config |
| `python tasks.py app` | build if stale, start the local server, open the browser |
| `npm run dev` | Vite dev server (inside `web/`, dev only — never for a field session) |

## Documentation index — read on demand, never all at once

Load a doc **only** when the task touches it. Each is self-contained.

| Doc | Read it when you are… |
|---|---|
| `docs/01-domain-physics.md` | writing didactic copy, labelling patterns, choosing frequencies |
| `docs/02-dataset-spec.md` | capturing, ingesting, naming, splitting or auditing data |
| `docs/03-preprocessing-spec.md` | touching **any** image transform |
| `docs/04-model-spec.md` | changing architecture, training, metrics or acceptance gates |
| `docs/05-webapp-spec.md` | building the API, the UI, or the didactic module |
| `docs/06-research-protocol.md` | producing anything that goes into the TDR document |
| `docs/07-glossary-ca-en.md` | writing user-facing Catalan strings |
| `docs/adr/*.md` | about to reverse a past decision — check it was not already litigated |

If a task needs more than **two** of these, stop and split the task.

## Delegation

Prefer subagents for anything that reads many files: `wittenberg-dataset-curator` for data
audits, `wittenberg-repro-auditor` before any number goes into the TDR, `wittenberg-tdr-writer`
for Catalan prose. They return summaries, not file dumps.
