# Wittenberg

Web tool that predicts the excitation frequency of a Chladni plate from a photo of its
nodal pattern, and explains wave physics to people with no physics background.

Research artefact for a TDR (Catalan high-school research project). The user interface is
Catalan; the code, the comments and the documentation are English.

## What it is

One local process on one computer: FastAPI on `127.0.0.1`, a 1.17 MB PyTorch checkpoint on
the **CPU**, and a Vite + React front end built to static files and served by the same
process. No GPU, no API key, no network — see `docs/adr/0004` and `docs/adr/0005`.

Training is a separate activity that runs offline on the team's own GPU and is not part of
the app.

```
config/       preprocessing, classes, calibration and quality-gate parameters
data/         captures, per-session calibrations, metadata.csv, modes.csv
docs/         the specifications; docs/adr/ the decisions
runs/         one directory per training run
src/          preprocess, imageio, calibrate, quality, ingest, split, metadata, modes
tools/        synth_capture.py, check_capture.py — lab and dry-run helpers
tasks.py      the task runner (there is no make)
web/          the front end
```

## Getting started

```bash
python tasks.py setup      # venv, dependencies, version and device checks
python tasks.py --list     # every target and what it does
```

`python tasks.py app` builds the front end if it is stale, starts the local server and
opens the browser.

There is no `make` and there are no symlinks anywhere: the project must run unchanged on
native Windows without WSL.

## Documents

- `CLAUDE.md` — the rules that survive a `/clear`
- `docs/00-master-plan.md` — the eight phases and their gates
- `docs/02-dataset-spec.md` — capture, ingest, splitting
- `docs/03-preprocessing-spec.md` — the preprocessing contract
- `docs/04-model-spec.md` — architecture, training, acceptance gates
- `docs/06-research-protocol.md` — what the written work needs
- `WITTENBERG_CONTEXT.md`, `WITTENBERG_BRAND_GUIDE.md` — scope and visual identity
