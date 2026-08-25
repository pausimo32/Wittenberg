# ADR 0004 — Local-only application: FastAPI + PyTorch, one preprocessing implementation

- **Date:** 2026-08-17
- **Status:** accepted. **Supersedes ADR 0001.**
- **Trigger:** the app is not for public deployment. It runs on the team's own machine and is
  used by the team. Participants only touch Google Forms, from their own devices.

## Context

ADR 0001 chose browser-side inference (`opencv.js` + ONNX Runtime Web) over a Python server.
Every one of its three reasons was about *distribution*:

| ADR 0001 reason | Still true? |
|---|---|
| No hosting budget | **Moot.** `localhost` costs nothing. |
| Must work offline on unreliable school Wi-Fi | **Moot.** Localhost is offline by construction. |
| Minors' photos must not be uploaded | **Moot.** Nothing leaves the machine either way. |

And the cost ADR 0001 accepted — an entire phase (F6) spent making two preprocessing
implementations agree, plus a parity harness, a golden set, an audit subagent and a ~12 MB WASM
chunk — was paid *only* to buy those three properties.

There is also a new fact: the machine that runs the app is the machine with the CUDA GPU.

## Decision

**One process, one language, one preprocessing implementation.**

- **Backend** — FastAPI on `127.0.0.1`, loading the trained PyTorch model onto the GPU once at
  startup, and calling `src/preprocess.py` — *the same function used to build the training set*.
- **Frontend** — Vite + React + TypeScript, built to static files that FastAPI serves. Next.js
  was chosen for its static export and free hosting; neither applies now, and its App Router and
  SSR machinery are complexity with no return on `localhost`.
- **No ONNX, no quantisation, no `opencv.js`, no parity gate.**

## Consequences

**Train/serve preprocessing skew — the risk this project was organised around — is eliminated
rather than mitigated.** Not "reduced to within a tolerance": eliminated, because inference calls
the identical Python function on the identical machine. This is the single biggest simplification
available to the project, and it is only available because the app is local.

Removed:

- Phase F6 (browser preprocessing) in its old form
- The 20-image parity set `tests/golden/`, `scripts/parity_check.py`, the `python tasks.py
  parity` gate. (F2 and F5 still use single-image fixtures, now under `tests/fixtures/`.)
- `opencv.js`, `onnxruntime-web`, `onnx`, `onnxruntime`, the INT8 export step
- The old parity-threshold `§4` of `docs/03-preprocessing-spec.md` (that section number has been
  **rewritten** as the version-consistency guard, which is live and mandatory) and the
  cross-language traps that used to make up most of `§5`
- The `wittenberg-parity-auditor` subagent — **retired, not repurposed**. It existed to audit a
  divergence that can no longer occur. It is replaced by `wittenberg-repro-auditor`, which
  audits what is now the highest-severity failure: reporting a number in the TDR that cannot be
  reproduced or traced to a run.

Kept, unchanged:

- The whole of `docs/02-dataset-spec.md`, `04-model-spec.md`, `06-research-protocol.md`
- `docs/03-preprocessing-spec.md` §1–§3 — the pipeline and its parameters are still normative,
  and `config/preprocess.json` is still the only place they are defined. One implementation does
  not mean no specification: the spec is what makes the pipeline reproducible by a reader.
- The nine ablation variants, Grad-CAM, session-wise splitting, the acceptance gates
- The didactic module, which still drives research questions P2 and S3

**The critical path moves.** It was F2 → F6 (preprocessing parity). It is now **F2 → F4** — the
science. Nothing downstream of F4 can make the results better; it can only present them.

**One risk replaces the one we removed.** With no parity gate, nothing structurally prevents
serving a model whose weights were trained under a *different* preprocessing version. The
mitigation is cheap and mandatory: every checkpoint records the resolved
`config/preprocess.json` and its hash, and the server refuses to start if the running config's
hash differs. Asserted by the API in its own startup hook; `python tasks.py check-model` runs
the same check from the command line.

## What to say in the TDR

Do not present browser-side inference as something that was tried and failed. Present it as
what it was: a design considered for a public deployment, made unnecessary by the decision to
keep the tool local. Then state the consequence honestly — because training and inference share
one implementation, the accuracy reported for the model *is* the accuracy of the tool, with no
gap to argue about. That is a stronger claim than the parity gate would have bought.
