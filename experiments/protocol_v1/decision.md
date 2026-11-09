# Decision — chladni_v1_20261108

> **Historical.** This is the v1 rule's pick. What ships is decided in ADR 0010 and ADR 0011
> (`wittenberg_v1`, preprocessing 1.2.0); HOG + logistic regression is kept as the measured reference.

Protocol "CNN para patrones de Chladni" v1.0, pages 21–27. All numbers are from
`runs/protocol_v1/metrics.json` (prereg SHA-256 recorded there). Nothing here is a confirmed
model: status is **provisional** until the prospective session below exists.

## Scope

4 classes (497, 863, 1218, 2241 Hz), 125 originals, one plate, one day. Evaluated use:
**a new sand preparation in the same session, same rig**. Not evaluated: new sessions, new
days, other plates, 165 Hz (no captures).

Independent unit: sand-replenishment block. 13 groups (`runs/protocol_v1/metrics.json`
`n_groups`): 10 in session 20261108-03, each one multiclass, shared by 497/863/1218, which
were interleaved; 3 for 2241, alone in 20261108-04.
Nested CV: outer 3 folds, inner 2 folds, round-robin blocks (amendment A1).

## Results (OOF, group-weighted macro-F1; bootstrap of groups, 1 939/2 000 valid replicates)

| Candidate family (best config per fold by inner score) | Macro-F1 | 95 % bootstrap | Min recall | CPU latency |
|---|---|---|---|---|
| B0 dummy | 0.167 | 0.11–0.21 | — | — |
| **B1 HOG + logistic regression** | **0.989** | 0.96–1.00 | 0.977 | 0.37 ms |
| C1 small CNN (protocol p.15) | 0.711 | 0.59–0.81 | 0.550 | 1.2 ms |
| W1 wittenberg_v1 (docs/04) | 0.832 | 0.64–0.96 | 0.674 | 4.3 ms |
| F1 MobileNetV3-S embeddings + linear | 0.988 | 0.95–1.00 | — | 4.4 ms |
| F2 EfficientNet-B0 embeddings + linear | 0.989 | 0.96–1.00 | — | 10.3 ms |
| **Nested selector (the reportable estimate)** | **0.989** | 0.96–1.00 | 0.977 | — |

Selector OOF in counts: 123/125 correct (one 497→1218, one 863→497). It chose B1 in all
three outer folds and again on all development data; the tie set (within δ = 0.03) was B1, F1
and F2, and B1 wins on measured latency. Family rows are exploratory (protocol p.9): picking
the highest of them afterwards would be optimistic.

## Controls

| Control | Macro-F1 | Reading |
|---|---|---|
| Random frame split (leakage diagnostic) | 0.976 | Close to the grouped estimate: frame leakage is not what drives the score here. |
| Background ring, outer 8 px (pre-registered) | 0.969 | **Trips the insufficient-evidence rule (≥ 0.60).** But the ring is not clean background: it contains the plate edge, where nodal lines of a free-edge plate end at mode-specific positions. |
| Corners only, markers + table (exploratory, post hoc) | 0.470 | Within session 03: 38 % vs 33 % chance, so **no shortcut for 497/863/1218**. 2241: recall 0.65 from corners alone, so **session 04 carries a visual signature** and the 2241 result cannot be attributed to the pattern alone. |

## Decision

- **Provisional candidate: B1, HOG + logistic regression (C = 0.1)**, `runs/protocol_v1/final_model.pkl`
  (SHA-256 `e72f0779…d84b`). Chosen by the pre-registered rule; no composite score.
- **Evidence is sufficient for 497 / 863 / 1218** within the declared scope (interleaved,
  shortcut-free, 10 groups each). **Insufficient for 2241**: one session, three near-burst
  blocks, and a demonstrated session cue.
- The CNNs underperform the linear models on 80 training images. W1 frequently picks epoch 1
  and predicts a single class: consistent with BatchNorm running statistics not settling at
  2–3 optimiser steps per epoch (protocol p.25, "todos predicen una clase"). Changing that
  recipe is a new version and needs a new evaluation; it is not done here.
- Note for the app: the shipped pipeline expects a `wittenberg_v1` checkpoint. Adopting B1
  changes the model contract (`src/api.py`) and docs/04 §2; that is a separate decision (ADR).

## What would resolve the doubt

One new session, captured **after** this freeze, with **all 4 classes (+165 Hz if wanted)
interleaved and shuffled per round**, same rig. Evaluate `final_model.pkl` once, unchanged,
with the same report. That both confirms prospectively and breaks the 2241 ≡ session confound.
