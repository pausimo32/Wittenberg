# ADR 0010 — Ship wittenberg_v1, trained with D4 symmetry augmentation

- **Date:** 2026-11-08
- **Status:** accepted (team decision); shipped run **superseded by ADR 0011** (same recipe,
  preprocessing 1.2.0 without the ROI border; accuracy 0.968).
- **Amends** ADR 0009 decision 5 (the selection rule ranks; it does not by itself decide what
  ships) and `docs/04-model-spec.md` §1, §3, §6. ADR 0004/0005 untouched: CPU-only, offline.
- **Evidence:** `experiments/protocol_v2/` (pre-registration, `run_v2.py`) and
  `runs/protocol_v2/metrics.json`; v1 comparison in `runs/protocol_v1/metrics.json`.

## Context

ADR 0009's pre-registered rule picked HOG + logistic regression (macro-F1 0.989). It is not a
neural network, and it carries no physical prior: half of its evidence came from the plate's
border band, and nothing ties it to the shape of a mode. The research question (Pla de
treball P1) is about the team's own CNN, and the team wants a model whose behaviour can carry
over to other setups, accepting some loss of accuracy.

In v1, `wittenberg_v1` scored 0.83 with a diagnosed failure: at batch 32 and ~55 training
images, 2–3 optimiser steps per epoch left BatchNorm unsettled; several seeds stopped at
epoch 1 predicting one class.

## Decision

1. **Ship `wittenberg_v1`**, architecture exactly as `docs/04` §1 B.
2. **Train it with A2** = a uniformly drawn D4 symmetry of the square (quarter turns,
   reflections) + A1 (±5°, ±2 %, brightness/contrast 0.9–1.1). Justification: a square,
   centre-driven plate has a D4-invariant operator, so the patterns at one frequency are closed
   under D4. Caveat (protocol R1): imperfections can break that symmetry. The assumption is
   deliberate, made for orientation and rig independence.
3. **Batch 8** instead of 32 (more steps per epoch for BatchNorm). Everything else per `docs/04`
   §3. One network, seed 42, chosen before results; seeds 17 and 29 only measure variability.
4. Final model refitted on all 125 photos with epochs from the inner splits (protocol p. 27).
5. HOG + logistic regression stays in the code (`src/hoglr.py`, `tasks.py train-hoglr`) as the
   measured reference. The API serves either kind of run; `runs/latest.txt` decides which.

## Consequences

- 2026-11-08, out-of-fold on the v1 splits (`runs/protocol_v2/metrics.json`): **accuracy 0.944
  (118/125), Wilson 95 % 0.889–0.973 (frame level)**; group-weighted macro-F1 0.954
  (bootstrap 0.88–1.00); recall 497 0.98 / 863 0.95 / 1218 0.89 / 2241 1.00. Seeds 17/29:
  0.968 / 0.984. Robustness: all 8 D4 symmetries of the test images 0.955; raw photo at
  ×0.6 / ×1.4 lighting 0.944 / 0.912.
- **Exploratory, not confirmatory:** v2 was designed after seeing v1, on the same photos.
- 2241 Hz remains confounded with its session: a network trained on the corners alone still
  recognises it (recall 0.93) while the other three stay near chance.
- "Serves in other contexts" means robust to orientation, symmetry and lighting, as measured
  above. Transfer to another plate is **not** measured (one plate exists). It would also need
  that plate's own frequency for each mode shape.
- App: `/api/predict` ≈ 150–180 ms per request on CPU with Grad-CAM, at the edge of the
  `docs/05` §7 budget of 150 ms.
