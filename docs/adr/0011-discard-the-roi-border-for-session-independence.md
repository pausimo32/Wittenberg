# ADR 0011 — Discard the ROI border so the model reads the pattern, not the session

- **Date:** 2026-11-08
- **Status:** accepted
- **Amends** `docs/03-preprocessing-spec.md` (spec 1.1.0 → 1.2.0, step 6) and ADR 0010 (the
  shipped run). Every 1.1.0 checkpoint is refused by the startup gate, as `docs/03` §6 intends.
- **Requirement (team):** the model need not work on another plate; it **must** work on a new
  capture session with this plate.
- **Evidence:** `experiments/protocol_v3/`, `runs/protocol_v3/metrics.json`; the diagnosis is in
  `runs/protocol_v2/exploratory_masking.json`.

## Context

The v2 network (ADR 0010) ignored the ArUco markers (hiding the corners changed nothing), but
it read the band at the edge of the ROI: with that band hidden, 2241 Hz fell from 100 % to
39 %. That band holds the plate's rim, the table and the markers, all of which differ between
sessions. 2241 Hz is the only class shot in session 04, so the model had partly learned
"session 04" in place of "mode at 2241 Hz". That fails on a new session.

## Decision

1. `src/preprocess.py` step 6 discards Canny edges within `canny.border_frac` = 0.0625 of each
   side (32 px of the 512 ROI, 8 px of the 128 input). The rim was measured at 5–6 px of 128.
   Edges are zeroed after Canny so that no artificial edge appears at the boundary.
2. `wittenberg_v1` is retrained with the ADR 0010 recipe (A2 = D4 + A1, batch 8, seed 42) on
   the re-ingested data. The run `20261108-232629` is the app's model.

## Consequences

- Out-of-fold on the same splits: **accuracy 0.968 (121/125), Wilson 95 % 0.921–0.987**;
  group-weighted macro-F1 0.974 (bootstrap 0.92–1.00); recall 497 0.95 / 863 0.98 /
  1218 0.97 / 2241 1.00; seeds 17/29 0.976. D4 test 0.971; raw photo at ×0.6 / ×1.4 light
  0.968 / 0.952. Removing the border *improved* the model (v2: 0.944).
- **One pre-registered criterion fails:** a network trained on pixel-shuffled inputs (shape
  destroyed, edge density kept) still recognises 2241 Hz at 0.83 (limit 0.50). Edge density
  is partly physical (2241 is the highest mode and draws the most nodal lines), and may be
  partly the session (sand amount, light). These photos cannot separate the two, and the
  result is reported as such.
- Cost: nodal lines are not seen where they meet the plate's edge.
- What is shown is that the identified session cues are gone. It does **not** show performance
  on a new session. Only a new session can show that (prospective confirmation, ADR 0009).
