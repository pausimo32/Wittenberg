# ADR 0009 — Grouped nested validation, and a model chosen by a pre-registered rule

- **Date:** 2026-11-08
- **Status:** accepted; decision 5 **amended by ADR 0010**: the rule ranks candidates, an ADR
  decides what ships (the team shipped `wittenberg_v1`).
- **Amends** ADR 0002 (the split unit and the CV scheme; its principle stands),
  `docs/02-dataset-spec.md` §2 and §5, `docs/04-model-spec.md` §1–§4, `CLAUDE.md` rule 3.
  ADR 0003, 0004, 0005 are untouched: training stays local, the app stays CPU-only and offline.
- **Evidence:** `experiments/protocol_v1/` (pre-registration, amendments, code, `decision.md`)
  and `runs/protocol_v1/metrics.json`. Method: the protocol "CNN para patrones de Chladni"
  v1.0 (08/11/2026), adapted.

## Context

The first real captures (2026-11-08) are 125 photos of 4 classes in two sessions: 497, 863 and
1218 Hz interleaved in `20261108-03`, 2241 Hz alone in `20261108-04`. Every class is in exactly
one session. ADR 0002's leave-one-session-out cannot run (a held-out session would contain
classes the model never saw), and `tasks.py split` correctly refuses.

ADR 0002 also fixed the model before any data existed: `wittenberg_v1` ships, pretrained
backbones are excluded on argument (docs/04 §2). With 10–30 photos per class, the protocol
recommends comparing a finite set of families under the same splits and letting a rule
written in advance choose.

## Decision

1. **Never split randomly — unchanged.** Group by the *independent unit*, chosen by what the
   data can support, and limit the claim to that unit:
   - `session_id` when every class appears in ≥ 3 sessions (ADR 0002, the preferred case);
   - otherwise the **preparation block** (the frames between two sand replenishments,
     `reseed_every` in `session.json`), and the result is stated as "new preparation, same
     session/plate", never "new session".
   Falling back to a smaller unit to obtain a number is forbidden. The fallback exists only
   because the protocol (p. 6) defines it with an explicitly weaker claim.
2. **Nested grouped cross-validation.** Outer K folds, inner folds strictly inside each outer
   train, K set by G_min (the fewest groups any class has): G_min = 2 → outer 2, no inner
   selection; G_min = 3 → 3 × 2; G_min ≥ 4 → 3–5 × 2–3. Every fold must contain every class
   on both sides; the assignment uses labels and groups only, never results.
3. **Pre-registration.** Candidates, seeds, metric, thresholds and the tie rule are written
   and hashed before the first fit. Later changes go to `amendments.md` with the reason.
4. **Metric.** Primary: macro-F1 with weight `1 / n_originals_in_group`, over pooled
   out-of-fold predictions. Uncertainty: bootstrap of whole groups (descriptive). Accuracy
   with a Wilson interval is still reported when the unit is the session.
5. **Model choice.** `wittenberg_v1` becomes a **candidate**, not the fixed answer. The shipped
   model is the one the rule picks: eligible candidates within δ = 0.03 of the best inner
   score → lowest measured CPU latency → smaller artefact.
6. **Mandatory shortcut controls**, outside the ranking: background-only and random-split.
   A class that is alone in a session is reported as confounded with it.

## Consequences

- 2026-11-08 result: the selector picked **HOG + logistic regression** (macro-F1 0.989,
  bootstrap 0.96–1.00). Frozen MobileNetV3/EfficientNet embeddings tied, and HOG + LR won on
  latency. `wittenberg_v1` reached 0.83; the protocol's small CNN 0.71. Status:
  **provisional**. Sufficient for 497/863/1218; insufficient for 2241 (one session, and the
  corners alone recognise it 65 % of the time).
- Shipping HOG + LR changes the app's model contract (`src/api.py` loads a `wittenberg_v1`
  checkpoint) and adds a scikit-learn pickle. That integration is a separate change. Until it
  lands, the app keeps `wittenberg_v1`. HOG is the classifier's feature extractor, not a
  preprocessing step: `src/preprocess.py` and its spec are unchanged (rule 1).
- Prospective confirmation is required before "confirmed": a new session with all classes
  interleaved, the frozen artefact evaluated once.
- Capture guidance does not change: ≥ 4 sessions, each covering every class, remains the
  target, because only that supports the session-level claim.
