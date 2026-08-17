---
name: preprocessing-parity
description: Run and interpret the Python↔browser preprocessing parity gate for Wittenberg. Use when npm run parity fails, when changing any image transform, when bumping OpenCV, or before exporting a model. Bisects by pipeline stage and identifies the divergent step.
allowed-tools: Read Bash Grep Glob
---

# Preprocessing parity

The gate that proves the browser preprocesses images exactly as training did. Contract:
`docs/03-preprocessing-spec.md`. Read §4 (thresholds) and §5 (traps) before diagnosing anything.

## Run

```bash
npm run parity                  # full gate — this is the merge blocker
npm run parity -- --stage 6     # stop after Canny
npm run parity -- --dump /tmp/parity   # write both sides' arrays per stage
```

## Diagnosing a failure — in this order

**1. Version pin.** `python -c "import cv2; print(cv2.__version__)"` against the `opencv.js`
build in `web/public/opencv/`. Minor versions must match. This is the most common cause after
a dependency update and the cheapest to check.

**2. Bisect by stage.** Never analyse the final tensor. Dump per-stage and find the *first*
stage over tolerance. **The thresholds live in `docs/03-preprocessing-spec.md` §4 — read them
there.** They are not repeated here on purpose: a second copy would drift from the normative one.

**3. Match the symptom to a known trap** (spec §5):

| Symptom | Trap |
|---|---|
| Uniform small offset from step 3 | canvas luma used instead of `cv.cvtColor` (§5.1) |
| Divergence appears only at step 7, edges "softer" | `drawImage`/CSS resize instead of `cv.resize` + `INTER_AREA` (§5.2) |
| ±1–2 levels, random, only on JPEG fixtures | JPEG decoder difference (§5.3) — golden set must be PNG |
| 90° rotation, or a global colour cast | EXIF orientation / Display P3 not handled (§5.4) |
| Appeared right after `npm update` or `pip install -U` | version drift (§5.5) |
| Tiny diff only in the final tensor | scale fused into resize instead of applied last (§5.6) |

**4. Decide which side is wrong** by reading the spec, not by preferring the side that is
easier to change.

## Never

- Raise a threshold to make the gate pass.
- Change both sides in the same commit "to make them agree".
- Ship a model whose parity gate is red.

## After a fix

1. `npm run parity` green.
2. Bump `version` in `config/preprocess.json` if the pipeline changed, not just a bug in one side.
3. Retrain if the Python side changed — `runs/` before the fix are invalid.
4. Note it in `docs/03-preprocessing-spec.md`; add an ADR if the pipeline shape changed.
