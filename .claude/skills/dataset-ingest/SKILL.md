---
name: dataset-ingest
description: Ingest a new Chladni capture session — HEIC conversion, calibration, quality gating, metadata rows, split assignment. Use after every lab session, or when adding, re-labelling or re-splitting images.
allowed-tools: Read Bash Grep Glob Edit
---

# Ingest a capture session

Contract: `docs/02-dataset-spec.md`. Run this after every lab session, before any training.

## Before the session even starts

Run `python tools/check_capture.py <first photo>` on the very first frame. It reports the
**interior** edge density (the ROI border is excluded — see `docs/02-dataset-spec.md` §6) and
exits 1 with actionable advice if Canny is not seeing the pattern. One second there saves a day.

## Preconditions — check before running anything

- `data/raw/<session_id>/` exists, `session_id` matches `YYYYMMDD-NN`
- `data/calib/<session_id>.json` exists (4 ArUco corners + px/mm), or the session is
  explicitly `center_crop`
- The session log records: material, grain mass, distance, lighting, dwell
- Subfolders are named by integer Hz and all appear in `config/classes.json`

Sessions whose id starts with `SYNTH-` are synthetic (`tools/synth_capture.py`). Ingest them to
test the plumbing, never to train a reported model, and never in the same dataset as real ones.

A session missing calibration is not blocked — it is ingested with `roi_mode: center_crop` and
flagged. A session missing `session_id` is blocked; those images cannot be used at all.

## Run

```bash
python tasks.py ingest --session 20260916-01
```

Which does, in order:

1. **HEIC → PNG**, EXIF orientation applied then stripped, Display P3 → sRGB, profile stripped.
   Once, here — never later (spec §5.1).
2. **Calibrate** — homography from the 4 markers, warp to 512×512.
3. **Quality gate** (`docs/02-dataset-spec.md` §6), **in this order**: markers → blur →
   luminance → interior edge density. Markers are checked on the raw frame; everything else needs
   the 512 ROI, which does not exist until the markers are found.
   Thresholds: Laplacian variance ≥ 100; markers 0-3 all found; mean luminance ∈ [40, 215];
   **interior** Canny edge density ∈ [0.5 %, 25 %] with a 2 % margin excluded on every side.
   Never on the full frame — the ROI border alone can carry the whole gate.
4. **Preprocess** through `src/preprocess.py` → `data/processed/unassigned/<freq_hz>/`.
   The split is not known yet; `tasks.py split` moves files into the §4 layout afterwards.
5. **Append** rows to `data/metadata.csv` — **one row per raw image, accepted or not**, with
   `quality` in {ok, blurred, partial, reject} and `split` empty for anything not `ok`. Also
   write `data/ingest_report.json`. Writing only the accepted rows makes `quality` a column
   that is always `ok` and moves the yield figure into a file that is not committed.

Idempotent. Re-running a session replaces its rows; it never duplicates them.

## Assign the split — the step that is easy to get wrong

Splits are per **session**, never per image (`docs/adr/0002`).

```bash
python tasks.py split   # GroupShuffleSplit on session_id, seed 42
```

Never hand-edit `split` in `metadata.csv`. Never move a single image between splits. If a
session belongs in test, the whole session goes to test.

## After ingest — always

Run the `wittenberg-dataset-curator` subagent. Do not start training on a session that has not
been audited: the failure it catches (a condition confound, a class present in only one session)
costs a full training cycle and can silently invalidate a reported result.

## Reporting yield

`data/ingest_report.json` rejection rates go into the TDR as an honest yield figure, broken
down by reason and condition. A high reject rate under one lighting condition is a *result*
(research question S4), not an embarrassment.
