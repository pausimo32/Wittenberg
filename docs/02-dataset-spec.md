# Dataset specification

## 1. Capture rig

Fixed zenithal mount (documented in `Montura_cenital_iPhone11_Wittemberg.pdf`).

| Parameter | Value |
|---|---|
| Camera | iPhone 11, wide lens, f = 4.25 mm, sensor 5.645 × 4.234 mm |
| Plate | 200 × 200 mm, top face of the rig at z = 333 |
| Nominal distance | **210 mm** (frame 209 × 279 mm, ≈ 14.5 px/mm) |
| Useful range | 150–270 mm |
| Mount height | `H = d − 29.3 mm`; nominal `H = 180.7 mm` |
| Capture mode | manual exposure locked, AE/AF locked, HDR **off**, Live Photo **off** |

Lock focus and exposure before the first capture of a session and do not touch them again.
An auto-exposure change mid-session is the single biggest source of within-class variance.

## 2. Session model — why it matters more than anything else

A **session** = one continuous run: rig untouched, lighting unchanged, sand replenished the
same way, one calibration. `session_id = YYYYMMDD-NN`.

Within a session, consecutive frames of the same frequency are near-duplicates. If those
frames straddle the train/test boundary, the model memorises the session, not the physics,
and test accuracy is meaninglessly high.

**Therefore: splits are grouped by `session_id`, never random.**

```python
from sklearn.model_selection import GroupShuffleSplit  # groups = session_id
```

Minimum for a defensible result: **≥ 4 sessions**, and every frequency class must appear in
at least 3 of them. Target split 2 sessions train / 1 val / 1 test, rotated (leave-one-
session-out CV) so every session is the held-out one once.

Report **both** numbers in the TDR: the naive random-split accuracy and the session-wise
accuracy. The gap between them is a genuine finding — the reference study does not describe
its split, and its 5-test-images-per-class protocol cannot rule out this leak.

## 3. Frequency classes

Do **not** copy the reference study's 14 frequencies (330–3910 Hz). Those belong to a
16 × 16 cm, 0.61 mm aluminium plate. Ours is 200 × 200 mm — the modes are elsewhere.

Classes are determined empirically in phase F1:

1. Sweep 100 → 5000 Hz in 5 Hz steps, 3 s dwell.
2. Mark a frequency as a **mode** when the pattern is stationary, symmetric and reforms after
   the plate is re-seeded with sand.
3. Record the resonance width; the class label is the centre frequency.
4. Keep only modes whose neighbours are ≥ 40 Hz away, so classes are physically separable.

Expect 10–18 usable modes. Record all of them in `data/modes.csv`, keep the used subset in
`config/classes.json`.

## 4. Layout and naming

```
data/
├── raw/<session_id>/<freq_hz>/IMG_<n>.HEIC     # untouched, straight off the phone
├── calib/<session_id>.json                     # homography, 4 corner points, px/mm
├── processed/<split>/<freq_hz>/<session_id>_<n>.png
├── metadata.csv
└── modes.csv
```

`metadata.csv` — one row per raw image, the only file that is committed:

| Column | Type | Notes |
|---|---|---|
| `image_id` | str | `<session_id>_<freq_hz>_<n>` — primary key |
| `session_id` | str | `YYYYMMDD-NN` |
| `freq_hz` | int | the label |
| `split` | enum | `train` / `val` / `test` — assigned by session, never per-image |
| `material` | enum | `sand` / `semolina` / `salt` |
| `grain_mass_g` | float | |
| `distance_mm` | int | 210 unless deliberately varied |
| `lighting` | enum | `lamp_diffuse` / `lamp_grazing` / `ambient` |
| `dwell_s` | float | seconds between reaching the frequency and the shutter |
| `roi_mode` | enum | `homography` / `center_crop` |
| `quality` | enum | `ok` / `blurred` / `partial` / `reject` |
| `notes` | str | free text |

## 5. Volume targets

| | Reference study | Wittenberg target | Why |
|---|---|---|---|
| Images per class, **total** | 205 | **≥ 240** (≥ 60 per session × 4) | spread across 4 sessions |
| Images per class, train | 200 | ≈ 120 (2 of 4 sessions) | follows from the 2/1/1 session split |
| Images per class, test | 5 | **≥ 30** | see below |
| **Capture sessions** | not stated | **≥ 4** | the figure that actually sets the interval |
| Total | ~3 000 | ~4 000 | |

Read the test-set row carefully, because the naive reading is wrong. Going from 5 frames to 30
frames does **not** narrow the confidence interval, because frames from one burst are one
observation (below). What ≥ 30 buys is a usable per-class recall estimate and enough variety to
see *which* patterns fail. What narrows the interval is **more sessions** — that is why the
session count is in this table.

For scale: 4 correct out of 5 gives a 95 % Wilson interval of roughly 38–96 %. The reference
study's 5-image test set therefore cannot distinguish a good model from a mediocre one,
whichever way its frames were sampled.

A burst of 60 frames at 1 fps with the rig untouched is **not** 60 independent samples.
Treat a burst as one observation for confidence-interval purposes and say so in the TDR.

## 6. Quality gate at ingest

`make ingest` rejects an image and writes the reason to `data/ingest_report.json` when:

- Laplacian variance < 100 → `blurred`
- fewer than 4 ArUco markers found and `roi_mode` is `homography` → `partial`
- mean luminance outside `[40, 215]` → `reject` (clipped highlights kill Canny)
- fewer than 0.5 % or more than 25 % edge pixels after Canny → `reject` (no pattern / noise)

Rejected images stay in `raw/` and are counted in the TDR as an honest yield figure.
