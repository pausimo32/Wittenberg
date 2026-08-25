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

### 1.1 ArUco markers — the calibration convention

Four `DICT_4X4_50` markers, **ids 0, 1, 2, 3**, assigned clockwise from the plate corner nearest
the camera's top-left: `0` top-left, `1` top-right, `2` bottom-right, `3` bottom-left. Print them
with a white quiet-zone border of at least one fifth of the marker side, or detection gets
unreliable at grazing light.

**Tape each marker just outside the plate so that its corner nearest the plate centre touches the
plate corner.** That touching corner — not the marker centre — is the calibration reference
point. Using centres instead puts the marker bodies and the dark surround inside the ROI, and the
warp is then of the wrong quadrilateral.

In `detectMarkers` order (TL, TR, BR, BL of each marker), the touching corner is the diagonally
opposite one:

| marker id | plate corner | index into that marker's 4 corners |
|---|---|---|
| 0 | top-left | 2 (its BR) |
| 1 | top-right | 3 (its BL) |
| 2 | bottom-right | 0 (its TL) |
| 3 | bottom-left | 1 (its TR) |

`data/calib/<session_id>.json` stores those four points, in id order, plus the derived px/mm.
Photograph the calibration frame once per session, before any pattern.

**The marker id says which corner it is, and ingest orders the quad by id, never by geometry.**
Ordering by position instead would silently mis-order a frame shot at an angle, and a mis-ordered
quad warps to a mirrored or rotated plate that still looks entirely plausible.

The two plausible alternatives to the touching corner are both wrong by a knowable amount: at the
nominal 210 mm working distance (§1, ≈ 14.5 px/mm) a 25 mm marker is ≈ 360 px, so taking the
marker **centre** drags ≈ 12 mm of rig into the ROI on every side, and taking the **outer** corner
drags ≈ 25 mm. Either shrinks the plate within the warped image and displaces every nodal line the
model is asked to read. The rule holds under perspective: the warp is projective, and a projective
map preserves which corner of a marker is nearest the interior.

The dictionary and the id map are a **rig convention, not a physical constant**, so they live in
`config/calibration.json` and are copied into every `data/calib/<session_id>.json` alongside the
four points. This section is the definition; the config refers to it and must not restate it.

Fewer than four markers in a frame is a `partial` rejection under §6, never a silent fall-back to
`center_crop`.

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
| `split` | enum | `train` / `val` / `test` — assigned by session, never per-image. **Empty** when `quality` is not `ok`: a rejected frame belongs to no partition (§6) |
| `material` | enum | `sand` / `semolina` / `salt` |
| `grain_mass_g` | float | |
| `distance_mm` | int | 210 unless deliberately varied |
| `lighting` | enum | `lamp_diffuse` / `lamp_grazing` / `ambient` |
| `dwell_s` | float | seconds between reaching the frequency and the shutter |
| `roi_mode` | enum | `homography` / `center_crop` |
| `quality` | enum | `ok` / `blurred` / `partial` / `reject` — the §6 verdict. Only `ok` rows are training data |
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

**Order matters, and it is not the order these bullets used to be written in.** Markers are
checked on the raw frame; everything else is measured on the 512 ROI, which does not exist until
the markers are found. So:

1. markers → 2. blur → 3. luminance → 4. interior edge density

`python tasks.py ingest` rejects an image and writes the reason to `data/ingest_report.json`
**and to `metadata.csv`** when:

- fewer than 4 ArUco markers found and `roi_mode` is `homography` → `partial`
- Laplacian variance < 100 → `blurred`
- mean luminance outside `[40, 215]` → `reject` (clipped highlights kill Canny)
- **interior** edge density outside `[0.5 %, 25 %]` after Canny → `reject` (`no_pattern` / `noise`)

**The reported reason is the first that fails.** A frame failing several rules is counted once,
under the first — otherwise the same frame inflates several rejection categories and the yield
figure stops adding up.

**Where each metric is measured.** The marker count comes from the raw frame, which is where the
markers are. Blur, luminance and edge density are measured on the warped 512 × 512 ROI, so a
session shot at a different distance is not judged on a different number of pixels.

The thresholds live in `config/quality_gate.json`, not in code, so a change to the gate is a
reviewable diff; ingest copies them verbatim into every `data/ingest_report.json`, because a yield
figure is meaningless without the gate that produced it.

### The interior margin — do not skip this

Edge density is measured on the ROI **excluding a 2 % margin on every side** (10 px of the 512).
Measuring the whole frame is a trap, and it is not hypothetical: on a synthetic capture whose
pattern Canny did not detect at all, full-frame density was 0.78 % — comfortably inside the gate
— and **100 % of those edge pixels were the frame border**. `BORDER_REPLICATE` plus the
plate/surround transition draws a rectangle of edges at the ROI boundary of every single image.

A gate that counts them passes images containing no pattern at all, and those images then poison
a class. Measure the interior, or the gate is measuring the frame.

This lives in the ingest layer, not in `src/preprocess.py`: it changes no transform, needs no
version bump, and invalidates no checkpoint.

### Validate the thresholds on the FIRST photo, not on day four

Canny's fixed 50/150 thresholds assume a certain sand-to-plate contrast. On a low-contrast
capture they can return an essentially empty interior for a whole class. Run
`python tools/check_capture.py <first photo>` at the start of every session, before capturing
anything else: it reports interior edge density and writes a side-by-side PNG. If the interior
density is near zero, fix the lighting or the material **then**.

### Rejected images get a row too

`metadata.csv` has **one row per raw image**, accepted or not — that is what §4 says and it is
right. The `quality` column carries `ok` / `blurred` / `partial` / `reject`, and `split` is empty
for anything that is not `ok`.

Writing only the accepted rows is tempting and wrong: it makes `quality` a column that is always
`ok`, and it moves the yield figure into `data/ingest_report.json`, which is not committed. The
TDR reports yield (`docs/06-research-protocol.md`), and every reported number must be traceable
to a committed file. `.gitignore` therefore needs `!data/metadata.csv`.

Rejected images stay in `raw/` and are counted in the TDR as an honest yield figure.

### Between ingest and split

Ingest does not know the split — `tasks.py split` assigns it afterwards. Processed images land in
`data/processed/unassigned/<freq_hz>/` and `split` moves them into the `§4` layout. The layout in
§4 describes the state *after* splitting.

### The 40 Hz rule is symmetric

When two modes are closer than 40 Hz, **both** are rejected, not just the higher one. Keeping one
of an unresolvable pair is an arbitrary choice that the data cannot justify.
