# Dataset specification

## 1. Capture rig

Fixed zenithal mount (documented in `Montura_cenital_iPhone11_Wittemberg.pdf`, which is not in
this repository).

**The camera is an iPhone 13, not the iPhone 11 this mount was designed for** (`docs/adr/0008`).
Every row marked *iPhone 11* below was computed for the old phone and is kept only until the
feasibility probe of ADR 0008 replaces it with a measurement on the iPhone 13. Check first that
the iPhone 13 fits the mount at all: its body is a different size and its lens sits elsewhere,
so the mount may not put the optical axis over the plate centre.

| Parameter | Value |
|---|---|
| Camera | **iPhone 13**, wide camera, f = 5.1 mm, sensor ≈ 6.85 × 5.14 mm (*published spec*, 12 MP at 1.7 µm). The same physical phone in every session — §1.3 |
| Plate | **180 × 180 mm, 1 mm black-lacquered steel**, centre-driven, free edges. Top face of the rig at z = 333 (*measured on the old rig*). `plate_size_mm` in `config/calibration.json` |
| Nominal distance | *iPhone 11:* **210 mm** (frame 209 × 279 mm, ≈ 14.5 px/mm). iPhone 13 at 210 mm: frame ≈ 212 × 282 mm (*computed*) — but see §1.2, the markers do not fit |
| Useful range | *iPhone 11:* 150–270 mm |
| Mount height | *iPhone 11 mount:* `H = d − 29.3 mm`; nominal `H = 180.7 mm` |
| Capture mode, photos by hand | manual exposure locked, AE/AF locked, HDR **off**, Live Photo **off** |
| Capture mode, capture station | Continuity Camera on the MacBook; **Center Stage, Portrait, Studio Light and Reactions off**; exposure and focus locked if the browser can lock them (ADR 0008 — unknown until probed) |

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

**The rule is geometric: of a marker's four corners, take the one nearest the centroid of the
four markers.** Not an index. Ingest selects that way, and the two facts that make it the right
rule are that the plate lies *between* the markers and that a projective warp preserves which
corner is nearest the interior — so it survives perspective, and it survives a marker taped on
rotated.

On the **nominal zenithal rig**, with every marker taped square and the plate upright in the
frame, that rule evaluates to a fixed index into each marker's `detectMarkers` quad (TL, TR, BR,
BL) — the diagonally opposite corner:

| marker id | plate corner | what the rule evaluates to on the nominal rig |
|---|---|---|
| 0 | top-left | 2 (its BR) |
| 1 | top-right | 3 (its BL) |
| 2 | bottom-right | 0 (its TL) |
| 3 | bottom-left | 1 (its TR) |

**This table is an assertion about the mounting, not the definition of the calibration point.**
Ingest computes the corner geometrically and then compares the answer with this table
(`nominal_corner_index` in `config/calibration.json`). A disagreement means a marker is taped on
rotated, or the plate is turned within the frame. Both are legitimate — the calibration is
correct either way, because the geometry chose the point — so the disagreement is written to the
session's flags and to `data/ingest_report.json` as a **warning, never an error**. Only the
operator can tell a deliberately turned plate from a mis-taped marker, and the value of the
warning is entirely in arriving on the day of the session rather than three weeks later.

Selecting *by* the table would be the worse design: it is correct only while the mounting is
nominal, it fails silently when it is not, and it throws away the very signal that says so.

`data/calib/<session_id>.json` stores those four points, in id order, plus the derived px/mm,
the rule used and any disagreement with this table. Photograph the calibration frame once per
session, before any pattern.

**The marker id says which corner it is, and ingest orders the quad by id, never by geometry.**
Ordering by position instead would silently mis-order a frame shot at an angle, and a mis-ordered
quad warps to a mirrored or rotated plate that still looks entirely plausible.

The two plausible alternatives to the touching corner are both wrong by a knowable amount: at the
nominal 210 mm working distance (§1, ≈ 14.5 px/mm) a 25 mm marker is ≈ 360 px, so taking the
marker **centre** drags ≈ 12 mm of rig into the ROI on every side, and taking the **outer** corner
drags ≈ 25 mm. Either shrinks the plate within the warped image and displaces every nodal line the
model is asked to read.

The dictionary and the id map are a **rig convention, not a physical constant**, so they live in
`config/calibration.json` and are copied into every `data/calib/<session_id>.json` alongside the
four points. This section is the definition; the config refers to it and must not restate it.

Fewer than four markers in a frame is a `partial` rejection under §6, never a silent fall-back to
`center_crop`.

### 1.2 Framing — plate *and* markers must fit, which fixes the distance

The markers sit **outside** the plate (§1.1), diagonally off each corner, so the frame must
cover, on its *short* side:

```
180 mm  +  2 × 1.2 × s  +  2 × m        s = printed marker side, 1.2 = the §1.1 quiet zone,
                                        m = a safety margin per side
```

With `s = 25 mm` and `m = 10 mm` that is **260 mm**. The short side of the frame at distance `d`
is `d × (short sensor side / f)`:

| Source | Short side per mm of `d` | At 210 mm | `d` needed for 260 mm |
|---|---|---|---|
| iPhone 11 photo, 4:3 | 0.996 | 209 mm | ≈ 261 mm |
| iPhone 13 photo, 4:3 | 1.008 | 212 mm | ≈ 258 mm |
| iPhone 13 stream, 16:9, full sensor width | 0.756 | 159 mm | ≈ 344 mm |

*All computed from published sensor sizes, none measured.*

**This is not new with the iPhone 13: the nominal 210 mm never fitted the markers.** With the
original 200 mm plate, a 209 mm short side left 4.5 mm per side, less than any usable marker.
With the 180 mm plate it leaves 14.5 mm per side — room for a marker of at most 12 mm with its
quiet zone and no margin at all, which is not a design either. No real session has been captured
yet, so nothing was lost. The synthetic sessions do not show the problem because
`tools/synth_capture.py` draws the scene rather than photographing it.

Two things follow:

- **The working distance is set by the framing, then measured**, not taken from this table.
  Fix `s` when the markers are printed, read the stream's real aspect ratio, place the phone
  where plate plus markers fit with margin, and let the calibration report the px/mm.
  `data/calib/<session_id>.json` records it per session either way.
- **Resolution is not what limits the distance.** The 512 px ROI needs ≥ 2.84 px/mm over the
  180 mm plate. A 1920 px stream across the long side gives that up to ≈ 500 mm (*computed*),
  well beyond the 344 mm the framing asks for.

### 1.3 One camera for every session

The camera is a capture condition, exactly like lighting and material
(`docs/00-master-plan.md` F3b): a phone used in only one session predicts the session-wise
split perfectly, and the test accuracy would measure "a different phone" instead of "a
different pattern". **Use the same iPhone 13 in every session**, and record how each frame was
taken in the session log's `camera` key (§4).

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
16 × 16 cm, 0.61 mm aluminium plate. Ours is 180 × 180 mm, 1 mm black-lacquered steel — the modes
are elsewhere. It is closer in size to the reference plate than the 200 mm one it replaced, which
makes the temptation to reuse their list stronger, not the list more applicable: thickness and
material move every mode as much as the side length does.

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
├── raw/<session_id>/session.json               # the session log — REQUIRED, see below
├── raw/<session_id>/<freq_hz>/IMG_<n>.HEIC     # untouched, straight off the phone
│                                               #   (.png for a capture-station frame, ADR 0008)
├── calib/<session_id>.json                     # homography, 4 corner points, px/mm
├── processed/<split>/<freq_hz>/<session_id>_<n>.png
├── metadata.csv
└── modes.csv
```

Two directories under `data/raw/` are not sessions and never become part of one:

- `drafts/<session_id>/` (at the **repo root**, not under `data/`) is this same tree one step
  earlier — what `/dades` holds while a session is being labelled. It becomes
  `data/raw/<session_id>/` by being moved there, which is why it mirrors this layout exactly.
  Ignored by git, like `data/raw/` itself. See `docs/05` §9.
- `data/raw/.incoming-<session_id>/` is a session mid-commit: its captures have left the draft
  but the ingest has not finished, so it does not yet answer to the session's name. If you find
  one, the commit was interrupted — re-run it from `/dades`, or rename the directory and run
  `python tasks.py ingest <session_id>`.

`session_id` is `YYYYMMDD-NN` for a capture and `SYNTH-YYYYMMDD-NN` for a session generated by
`tools/synth_capture.py`. A synthetic id is a real id **with a prefix**, deliberately: if it had
a shape of its own, the F2.5 dry run would never once put a real id through the parsing that
reads it, which is most of what the dry run is for.

### `session.json` — written before the first shot, or the session is refused

`python tasks.py ingest` **refuses** a session without a complete `session.json`. It does not
warn. Five of these keys become columns of `metadata.csv` and none of them can be reconstructed
from the images: four days after the lab, nobody knows what the grain mass was.

| Key | Notes |
|---|---|
| `session_id` | must equal the directory name |
| `date` | `YYYY-MM-DD` or `YYYYMMDD`; must match the date in `session_id` (§2: one session, one day) |
| `operator` | who shot it |
| `plate` | which plate — the modes belong to a plate, not to the project |
| `material`, `grain_mass_g`, `distance_mm`, `lighting`, `dwell_s` | the §4 columns |
| `notes` | required to be present; **may be empty** — empty means "nothing anomalous", which is itself information |
| `synthetic` | `true`, on `SYNTH-*` sessions only. What keeps a rendered frame from later reading as a measurement |
| `camera` | *optional.* How the frames were taken: `iphone13-continuity-png` (capture station, ADR 0008) or `iphone13-heic` (photos by hand). Optional because ingest refuses only on missing *required* keys; it reaches the TDR through `data/ingest_report.json`, which records the whole log |

Ten seconds before the first capture against a whole session afterwards is not a close call.
A warning would be read as "carry on", the session would be ingested without the fields, and by
the time anyone noticed the rig would be dismantled.

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

The capture station's round-robin order (ADR 0008) does not change this. Its 60 frames of a
class are patterns that re-formed between frames rather than consecutive near-duplicates, which
buys variety, but they share the session's rig, sand and light. The session is still the unit.

### What ingesting this volume costs, measured before it exists

Timed on the F3.5 dry run — 4 synthetic sessions at `--frames 60`, 1 440 frames, this laptop,
one process, native Windows:

| | |
|---|---|
| ingest, 600 px synthetic PNG | **18.9 ms/frame** — 1 440 frames in 27 s |
| `tasks.py split` on the result | 0.7 s |
| pipeline alone, at iPhone 11 resolution (4032 × 3024) | 29 ms/frame — the ROI warp is destination-bound, so it barely notices the source size; marker detection is what scales |
| **HEIC decode, 12 MP** | **529 ms/frame** — PNG at the same size is 181 ms, JPEG 138 ms |

So the real ingest is **decode-bound, roughly 20 to 1**. At ≈ 0.56 s/frame, the full ≈ 4 000
frame dataset is **about 35–40 minutes** in one process — not ten minutes, and nowhere near
three hours.

That ratio is why `tasks.py ingest` puts frames through a process pool: the parallelism is over
frames, so it is the decode that scales and not the pipeline, and ~4 000 real frames come down
to **under ten minutes**. It is chosen from the *measured* cost of the session's first frame,
not from the frame count — the same pool is a pessimisation on the 600 px synthetic frames of
the dry run, which cost 19 ms each. `--jobs N` overrides it; `--jobs 1` is what to use when
something looks wrong.

**It changes the clock and nothing else.** `metadata.csv`, `ingest_report.json` and every
processed PNG come out byte-identical however many workers ran, and `tests/test_ingest.py`
asserts exactly that. If it ever stopped being true, the dataset would depend on the core count
of whichever laptop ingested it.

Worth knowing before the lab, not after: an ingest that took three hours would have to be
started overnight, and the point of `tools/check_capture.py` is that you find out on the day.

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

### The thresholds live in `config/quality_gate.json`

Not in code. Two reasons, and the second is the one that matters.

1. A change to the gate is then a reviewable diff on a four-line file, rather than a number
   moving inside a function nobody diffs closely.
2. **Ingest copies the file verbatim into every `data/ingest_report.json`.** The TDR reports the
   acceptance yield (`docs/06-research-protocol.md`, research question S4), and a yield figure
   without the thresholds that produced it is not a reproducible number — 27.5 % rejected means
   nothing unless the reader can see that `laplacian_variance_min` was 100. Reading them out of
   the source at whatever commit happens to be checked out is not the same thing.

The report also records the whole `session.json`, including `operator` and `plate`, which are
not `metadata.csv` columns and would otherwise exist only in an ignored directory.

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
