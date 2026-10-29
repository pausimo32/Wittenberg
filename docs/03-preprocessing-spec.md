# Preprocessing specification — the contract

**Status:** normative. Reference study: Kumar & Wani (2024), *Modal Frequency Prediction of
Chladni Patterns Using Machine Learning*, Open Journal of Acoustics 12(1), 1–16.

There is **one** implementation: `src/preprocess.py`, using `cv2`. It is called in two places —
when building the training set, and by the local API at inference time (`docs/05-webapp-spec.md`
§3). Same function, same machine, same parameters. Train/serve skew is structurally impossible;
see `docs/adr/0004`.

This file still exists, and is still normative, for three reasons: `config/preprocess.json` must
remain the only place the parameters live; a reader of the TDR must be able to reproduce the
pipeline from a document rather than from source; and any change here invalidates every trained
model, which needs saying out loud.

## 1. Parameters — the only place they are defined

All parameters live in `config/preprocess.json`. Training and the API both read that file; no
numeric literals anywhere in the transform path.

```json
{
  "version": "1.1.0",
  "roi": {
    "mode": "homography",
    "output_size": 512,
    "interpolation": "INTER_LINEAR",
    "border_mode": "BORDER_REPLICATE"
  },
  "grayscale": { "code": "COLOR_RGBA2GRAY" },
  "clahe": { "enabled": true, "clip_limit": 2.0, "tile_grid_size": [8, 8] },
  "blur": { "ksize": [5, 5], "sigma_x": 1.4, "sigma_y": 1.4, "border": "BORDER_DEFAULT" },
  "canny": { "threshold1": 50, "threshold2": 150, "aperture_size": 3, "l2_gradient": false },
  "resize": { "size": [128, 128], "interpolation": "INTER_AREA" },
  "output": { "dtype": "float32", "scale": 0.00392156862745098, "shape": [128, 128, 1] }
}
```

`scale` is exactly `1/255`, written out in full so the stored value is exact.

## 2. Pipeline

Ordered. No step may be reordered, skipped or fused.

1. **Decode** → RGBA, 8-bit, sRGB. EXIF orientation applied then stripped; ICC profile
   discarded (force sRGB). See §5.1.
2. **ROI / geometric normalisation** → warp the plate to a `512×512` square.
   - `mode: "homography"` — the four plate corners are fixed by the rig (zenithal phone mount;
     camera, distance and px/mm in `docs/02-dataset-spec.md` §1). The homography is
     calibrated **once per session** from four ArUco markers taped at the plate corners and
     stored in `data/calib/<session_id>.json`.
   - Fallback `mode: "center_crop"` — largest centred square, for images with no calibration.
     Records `roi_mode` in metadata so mixed-provenance data is auditable.
   - `cv2.warpPerspective(..., flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)`.
     Both flags come from `config/preprocess.json` (`roi.interpolation`, `roi.border_mode`) as of
     spec 1.1.0 — they used to live only in this prose, which meant a reader could not rebuild
     the pipeline from the config alone.
   - **Destination quad is `[0, size-1]`, not `[0, size]`.** Pinned by
     `test_roi_of_a_frame_already_at_roi_size_is_an_identity_warp`. The other convention shifts
     everything by a sub-pixel and is silent.
   - `center_crop` goes through the *same* `warpPerspective` call as the homography path, so the
     two ROI modes cannot drift apart.
   - **Fewer than 4 markers raises `CalibrationError`** — never a silent fallback to
     `center_crop`. `roi_mode` has to stay auditable, and `docs/02-dataset-spec.md` §6 requires
     the image be rejected as `partial`. The caller decides, not the transform.
3. **Grayscale** — `cv2.cvtColor(src, cv2.COLOR_RGBA2GRAY)`.
   OpenCV uses ITU-R BT.601 luma: `0.299 R + 0.587 G + 0.114 B`, computed in fixed-point.
   This exact rounding is what the model was trained on; do not substitute a float formula.
4. **CLAHE** — `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)`.
   *Deviation from the reference study, deliberate.* Kumar & Wani used a fixed lab lamp; our
   captures happen in classrooms with varying light. CLAHE makes Canny's fixed thresholds
   behave consistently across sessions. Justify this in the TDR (§3.17). Ablate it in F4.
5. **Gaussian blur** — `cv2.GaussianBlur(src, (5,5), sigmaX=1.4, sigmaY=1.4)`.
   **`cv2.Canny` does no smoothing of its own** — despite the textbook description of the
   algorithm, OpenCV's implementation goes straight to a `Sobel` of size `apertureSize`. This
   explicit blur is therefore the *only* smoothing in the pipeline, not a redundant one. It sets
   the noise floor deterministically. Do not remove it as
   "already done by Canny" — it is not.
6. **Canny** — `cv2.Canny(src, 50, 150, apertureSize=3, L2gradient=False)` → binary `{0,255}`.
   The reference study reports that adding Canny after grayscale raised accuracy materially;
   we replicate it and ablate it in F4.
7. **Resize** — `cv2.resize(src, (128,128), interpolation=cv2.INTER_AREA)`.
   `INTER_AREA` on a binary image produces a **greyscale** anti-aliased result. That is
   intended — it preserves nodal-line density, which is the discriminative signal.
8. **Scale** — `astype(float32) * (1/255)` → shape `(128,128,1)`, range `[0,1]`.

Output tensor: `float32[1,128,128,1]`, NHWC.

### `preprocess_stages()` — the intermediates

`preprocess_stages(image_rgba, calib, config=None, collect_timings=False)` returns the array
after **each of steps 1–8**, as a `NamedTuple`: `source` `(H,W,3) uint8`, `roi` `(512,512,3)
uint8`, `gray` / `clahe` / `blur` / `edges` `(512,512) uint8`, `resized` `(128,128) uint8`,
`tensor` `(128,128,1) float32`.

Steps 1 and 2 are included **for the pipeline viewer** (`docs/05-webapp-spec.md` §8), which shows a
user what the machine did to their photo. They are also what makes a wrong prediction diagnosable
and what generates annex 7. Exposing an intermediate is not a transform change: it does not touch
the pipeline and does not invalidate a checkpoint.

**`collect_timings` — `timings_ms`.** With `collect_timings=True` the tuple's last field,
`timings_ms`, is `{step: milliseconds}` for steps 2–8, keyed by the field name of the array each
one produced (`roi`, `gray`, `clahe`, `blur`, `edges`, `resized`, `tensor`). It is `None`
otherwise, and off by default. Step 1 is not there — it belongs to `src/imageio.py` (§5.1) and is
timed by its own caller.

This exists for `docs/05` §8, which shows the *real measured* cost of each stage beside its image
rather than a remembered figure. The serving API must therefore never time the steps by re-running
them: §8 forbids running the pipeline twice, so the pipeline measures itself or nothing does.

**Timing is not a pipeline change** — see §6.0. The steps run in the same order, on the same
arrays, with the same parameters whichever way the flag is set; a stopwatch reads a clock between
them and touches no pixel. No version bump, no retrain, and the config hash is untouched, because
`config/preprocess.json` gains no key. `tests/test_preprocess.py` asserts the property directly:
the timed and untimed paths must produce byte-identical arrays.

**`timings_ms` is not an array.** Anything that compares the pipeline's output field by field —
determinism checks, the OpenCV baseline diff, annex tooling — must iterate `Stages.arrays`, not
the tuple. A wall clock in a byte-for-byte comparison fails on a machine that was merely busy,
which says nothing about the pixels.

**Non-mutation contract.** `source` is the caller's input array itself, not a defensive copy —
copying a full-resolution capture on every call does not pay for itself. `roi` is likewise the
array that fed step 3. **Callers must not write to either.** If you need to draw on one, copy it
first.

This is not hypothetical. The pipeline viewer will want to draw the detected ROI quad on top of
stage 1, and the ingest report will want to mark rejected regions. Both would silently corrupt
the array the caller passed in, and the corruption would surface as a wrong prediction much later.
Every overlay goes on `.copy()`.

## 3. Augmentation — training only, never at inference

Applied **after** step 8, only to the training split. Never to val/test. Never at inference.

| Transform | Range | Rationale |
|---|---|---|
| Rotation | ±4° | rig jitter only. A square plate's modes are **not** rotation-invariant — large rotations destroy the label. |
| Translation | ±3 % | mount play |
| Scale | ±3 % | camera distance drift within the 150–270 mm useful range |
| Brightness/contrast | ±8 % | applied **before** step 3 in the offline augmentation pass, not after Canny |
| Horizontal/vertical flip | **forbidden** | a square plate is symmetric, but the sand distribution is not; flips create labels the physical rig cannot produce |

Augmentation runs offline into `data/processed/` so the exact augmented set is reproducible
and can be shipped as a TDR annex.

## 4. Version consistency — the one guard

With a single implementation there is nothing to compare. What *can* still go wrong is serving a
checkpoint trained under a different version of this spec.

The guard has **two keys**, and both are hard failures.

1. Every checkpoint stores the fully resolved `config/preprocess.json` **and its SHA-256**, plus
   **`cv2.__version__` and the OpenCV package name**, in `runs/<ts>/config.json` — and all of
   them again inside `model.pt`, because the checkpoint travels with the repository
   (`docs/adr/0005`) and has to carry its own provenance.
2. The API asserts at startup that *both* match the running environment. On either mismatch it
   **refuses to start** and prints what differs.
3. `python tasks.py check-model` runs the same two assertions from the command line.

### Key one — the hash, over the resolved config

`json.dumps(config, sort_keys=True, separators=(",", ":"))`, in `src/runs.py`. Not over the
file's bytes: re-indenting the JSON, reordering two keys, or Git checking the file out with
CRLF on Windows would each change the bytes without changing a single pixel, and a guard that
fires on those is a guard everyone learns to ignore. Any changed *value* changes the hash.

### Key two — the OpenCV build

**The hash pins the parameters. It does not pin the implementation.** Reinstall the venv with a
different OpenCV and the hash still matches, while the pixels have moved: `docs/adr/0006`
measures `cv2.warpPerspective` differing by up to 7 grey levels in **every** frame between
4.12 and 5.0, all 120 final tensors differing, and up to 5 298 of 262 144 edge pixels flipping
state. A model trained on one of those edge maps and served against the other answers
confidently and wrongly, and key one sees nothing at all.

So `opencv_version` and `opencv_package` are compared exactly, and a mismatch is refused exactly
as hard as a hash mismatch. What is compared is `cv2.__version__`, not the wheel version: a bump
from `opencv-contrib-python` 4.12.0.88 to 4.12.0.90 leaves `cv2.__version__` at `4.12.0` and
changes no pixel, so it does not fire. The wheel version is recorded for the audit trail. The
package name is part of the key because the contrib and non-contrib builds are different
software and only one of them has `cv2.aruco` (§5.3).

A checkpoint that records neither key is refused too. Unverifiable is not the same as fine.

This is a hard failure, never a warning. It is the whole of what used to be a parity harness.

## 5. Traps that still apply

Most of the old divergence traps concerned two implementations and are gone. Three survive
because they are about the *input*, not about a second language.

**5.1 iPhone EXIF and HEIC.** An iPhone — the 11 and the 13 alike — writes EXIF orientation and a
Display P3 ICC profile.
Untreated, the same photo arrives rotated and colour-shifted. Handled **once**, in the ingest
layer: HEIC→PNG, orientation applied then stripped, P3→sRGB, profiles stripped. The API applies
the identical normalisation to uploads before calling `preprocess` — this is the one place where
"the same thing must happen in two code paths" still holds, so it lives in a single shared
helper, `src/imageio.py`, used by both.

Capture-station frames (`docs/adr/0008`) arrive differently: PNG grabbed from the Continuity
Camera stream by the browser, with no EXIF orientation and, expected but not yet verified, no
ICC profile. The same helper decodes them, so nothing in the pipeline changes. What the station
does change is the *source*: a video frame, already compressed by the stream before the PNG
encode. That is the §5.2 question asked of training data rather than of a user's upload, and
ADR 0008 says how to test it.

**5.2 JPEG is lossy and the camera is not.** Training images come from HEIC via a lossless PNG
conversion. If a user later feeds a re-compressed JPEG, quantisation artefacts survive Canny as
spurious edges. Test it (`tests/test_jpeg_robustness.py`) rather than assuming; if accuracy drops
materially, say so in the TDR limitations.

**5.3a Resolve the version pin BEFORE the first real ingest — not after.**

The pin in `config/versions.json` and the version actually installed must agree, and the moment
to make them agree is *now*, while no real dataset and no checkpoint exist.

Preprocessing output **is** the dataset. Ingest 4 000 real photographs under one OpenCV version
and later "fix" the pin to another, and every processed image changes: the edge maps differ, the
trained model no longer matches them, and the only honest remedy is to re-ingest and re-train.
Before F3b that costs nothing; after it, it costs the dataset.

**Measured, not assumed** — `docs/adr/0006` did this over 120 frames, 4.12.0 against 5.0.0 with
Python and numpy held fixed. The threat is **`cv2.warpPerspective` in step 2**: fed identical
input, CLAHE, the blur, Canny and the resize are byte-identical across the two versions, and the
warp differs by up to 7 grey levels in every single frame. CLAHE then *doubles* what the warp
hands it (median worst pixel 4 → 8, worst frame 12) without contributing anything of its own —
it is the amplifier, not the origin. By the end, all 120 tensors differ and up to 5 298 of
262 144 edge pixels flip state.

No gate verdict changed and `metadata.csv` came out byte-identical, which is the trap: the gate
has margin, the pixels do not. `pytest` passing proves the code runs, **not** that the pixels
match what an earlier version produced — the tests assert shapes, dtypes, ranges and
determinism, and all of those hold under either version.

Two acceptable resolutions, and only two:

1. **Align the pin to the environment** (usually right — do not fight the toolchain). Update
   `config/versions.json`, re-run the fixtures, and record in `docs/adr/` that the baseline moved
   and why. Bump `config/preprocess.json` if any fixture output changed.
2. **Align the environment to the pin.** Install the pinned versions in the venv.

What is *not* acceptable is leaving `check-versions` red and continuing. A red version check that
everyone has learned to ignore is worse than no check at all.

**5.3 OpenCV: the contrib build, pinned.** The package is **`opencv-contrib-python`**, not
`opencv-python`. `cv2.aruco` — which the per-session calibration in step 2 depends on — ships
only in the contrib build. Plain `opencv-python` imports fine and then raises
`AttributeError: module 'cv2' has no attribute 'aruco'` the first time a real session is
ingested, i.e. *after* the lab work. Never install both packages: they conflict, and which one
wins depends on install order. `opencv-python` is pinned in `config/versions.json` and asserted
by `python tasks.py check-versions`. Not for cross-language parity any more, but because **the
OpenCV version is part of the experimental method** (`docs/adr/0006`) — it belongs beside the
seed, the camera distance and the name of the GPU, and a reader reproducing the work in a year
needs it for the same reason they need `d = 210 mm`. What moves between versions is the
geometric warp of step 2, measured; a reader who remembers "something about CLAHE" is
remembering the amplifier rather than the cause.

## 6. Changing this spec

0. First decide whether the pipeline actually changed. Exposing a new intermediate, renaming a
   variable, adding a test or **measuring how long a step takes** does **not** change it —
   `collect_timings` (§2) was added under this clause, with no bump and no retrain. Moving a
   parameter out of prose and into
   `config/preprocess.json` does not change the computed pixels either, but it *does* change the
   config hash, so it needs a version bump and a retrain if any checkpoint already exists. Spec
   1.1.0 did exactly this, before F4 had produced any checkpoint — which is the cheap moment.
1. Bump `version` in `config/preprocess.json` (semver; any pipeline change is at least a minor).
2. Update this file in the **same commit**.
3. Retrain. A preprocessing change invalidates every existing checkpoint, and the startup guard
   in §4 will refuse to serve them — which is the intended behaviour, not an obstacle.
4. Add an ADR under `docs/adr/` if the change alters the pipeline shape, not just a constant.
