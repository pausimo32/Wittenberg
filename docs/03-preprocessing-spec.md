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
   - `mode: "homography"` — the four plate corners are fixed by the rig (zenithal iPhone 11
     mount, d = 210 mm, ≈14.5 px/mm — see `docs/02-dataset-spec.md`). The homography is
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

`preprocess_stages(image_rgba, calib, config=None)` returns the array after **each of steps 1–8**,
as a `NamedTuple`: `source` `(H,W,3) uint8`, `roi` `(512,512,3) uint8`, `gray` / `clahe` / `blur` /
`edges` `(512,512) uint8`, `resized` `(128,128) uint8`, `tensor` `(128,128,1) float32`.

Steps 1 and 2 are included **for the pipeline viewer** (`docs/05-webapp-spec.md` §8), which shows a
user what the machine did to their photo. They are also what makes a wrong prediction diagnosable
and what generates annex 7. Exposing an intermediate is not a transform change: it does not touch
the pipeline and does not invalidate a checkpoint.

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

1. Every checkpoint stores the fully resolved `config/preprocess.json` **and its SHA-256** in
   `runs/<ts>/config.json`.
2. The API asserts at startup that the hash of the running `config/preprocess.json` equals the
   checkpoint's. On mismatch it **refuses to start** and prints both versions.
3. `python tasks.py check-model` runs the same assertion from the command line.

This is a hard failure, never a warning. It is the whole of what used to be a parity harness.

## 5. Traps that still apply

Most of the old divergence traps concerned two implementations and are gone. Three survive
because they are about the *input*, not about a second language.

**5.1 iPhone EXIF and HEIC.** iPhone 11 writes EXIF orientation and a Display P3 ICC profile.
Untreated, the same photo arrives rotated and colour-shifted. Handled **once**, in the ingest
layer: HEIC→PNG, orientation applied then stripped, P3→sRGB, profiles stripped. The API applies
the identical normalisation to uploads before calling `preprocess` — this is the one place where
"the same thing must happen in two code paths" still holds, so it lives in a single shared
helper, `src/imageio.py`, used by both.

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

The same applies to the Python version. A major-version gap (4.x → 5.x) can change CLAHE tile
interpolation and Canny's internals; `pytest` passing proves the code runs, **not** that the
pixels match what an earlier version produced.

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
by `python tasks.py check-versions`. Not for cross-language parity any more, but for
reproducibility across the three team members' machines and across time: CLAHE tile
interpolation has changed between minor versions before, and a reader reproducing the work in a
year needs the version.

## 6. Changing this spec

0. First decide whether the pipeline actually changed. Exposing a new intermediate, renaming a
   variable or adding a test does **not** change it. Moving a parameter out of prose and into
   `config/preprocess.json` does not change the computed pixels either, but it *does* change the
   config hash, so it needs a version bump and a retrain if any checkpoint already exists. Spec
   1.1.0 did exactly this, before F4 had produced any checkpoint — which is the cheap moment.
1. Bump `version` in `config/preprocess.json` (semver; any pipeline change is at least a minor).
2. Update this file in the **same commit**.
3. Retrain. A preprocessing change invalidates every existing checkpoint, and the startup guard
   in §4 will refuse to serve them — which is the intended behaviour, not an obstacle.
4. Add an ADR under `docs/adr/` if the change alters the pipeline shape, not just a constant.
