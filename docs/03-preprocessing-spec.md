# Preprocessing specification — the contract

**Status:** normative. Both implementations must satisfy this file byte-for-byte within the
stated tolerances. Reference study: Kumar & Wani (2024), *Modal Frequency Prediction of Chladni
Patterns Using Machine Learning*, Open Journal of Acoustics 12(1), 1–16.

Two implementations exist and must agree:

| Side | File | Library |
|---|---|---|
| Training | `src/preprocess.py` | `cv2` (opencv-python) |
| Inference | `web/lib/preprocess.ts` | `opencv.js` (OpenCV compiled to WASM) |

They use the **same OpenCV C++ core**, which is why they can agree. Do not reimplement any
step with Canvas 2D, `drawImage`, or hand-written loops — those diverge (see §5).

## 1. Parameters — the only place they are defined

All parameters live in `config/preprocess.json`. Both sides read that file; no literals in code.

```json
{
  "version": "1.0.0",
  "roi": { "mode": "homography", "output_size": 512 },
  "grayscale": { "code": "COLOR_RGBA2GRAY" },
  "clahe": { "enabled": true, "clip_limit": 2.0, "tile_grid_size": [8, 8] },
  "blur": { "ksize": [5, 5], "sigma_x": 1.4, "sigma_y": 1.4, "border": "BORDER_DEFAULT" },
  "canny": { "threshold1": 50, "threshold2": 150, "aperture_size": 3, "l2_gradient": false },
  "resize": { "size": [128, 128], "interpolation": "INTER_AREA" },
  "output": { "dtype": "float32", "scale": 0.00392156862745098, "shape": [128, 128, 1] }
}
```

`scale` is exactly `1/255`. Written out to avoid a float literal mismatch between languages.

## 2. Pipeline

Ordered. No step may be reordered, skipped or fused.

1. **Decode** → RGBA, 8-bit, sRGB. EXIF orientation applied then stripped; ICC profile
   discarded (force sRGB). See §5.4.
2. **ROI / geometric normalisation** → warp the plate to a `512×512` square.
   - `mode: "homography"` — the four plate corners are fixed by the rig (zenithal iPhone 11
     mount, d = 210 mm, ≈14.5 px/mm — see `docs/02-dataset-spec.md`). The homography is
     calibrated **once per session** from four ArUco markers taped at the plate corners and
     stored in `data/calib/<session_id>.json`.
   - Fallback `mode: "center_crop"` — largest centred square, for images with no calibration.
     Records `roi_mode` in metadata so mixed-provenance data is auditable.
   - `cv2.warpPerspective(..., flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)`
3. **Grayscale** — `cv2.cvtColor(src, cv2.COLOR_RGBA2GRAY)`.
   OpenCV uses ITU-R BT.601 luma: `0.299 R + 0.587 G + 0.114 B`, computed in fixed-point.
   **This is why we do not use canvas luma** — the rounding differs (§5.1).
4. **CLAHE** — `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)`.
   *Deviation from the reference study, deliberate.* Kumar & Wani used a fixed lab lamp; our
   captures happen in classrooms with varying light. CLAHE makes Canny's fixed thresholds
   behave consistently across sessions. Justify this in the TDR (§3.17). Ablate it in F4.
5. **Gaussian blur** — `cv2.GaussianBlur(src, (5,5), sigmaX=1.4, sigmaY=1.4)`.
   **`cv2.Canny` does no smoothing of its own** — despite the textbook description of the
   algorithm, OpenCV's implementation goes straight to a `Sobel` of size `apertureSize`. This
   explicit blur is therefore the *only* smoothing in the pipeline, not a redundant one. It sets
   the noise floor deterministically and identically on both sides. Do not remove it as
   "already done by Canny" — it is not.
6. **Canny** — `cv2.Canny(src, 50, 150, apertureSize=3, L2gradient=False)` → binary `{0,255}`.
   The reference study reports that adding Canny after grayscale raised accuracy materially;
   we replicate it and ablate it in F4.
7. **Resize** — `cv2.resize(src, (128,128), interpolation=cv2.INTER_AREA)`.
   `INTER_AREA` on a binary image produces a **greyscale** anti-aliased result. That is
   intended — it preserves nodal-line density, which is the discriminative signal.
8. **Scale** — `astype(float32) * (1/255)` → shape `(128,128,1)`, range `[0,1]`.

Output tensor: `float32[1,128,128,1]`, NHWC.

## 3. Augmentation — training only, never at inference

Applied **after** step 8, only to the training split. Never to val/test. Never in the browser.

| Transform | Range | Rationale |
|---|---|---|
| Rotation | ±4° | rig jitter only. A square plate's modes are **not** rotation-invariant — large rotations destroy the label. |
| Translation | ±3 % | mount play |
| Scale | ±3 % | camera distance drift within the 150–270 mm useful range |
| Brightness/contrast | ±8 % | applied **before** step 3 in the offline augmentation pass, not after Canny |
| Horizontal/vertical flip | **forbidden** | a square plate is symmetric, but the sand distribution is not; flips create labels the physical rig cannot produce |

Augmentation runs offline into `data/processed/` so the exact augmented set is reproducible
and can be shipped as a TDR annex.

## 4. Parity gate

`npm run parity` is a **merge blocker**. It runs `scripts/parity_check.py` + a headless
Playwright page and compares.

Golden set: `tests/golden/*.png` — 20 images, PNG only (§5.3), covering every frequency class,
2 lighting conditions and both `roi_mode` values.

| Stage compared | Metric | Threshold |
|---|---|---|
| After step 3 (grayscale) | mean absolute difference | `≤ 1.0 / 255` |
| After step 3 | max absolute difference | `≤ 2.0 / 255` |
| After step 4 (CLAHE) | mean absolute difference | `≤ 1.0 / 255` |
| After step 4 (CLAHE) | max absolute difference | `≤ 3.0 / 255` |
| After step 6 (Canny binary) | IoU of edge pixels | `≥ 0.995` |
| After step 7 (resize) | mean absolute difference | `≤ 1.0 / 255` |
| After step 8 (tensor) | mean absolute difference | `≤ 2e-3` |
| End-to-end (with model) | argmax class agreement | `100 %` |
| End-to-end | max abs difference in softmax | `≤ 1e-3` |

A failure prints the first offending pixel coordinate and both values. Do not raise a
threshold to make the gate pass — find the divergent step.

## 5. Known divergence traps

Read this before debugging any parity failure.

**5.1 Canvas luma ≠ OpenCV luma.** `ctx.getImageData` then `0.299r+0.587g+0.114b` in float
JS gives values that differ from `cv2.cvtColor` because OpenCV uses fixed-point arithmetic with
specific rounding. The measured deviation is one level; the 2/255 tolerance above is deliberately
conservative. Always go through `cv.cvtColor` in opencv.js.

Note also that `opencv-python` wheels dispatch to IPP/AVX kernels that the WASM build does not
have. Bit-identity is likely but not guaranteed — which is exactly why the tolerances above are
non-zero rather than exact-equality assertions.

**5.2 `drawImage` resizing is browser-defined.** Chrome, Firefox and Safari use different
downscaling filters and none of them is `INTER_AREA`. Always use `cv.resize` with the
explicit interpolation flag. Never resize via CSS or canvas dimensions.

**5.3 JPEG decoders differ.** Pillow/libjpeg-turbo and the browser's decoder disagree by
±1–2 levels on the same JPEG. The golden set is therefore **PNG only**. In production the
browser decodes user JPEGs and this ±2 error is absorbed by the tolerance in step 3 — verify
that claim with `tests/jpeg_robustness.py`, do not assume it.

**5.4 iPhone EXIF and HEIC.** iPhone 11 writes EXIF orientation and a Display P3 ICC profile.
Untreated, the same photo is rotated 90° on one side and colour-shifted on the other. Ingest
converts HEIC→PNG, applies orientation, converts P3→sRGB and strips both, **once**, in
`make ingest`. The browser path calls `createImageBitmap(blob, {imageOrientation: 'from-image',
colorSpaceConversion: 'default'})`.

**5.5 opencv.js version pinning.** `opencv.js` must be the same OpenCV minor version as
`opencv-python`. Pin both in `config/versions.json` and assert at parity-check start. A minor
version bump has changed CLAHE's tile interpolation before.

**5.6 Float accumulation order.** Do the `*1/255` scale as the last operation on both sides,
on the full `uint8` array, never fused into the resize.

## 6. Changing this spec

1. Bump `version` in `config/preprocess.json` (semver; any pipeline change is at least a minor).
2. Update this file in the **same commit**.
3. Re-run `npm run parity` and `make train` — a preprocessing change invalidates the model.
4. Add an ADR under `docs/adr/` if the change alters the pipeline shape, not just a constant.
