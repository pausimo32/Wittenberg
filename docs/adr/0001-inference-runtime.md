# ADR 0001 — Browser-side inference with ONNX Runtime Web + opencv.js

- **Date:** 2026-08-16
- **Status:** accepted
- **Deciders:** Wittenberg team

## Context

The app must classify a Chladni photo by frequency. Training happens offline in Python with
`cv2` + Keras. The question is where inference and, critically, **preprocessing** run.

The dominant risk is not model accuracy — it is **train/serve preprocessing skew**. A model
trained on `cv2`-produced Canny edge maps and served on canvas-produced ones will silently
lose accuracy in a way that looks like a bad model. The reference study (Kumar & Wani 2024)
does not face this because it never leaves Python.

Secondary constraints: no budget for hosting; sessions happen in schools on unreliable Wi-Fi;
participants are minors, so uploading their photos is a consent problem we would rather not have.

## Options

**A. Python backend (FastAPI + cv2 + TF).** Preprocessing is literally the training code, so
zero skew. But it needs a paid host, adds latency, breaks offline, and sends minors' photos to
a server we must then secure and document.

**B. Browser, canvas preprocessing + TF.js.** Free, offline, private. But canvas cannot
reproduce `cv2`: luma rounding differs, `drawImage` downscaling is browser-defined and is not
`INTER_AREA`, and Canny would have to be reimplemented by hand. Skew is guaranteed and
unbounded.

**C. Browser, opencv.js preprocessing + ONNX Runtime Web.** Free, offline, private — and
`opencv.js` is the *same OpenCV C++ core* compiled to WASM, so `cvtColor`, `CLAHE`,
`GaussianBlur`, `Canny` and `resize` are the same implementations as `cv2`. Skew becomes
testable, and small enough to gate on. Cost: a ~8–10 MB lazy chunk.

## Decision

**Option C.**

Skew is the failure mode that would invalidate the project's central claim, and C is the only
option that makes it *measurable* while keeping the app free, offline and private. The download
cost is paid once, lazily, and cached.

## Consequences

- `docs/03-preprocessing-spec.md` becomes normative, and `npm run parity` becomes a merge gate.
- `opencv.js` and `opencv-python` minor versions must be pinned together and asserted at runtime.
- The lazy chunk must stay off the landing route; budget in `docs/05-webapp-spec.md` §5.
- ONNX rather than TF.js because `tf2onnx` + `onnxruntime-web` gives better INT8 quantisation
  and a smaller artefact for this architecture, and decouples the web app from the training
  framework should Keras be replaced later.
- If a future requirement forces a server (e.g. a much larger model), revisit — Option A stays
  viable and the preprocessing spec transfers unchanged.
