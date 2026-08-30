"""The local serving API — `docs/05-webapp-spec.md` §3, and nothing more than §3.

FastAPI on `127.0.0.1`, one process, one worker, the model loaded once. Not deployed, not
reachable from another device, no outbound call of any kind (`docs/adr/0004`, `0005`).

**The one property this module exists to preserve.** `/api/predict` calls
`preprocess.preprocess_stages` — the function training calls, with the config training read.
It does not reimplement a step, does not inline one "for speed", does not wrap one in a
faster equivalent. `docs/adr/0004` reorganised the whole architecture around having exactly
one preprocessing implementation; the moment a step is copied here, the app and the
checkpoint drift and *nothing reports it* — the images look the same and the probabilities
look the same. The same rule governs step 1: HEIC/EXIF/Display P3 normalisation comes from
`imageio.decode_bytes`, the function ingest uses (§5.1), and step 8 plus the NHWC->NCHW
permute come from `dataset.to_model_input`, the function training uses.

**Startup is a gate, not a warning.** Before the first request the checkpoint's
`preprocess_sha256` is compared with the SHA-256 of the working `config/preprocess.json`, and
the checkpoint's OpenCV build with the installed one (`docs/adr/0006`). A mismatch raises
`StartupRefused` with both hashes printed in full, and the server does not start. `docs/03`
§4: "This is a hard failure, never a warning."

**One call returns the prediction and every stage.** §3 and §8 both forbid running the
pipeline twice — once to display and once to predict — because the user would then be shown
images that did not produce the answer next to them. So the single forward pass carries
hooks: the three block activations become stages 9-11, and `cam.gradcam` turns the same
backward pass into stage 12.

**Every time on screen was measured.** §8 shows the real cost of each step next to its
image, so each stage carries `ms`: steps 2-8 from `preprocess_stages(collect_timings=True)`,
the three blocks from the gaps between the forward hooks, step 1 and Grad-CAM from the timers
around them. A step nobody timed carries `ms: null` rather than a plausible number — the
viewer is didactic, and a clean invented figure teaches worse than a blank.

**`/api/predict` writes nothing to disk.** Uploads are decoded from the request body, every
stage is PNG-encoded in memory, and there is no upload directory to configure. The one part
of this process that does write is `/dades` (§9), whose routes live in `src/dataset_api.py`
and whose writes go to `drafts/` until the moment someone presses commit.
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Annotated, Any, NamedTuple, Self

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.staticfiles import StaticFiles
from torch import nn

import preprocess as preprocess_module
import runs as runs_module
from calibrate import (
    ROI_MODE_CENTER_CROP,
    ROI_MODE_HOMOGRAPHY,
    calibrate,
    load_calibration_config,
)
from cam import gradcam
from dataset import to_model_input
from dataset_api import DatasetState, dataset_router
from drafts import DRAFTS_ROOT
from imageio import UnsupportedImageError, decode_bytes, thumbnail
from model import build_model
from preprocess import load_config, preprocess_stages
from quality import assess, load_gate

ROOT = Path(__file__).resolve().parents[1]
CLASSES_PATH = ROOT / "config" / "classes.json"
DATA_ROOT = ROOT / "data"
SITE_DIR = ROOT / "site"

HOST = "127.0.0.1"
PORT = 8000


class RevalidatedStaticFiles(StaticFiles):
    """`StaticFiles`, but the browser is told to check back every time.

    Starlette sends `etag` and `last-modified` and no `Cache-Control`. With no directive
    the browser is free to guess how long a file stays fresh, and Chrome and Edge guess
    from the age of the file: a stylesheet last modified an hour ago is treated as good
    for another six minutes, and is served from disk cache without ever asking. Editing
    `site/css/style.css` and reloading then shows the old page, which reads exactly like
    "the change did not apply" — and cost an afternoon once already.

    `no-cache` does not mean "do not store". The browser keeps the file and still
    revalidates: it sends the `etag` back, and an unchanged file answers `304` with an
    empty body. So the traffic saved by caching is kept, and only the guessing is
    removed. On `127.0.0.1` the extra round trip is under a millisecond.
    """

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["cache-control"] = "no-cache"
        return response

#: §4: top-3, never a bare single number.
TOP_K = 3

#: §8: "a small grid of channels (8 of the 32, 64 and 128, chosen once and fixed so the
#: display is stable across images)". The first eight, fixed here and nowhere else.
FEATURE_CHANNELS = 8
FEATURE_GRID = (2, 4)
FEATURE_UPSCALE = 4

#: Stage 1 is shown, never used, so only the *displayed* copy is capped — the pipeline still
#: receives the caller's full-resolution array.
#:
#: Measured, on a 4032x3024 phone-sized capture, encoding stage 1 alone:
#:
#: | cap        | PNG    | encode  |
#: |------------|--------|---------|
#: | 512        | 281 KB |    8 ms |
#: | **768**    | 590 KB |   19 ms |
#: | 1024       |   1 MB |   32 ms |
#: | uncapped   |  24 MB |  459 ms |
#:
#: Uncapped is a 3x breach of the 150 ms budget on the first real phone photo, in one stage,
#: and nothing else in the request comes close. 768 rather than 1024 because the extra 13 ms
#: and doubled payload buy detail nobody looks at: the viewer shows stage 1 in a card beside
#: twelve others. The original resolution travels in `shape`, so nothing is hidden.
DISPLAY_MAX_SIDE = 768

#: §8: stage 12 blends over the ROI at the alpha `evaluate` uses for the annex figure.
CAM_ALPHA = 0.45


class StartupRefused(RuntimeError):
    """The server must not start: the checkpoint and this working copy disagree."""


class Serving(NamedTuple):
    """Everything loaded once, at startup, and read-only afterwards."""

    run_dir: Path
    config: dict[str, Any]
    calibration_config: dict[str, Any]
    classes: list[dict[str, Any]]
    frequencies: list[int]
    model: nn.Module
    model_name: str
    device: torch.device
    preprocess_sha256: str
    opencv: dict[str, str]
    synthetic: bool
    do_not_report: str | None
    #: `config/quality_gate.json`, read once. §6's thresholds, applied to an upload as a
    #: warning rather than a rejection — see `_predict`.
    gate: dict[str, Any]


# --------------------------------------------------------------------------- #
# startup
# --------------------------------------------------------------------------- #

def load_serving(
    argv: list[str] | None = None,
    runs_root: Path | str | None = None,
    config_path: Path | str | None = None,
    classes_path: Path | str | None = None,
) -> Serving:
    """Resolve the run, run both gates, load the model. Raises `StartupRefused`."""
    config = _current_preprocess_config(config_path)

    try:
        run_dir = runs_module.resolve_run(list(argv or []), runs_root)
        absent = runs_module.missing_artefacts(run_dir)
        if absent:
            raise runs_module.RunsError(
                f"{run_dir.name} is missing {', '.join(absent)} — it is not a finished run"
            )
    except runs_module.RunsError as exc:
        raise StartupRefused(f"REFUSING TO START — {exc}") from exc

    checked = _gate(run_dir, config)

    checkpoint = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    classes = _classes(checkpoint, run_dir, classes_path)
    metrics = _metrics(run_dir)

    # ADR 0005 makes CPU the *supported* configuration of the app, not a fallback —
    # inference is ~12 ms there, and the app must start on any computer the team picks.
    # CUDA is used if it happens to be present. `train.select_device` states the same
    # policy; it is not imported because that module pulls in the whole training stack.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(checkpoint["model"], checkpoint["n_classes"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    return Serving(
        run_dir=run_dir,
        config=config,
        calibration_config=load_calibration_config(),
        classes=classes,
        frequencies=[int(c["freq_hz"]) for c in classes],
        model=model,
        model_name=str(checkpoint["model"]),
        device=device,
        preprocess_sha256=checked["sha256"],
        opencv={
            "package": checked["opencv_package"],
            "version": checked["opencv_version"],
        },
        synthetic=bool(metrics.get("synthetic", False)),
        do_not_report=metrics.get("do_not_report"),
        gate=load_gate(),
    )


def _metrics(run_dir: Path) -> dict[str, Any]:
    """`metrics.json`, for the one field the app has to surface: `do_not_report`.

    `src/train.py` writes it into the metrics of any run whose sessions are `SYNTH-*`, and
    the text says what the accuracy of such a run actually measures — how well the network
    inverted the closed-form mode shape that drew the images, not anything about a plate.
    The app serves that checkpoint perfectly happily; what it must not do is let the number
    on screen look like a result. `/api/health` exposes the field so the frontend has
    something to hang a visible banner on.

    A `metrics.json` that cannot be parsed is a refusal, not a shrug. The failure direction
    matters: defaulting to "no `do_not_report` key" would silently promote an unreadable
    synthetic run to a reportable one, which is the exact mistake the field exists to stop.
    """
    path = run_dir / "metrics.json"
    try:
        payload = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError) as exc:
        raise StartupRefused(
            f"REFUSING TO START — {run_dir.name}/metrics.json cannot be read ({exc}), so "
            f"whether this run's numbers may be reported cannot be determined. A run whose "
            f"provenance is unreadable is not servable for the same reason an unverifiable "
            f"preprocessing hash is not."
        ) from exc
    if not isinstance(payload, dict):
        raise StartupRefused(
            f"REFUSING TO START — {run_dir.name}/metrics.json is not an object"
        )
    return payload


def _current_preprocess_config(config_path: Path | str | None) -> dict[str, Any]:
    """Read `config/preprocess.json` from disk, past `preprocess`'s read cache.

    `preprocess.load_config` memoises per path so the transform never pays for I/O in a
    training loop. Startup is the one caller that must see the file as it is *now*: a test
    that edits the config and re-starts the app in the same process would otherwise be
    gated against the copy read at import time, and the guard would pass on a config that
    no longer exists.
    """
    preprocess_module._read_config.cache_clear()
    return load_config(config_path)


def _gate(run_dir: Path, config: dict[str, Any]) -> dict[str, str]:
    """`docs/03` §4 and `docs/adr/0006`, both keys, exactly as `tasks.py check-model`."""
    try:
        return runs_module.check_run(run_dir, config)
    except runs_module.PreprocessMismatch as exc:
        stored = runs_module.read_run_config(run_dir)
        raise StartupRefused(
            "REFUSING TO START — the checkpoint was trained under a different "
            "config/preprocess.json.\n"
            f"  checkpoint {run_dir.name}\n"
            f"    preprocess version {stored.get('preprocess_version')}\n"
            f"    sha256 {stored.get('preprocess_sha256')}\n"
            f"  config/preprocess.json (working copy)\n"
            f"    preprocess version {config.get('version')}\n"
            f"    sha256 {runs_module.preprocess_hash(config)}\n"
            "\n"
            f"{exc}"
        ) from exc
    except runs_module.RunsError as exc:
        raise StartupRefused(f"REFUSING TO START — {exc}") from exc


def _classes(
    checkpoint: dict[str, Any], run_dir: Path, classes_path: Path | str | None
) -> list[dict[str, Any]]:
    """The checkpoint's class list, cross-checked against `config/classes.json`.

    The checkpoint is the authority — its head is sized for that list and it stores a class
    *index*, not a frequency. `config/classes.json` is checked against it and never merged
    into it: a list that has been regenerated since training renumbers the classes, and the
    app would then answer a different frequency at full confidence, which is precisely the
    failure `src/model.py` describes. Same treatment as the preprocessing hash — refuse.
    """
    stored = [
        {"index": int(c["index"]), "freq_hz": int(c["freq_hz"])}
        for c in checkpoint["classes"]
    ]
    path = Path(CLASSES_PATH if classes_path is None else classes_path)
    if not path.exists():
        raise StartupRefused(
            f"REFUSING TO START — {path} not found. {run_dir.name} was trained against a "
            f"class list that is not in the working copy, so the frequency each class index "
            f"means cannot be confirmed. Run `python tasks.py modes` first."
        )

    payload = json.loads(path.read_text("utf-8"))
    live = [
        {"index": int(c["index"]), "freq_hz": int(c["freq_hz"])}
        for c in (payload.get("classes") or [])
    ]
    if live != stored:
        raise StartupRefused(
            "REFUSING TO START — the class list has changed since the checkpoint was "
            "trained.\n"
            f"  checkpoint {run_dir.name}: {[c['freq_hz'] for c in stored]}\n"
            f"  {path.name}: {[c['freq_hz'] for c in live]}\n"
            "The class index is frozen and is what the checkpoint stores instead of a "
            "frequency; serving it against a renumbered list answers the wrong frequency "
            "at full confidence. Retrain, or restore the list the run was made with."
        )
    return stored


# --------------------------------------------------------------------------- #
# the app
# --------------------------------------------------------------------------- #

def create_app(
    argv: list[str] | None = None,
    runs_root: Path | str | None = None,
    config_path: Path | str | None = None,
    classes_path: Path | str | None = None,
    site_dir: Path | str | None = None,
    data_root: Path | str | None = None,
    drafts_root: Path | str | None = None,
) -> FastAPI:
    """Load everything, then build the routes. Raises `StartupRefused` before binding.

    The loading happens here rather than in an `on_startup` handler on purpose: a refusal
    must stop the process before uvicorn opens a socket, not after it has told the user the
    server is running.
    """
    serving = load_serving(argv, runs_root, config_path, classes_path)

    # No CORS middleware, by omission: the built frontend is served from this same origin,
    # and in development Vite proxies /api to it (§2). A wildcard here would let any page
    # the browser happens to open drive the API on the user's behalf.
    app = FastAPI(title="Wittenberg", version=str(serving.config.get("version", "?")))
    app.state.serving = serving

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        """§3. What is loaded, so a wrong answer can be traced to a run without guessing.

        `do_not_report` is `null` for a run the TDR may quote and a sentence of prose for
        one it may not — see `_metrics`. It is a *field*, not a log line, precisely so the
        frontend can make it visible: a synthetic run reaches ~100 % accuracy, and a
        participant shown that number with nothing on screen to qualify it has been
        misled by the app rather than by anyone in particular.
        """
        return {
            "model_run": serving.run_dir.name,
            "preprocess_version": serving.config.get("version"),
            "preprocess_sha256": serving.preprocess_sha256,
            "device": serving.device.type,
            "classes": serving.classes,
            "model": serving.model_name,
            "opencv": serving.opencv,
            "synthetic": serving.synthetic,
            "do_not_report": serving.do_not_report,
        }

    @app.post("/api/predict")
    async def predict(
        file: Annotated[UploadFile, File()],
        stages: Annotated[
            bool, Query(description="false returns only the final edge map")
        ] = True,
    ) -> dict[str, Any]:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="empty upload")
        try:
            return _predict(serving, data, file.filename, with_stages=stages)
        except UnsupportedImageError as exc:
            raise HTTPException(status_code=415, detail=str(exc)) from exc

    # §9 — `/dades`. The only routes in this process that write anything, and they live in
    # their own module so §3's "nothing is written to disk" stays a readable property of the
    # code above rather than a claim with an exception attached. Included before the static
    # mount for the same reason that mount comes last: `/` catches everything.
    app.include_router(
        dataset_router(
            DatasetState(
                data_root=Path(DATA_ROOT if data_root is None else data_root),
                drafts_root=Path(DRAFTS_ROOT if drafts_root is None else drafts_root),
                calibration_config=serving.calibration_config,
                preprocess_config=serving.config,
                gate=load_gate(),
            )
        )
    )

    site = Path(SITE_DIR if site_dir is None else site_dir)
    if site.is_dir():
        # `RevalidatedStaticFiles` rather than `StaticFiles`: see its docstring. Editing a
        # page and reloading has to show the edit.
        # Mounted last so it cannot shadow /api. `html=True` serves index.html for `/`.
        #
        # No SPA fallback, and no `directory=ROOT`. Both were deliberate: this is a
        # multipage site where `/analyseur.html` is a real file, so a mistyped page must
        # stay a 404 rather than silently render the home page; and the doc root is
        # `site/` rather than the repo it lives in, because mounting the repo would put
        # `data/raw/`, `runs/`, `src/` and `.git/` on an HTTP port. Binding to 127.0.0.1
        # is not a reason to serve them.
        app.mount("/", RevalidatedStaticFiles(directory=site, html=True), name="site")

    return app


# --------------------------------------------------------------------------- #
# /api/predict — one decode, one pipeline, one forward pass
# --------------------------------------------------------------------------- #

def _predict(
    serving: Serving, data: bytes, filename: str | None, with_stages: bool
) -> dict[str, Any]:
    timings: dict[str, float] = {}
    started = time.perf_counter()

    with _timed(timings, "decode"):
        decoded = decode_bytes(data, filename)  # step 1 — §5.1, ingest's own decoder

    with _timed(timings, "calibrate"):
        calib, markers = _calibration(serving, decoded.pixels)

    with _timed(timings, "preprocess"):
        # THE call. Steps 2-8, from `src/preprocess.py`, with the config the checkpoint was
        # gated against. Once — these stages are the ones that produce the answer below.
        # `collect_timings` is the pipeline measuring itself; it is not a second run.
        pipeline = preprocess_stages(
            decoded.pixels, calib, serving.config, collect_timings=True
        )

    with _timed(timings, "quality"):
        # `docs/02` §6's gate, run on the upload for information only. It is the same
        # function `tasks.py ingest` calls, never a second copy — but here it *warns and
        # never rejects*: a blurred photo still gets an answer and is still told it is
        # blurred. Refusing it would invent a policy the analyser never had; the gate
        # decides what enters the dataset, not what a participant may ask.
        #
        # `stages=pipeline` is what keeps this from being a second run of the pipeline,
        # which §3 forbids: the gate measures the arrays the answer came from.
        verdict = assess(
            decoded.pixels, calib, markers, serving.config, serving.gate, stages=pipeline
        )

    with _timed(timings, "inference"):
        # Step 8 + the NHWC->NCHW permute by `dataset.to_model_input`, from `stages.resized`
        # (step 7) — the identical path `train.predict` and `evaluate` take, so the tensor
        # entering the model here is the tensor training would have built.
        x = to_model_input(
            torch.from_numpy(pipeline.resized).unsqueeze(0), serving.config, serving.device
        )
        blocks: list[tuple[torch.Tensor, float]] = []
        handles = _hook_blocks(serving.model, blocks)
        try:
            forward_started = time.perf_counter()
            logits = serving.model(x)
            probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        finally:
            for handle in handles:
                handle.remove()

    top = [int(i) for i in np.argsort(-probabilities)[:TOP_K]]

    body: dict[str, Any] = {
        "top3": [
            {
                "class_index": i,
                "freq_hz": serving.frequencies[i],
                "p": float(probabilities[i]),
            }
            for i in top
        ],
        "roi_mode": str(calib["roi_mode"]),
        "markers_found": markers,
        "quality": {
            "quality": verdict.quality,
            "reason": verdict.reason,
            "metrics": verdict.metrics,
        },
    }

    if with_stages:
        with _timed(timings, "gradcam"):
            # The same input the answer came from, run backward through the same weights.
            # §8: never recompute anything for display.
            cam = gradcam(serving.model, x, top[0])
        with _timed(timings, "encode"):
            body["stages"] = _stages(
                serving,
                pipeline,
                blocks,
                forward_started,
                cam,
                probabilities,
                decode_ms=timings["decode"],
                gradcam_ms=timings["gradcam"],
            )
    else:
        with _timed(timings, "encode"):
            body["stages"] = [_edge_stage(serving, pipeline)]

    body["timings_ms"] = timings
    body["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
    return body


class _timed:
    """Wall-clock for one phase, in milliseconds, into `timings`.

    Phases, not stages. The per-stage numbers §8 wants next to each image are measured
    where the work happens and travel on the stage itself: steps 2-8 by
    `preprocess_stages(collect_timings=True)`, the three blocks by the forward hooks, step
    1 by this timer around `decode_bytes`. Nothing is re-run to be measured, and nothing
    is estimated — a stage whose cost was not measured carries `ms: null` rather than a
    plausible number.
    """

    def __init__(self, timings: dict[str, float], name: str):
        self.timings = timings
        self.name = name

    def __enter__(self) -> Self:
        self.started = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.timings[self.name] = round((time.perf_counter() - self.started) * 1000.0, 3)


def _calibration(serving: Serving, pixels: np.ndarray) -> tuple[dict[str, Any], int]:
    """The ROI geometry for an upload that carries no session calibration.

    Ingest reads `data/calib/<session_id>.json`; an upload has none, so the markers are
    detected in the frame itself. Four found means the same homography the training frames
    went through. Fewer means `center_crop`, which is a *different* geometry from the one
    the checkpoint was trained on — so the mode is reported in the response rather than
    applied silently, and the UI can say the plate was not squared.
    """
    found = calibrate(pixels, "upload", serving.calibration_config)
    if found.corners is not None:
        return (
            {"roi_mode": ROI_MODE_HOMOGRAPHY, "corners": found.corners},
            found.n_markers,
        )
    return {"roi_mode": ROI_MODE_CENTER_CROP}, found.n_markers


def _hook_blocks(
    model: nn.Module, sink: list[tuple[torch.Tensor, float]]
) -> list[Any]:
    """Forward hooks on each pooled convolutional block — stages 9-11 of §8.

    Each hook records the activation *and* the clock, so the block's own cost is the gap
    between consecutive hooks (the first measured from the start of the forward pass).
    That is a measurement of the real forward, not a separate timing run: §8's rule that
    nothing is recomputed for display covers the times as much as the images.

    The pools are found by type rather than by index so the hooks survive an architecture
    change and work unchanged on `baseline_paper`, which has one block instead of three.
    """
    return [
        module.register_forward_hook(
            lambda _m, _i, o: sink.append((o.detach(), time.perf_counter()))
        )
        for module in model.features
        if isinstance(module, nn.MaxPool2d)
    ]


# --------------------------------------------------------------------------- #
# §8 — the thirteen stages
# --------------------------------------------------------------------------- #

def _stages(
    serving: Serving,
    pipeline: Any,
    blocks: list[tuple[torch.Tensor, float]],
    forward_started: float,
    cam: np.ndarray,
    probabilities: np.ndarray,
    decode_ms: float,
    gradcam_ms: float,
) -> list[dict[str, Any]]:
    cfg = serving.config
    step = pipeline.timings_ms or {}
    stages = [
        _stage(1, "capture", "Captura", pipeline.source, "imageio.decode_bytes", {},
               display=_downscaled(pipeline.source), ms=decode_ms),
        _stage(2, "roi", "ROI", pipeline.roi, "cv2.warpPerspective", cfg["roi"],
               ms=step.get("roi")),
        _stage(3, "gray", "Escala de grisos", pipeline.gray, "cv2.cvtColor",
               cfg["grayscale"], ms=step.get("gray")),
        _stage(4, "clahe", "CLAHE", pipeline.clahe, "cv2.createCLAHE().apply", cfg["clahe"],
               ms=step.get("clahe")),
        _stage(5, "blur", "Difuminat", pipeline.blur, "cv2.GaussianBlur", cfg["blur"],
               ms=step.get("blur")),
        _stage(6, "edges", "Canny", pipeline.edges, "cv2.Canny", cfg["canny"],
               ms=step.get("edges")),
        _stage(7, "resized", "Redimensionat", pipeline.resized, "cv2.resize", cfg["resize"],
               ms=step.get("resized")),
        _stage(8, "tensor", "Tensor", pipeline.tensor, "astype * scale", cfg["output"],
               display=_from_float(pipeline.tensor), ms=step.get("tensor")),
    ]

    previous = forward_started
    for n, (activation, finished) in zip(range(9, 12), blocks):
        stages.append(_feature_stage(n, activation, (finished - previous) * 1000.0))
        previous = finished

    stages.append(
        {
            "n": 12,
            "key": "gradcam",
            "title": "Grad-CAM",
            "kind": "image",
            "call": "cam.gradcam",
            "params": {"layer": "cam_layer", "alpha": CAM_ALPHA},
            "shape": list(cam.shape),
            "ms": round(gradcam_ms, 3),
            "png_base64": _png(_cam_overlay(pipeline.roi, cam)),
        }
    )
    stages.append(
        {
            "n": 13,
            "key": "probabilities",
            "title": "Probabilitats",
            "kind": "bars",
            "call": "torch.softmax",
            "params": {"dim": 1},
            "ms": None,
            "bars": [
                {"class_index": i, "freq_hz": hz, "p": float(probabilities[i])}
                for i, hz in enumerate(serving.frequencies)
            ],
        }
    )
    return stages


def _edge_stage(serving: Serving, pipeline: Any) -> dict[str, Any]:
    """`?stages=false` — the final edge map alone, for the rare caller that wants no more."""
    return _stage(
        7, "resized", "Redimensionat", pipeline.resized, "cv2.resize",
        serving.config["resize"],
    )


def _stage(
    n: int,
    key: str,
    title: str,
    array: np.ndarray,
    call: str,
    params: dict[str, Any],
    display: np.ndarray | None = None,
    ms: float | None = None,
) -> dict[str, Any]:
    """One stage. `params` comes from the config, never retyped — §8's rule for the viewer.

    `shape` and `dtype` describe the *pipeline* array — the real one, at its real size.
    `display` is what is actually encoded when that array is not directly showable (float
    tensors, full-resolution captures); it never re-enters the pipeline, and when it differs
    the response says so in `display_shape` rather than letting `shape` imply the PNG is
    full-size. The UI shows the original resolution beside a downscaled stage.

    `ms` is the measured cost of this step on this request — §8: "Show the *real measured*
    time of each step next to it". It is `None` when the step was not measured, never a
    stand-in figure: a number on screen that nobody timed is worse than a blank, because
    the reader cannot tell which one they are looking at.
    """
    return {
        "n": n,
        "key": key,
        "title": title,
        "kind": "image",
        "call": call,
        "params": params,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "display_shape": (
            None if display is None or display.shape == array.shape else list(display.shape)
        ),
        "ms": None if ms is None else round(ms, 3),
        "png_base64": _png(array if display is None else display),
    }


def _feature_stage(n: int, activation: torch.Tensor, ms: float) -> dict[str, Any]:
    """Stages 9-11: eight channels of one block, tiled.

    Deliberately not labelled with what any channel "detects" — §8 says not to claim that
    without having checked the specific channel, and this endpoint has not.
    """
    maps = activation[0, :FEATURE_CHANNELS].cpu().numpy()
    return {
        "n": n,
        "key": f"features_{n - 8}",
        "title": f"Mapes de característiques {n - 8}",
        "kind": "image",
        "call": "forward hook",
        "params": {
            "channels_shown": int(maps.shape[0]),
            "channels_total": int(activation.shape[1]),
            "grid": list(FEATURE_GRID),
            "upscale": FEATURE_UPSCALE,
        },
        "shape": list(activation.shape[1:]),
        "dtype": str(activation.dtype).replace("torch.", ""),
        "ms": round(ms, 3),
        "png_base64": _png(_feature_grid(maps)),
    }


def _feature_grid(maps: np.ndarray) -> np.ndarray:
    """`(C, h, w)` activations -> one uint8 tile sheet, nearest-neighbour upscaled.

    Each map is normalised on its own: the blocks differ by orders of magnitude in scale,
    and a shared normalisation would render the later ones black.
    """
    rows, cols = FEATURE_GRID
    tiles = [_normalised(m) for m in maps]
    tiles += [np.zeros_like(tiles[0])] * (rows * cols - len(tiles))
    sheet = np.vstack([np.hstack(tiles[r * cols: (r + 1) * cols]) for r in range(rows)])
    return cv2.resize(
        sheet,
        (sheet.shape[1] * FEATURE_UPSCALE, sheet.shape[0] * FEATURE_UPSCALE),
        # INTER_NEAREST, never a smoothing kernel: at this size the pixel grid is the
        # information, and interpolating it would invent structure the block did not find.
        interpolation=cv2.INTER_NEAREST,
    )


def _normalised(m: np.ndarray) -> np.ndarray:
    lo, hi = float(m.min()), float(m.max())
    if hi <= lo:
        return np.zeros(m.shape, np.uint8)
    return ((m - lo) * (255.0 / (hi - lo))).astype(np.uint8)


def _cam_overlay(roi: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """Stage 12 over the rectified plate, so the user sees *where on their own photo*."""
    height, width = roi.shape[:2]
    heat = cv2.applyColorMap(
        cv2.resize((cam * 255.0).astype(np.uint8), (width, height),
                   interpolation=cv2.INTER_LINEAR),
        cv2.COLORMAP_JET,
    )
    base = cv2.cvtColor(roi, cv2.COLOR_RGBA2BGR)
    return cv2.addWeighted(heat, CAM_ALPHA, base, 1.0 - CAM_ALPHA, 0.0)


def _downscaled(source: np.ndarray) -> np.ndarray:
    """A display copy of the capture. Never returned to the pipeline — see `_stage`."""
    return thumbnail(source, DISPLAY_MAX_SIDE)


def _from_float(tensor: np.ndarray) -> np.ndarray:
    """Step 8's `float32` [0, 1] back to something a browser can draw. Display only."""
    return np.clip(tensor[..., 0] * 255.0, 0, 255).astype(np.uint8)


def _png(array: np.ndarray) -> str:
    """PNG, base64, in memory. §3: images are never written to disk."""
    if array.ndim == 3 and array.shape[2] == 4:
        array = cv2.cvtColor(array, cv2.COLOR_RGBA2BGRA)
    ok, buffer = cv2.imencode(".png", array)
    if not ok:
        raise RuntimeError(f"cv2.imencode failed for a {array.shape} {array.dtype} stage")
    return base64.b64encode(buffer.tobytes()).decode("ascii")
