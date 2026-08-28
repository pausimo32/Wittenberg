"""Ingest quality gate — docs/02-dataset-spec.md §6.

Decides the `quality` column of `data/metadata.csv` for one raw capture: `ok`,
`blurred`, `partial` or `reject`. A rejected image is never deleted — it stays in
`data/raw/` and is counted in `data/ingest_report.json` as the project's honest
yield figure (research question S4).

Like `preprocess`, this is a pure function: no file I/O, no globals, and the input
array is never mutated. `load_gate` is the single I/O boundary and sits outside the
decision path.

**Every measurement is taken on the pipeline's own output.** The gate calls
`preprocess_stages` and reads `gray` and `edges` from it rather than re-implementing
grayscale or Canny, so the numbers describe exactly the image the model will see.
Forking that would recreate the drift `CLAUDE.md` rule 1 exists to prevent.

Two things the spec left open, decided here and recorded in the docstring because
`docs/02` §6 states the thresholds but not the procedure:

*Order of evaluation.* §6 lists blur first, but the marker check has to run before
anything else: with fewer than four markers there is no homography, so there is no
ROI on which to measure blur, luminance or edges at all. The order is therefore
markers → blur → luminance → edge density, and the reported `reason` is the first
rule that fails. A capture failing several is counted once, under the first.

*Where each metric is measured.* Marker count comes from the raw frame (that is where
the markers are). Blur, luminance and edge density are measured on the warped 512x512
ROI, so a session photographed at a different distance is not judged on a different
number of pixels.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

import cv2
import numpy as np

from preprocess import ROI_MODE_HOMOGRAPHY, Stages, preprocess_stages

GATE_PATH = Path(__file__).resolve().parents[1] / "config" / "quality_gate.json"

QUALITY_OK = "ok"
QUALITY_BLURRED = "blurred"
QUALITY_PARTIAL = "partial"
QUALITY_REJECT = "reject"

#: Written to `metadata.csv`; `docs/02` §4 fixes this vocabulary.
QUALITY_VALUES = (QUALITY_OK, QUALITY_BLURRED, QUALITY_PARTIAL, QUALITY_REJECT)

#: `reason` -> the `quality` recorded. Several reasons collapse to `reject`, which is
#: why the report breaks yield down by reason and not by quality alone.
REASON_QUALITY = {
    "partial": QUALITY_PARTIAL,
    "blurred": QUALITY_BLURRED,
    "luminance": QUALITY_REJECT,
    "edge_density": QUALITY_REJECT,
}


class GateResult(NamedTuple):
    """`quality` for the metadata row, `reason` for the yield report, `metrics` for the TDR.

    `reason` is `None` exactly when `quality` is `ok`. `metrics` always carries every
    number the gate managed to measure — a `partial` result has no ROI, so it carries
    only `n_markers` and `roi_mode`.
    """

    quality: str
    reason: str | None
    metrics: dict[str, Any]

    @property
    def accepted(self) -> bool:
        return self.quality == QUALITY_OK


def load_gate(path: Path | str | None = None) -> dict[str, Any]:
    """Read `config/quality_gate.json`. The only I/O in this module."""
    return json.loads(Path(GATE_PATH if path is None else path).read_text("utf-8"))


def assess(
    image_rgba: np.ndarray,
    calib: Mapping[str, Any] | None,
    n_markers: int,
    config: Mapping[str, Any] | None = None,
    gate: Mapping[str, Any] | None = None,
    stages: Stages | None = None,
) -> GateResult:
    """Apply §6 to one decoded capture.

    `image_rgba` is 8-bit RGBA straight from `imageio.decode`. `calib` is the session
    calibration. `n_markers` is how many ArUco markers ingest actually found in the
    raw frame — the gate does not detect them itself, so it stays pure and testable
    without a marker board.

    `stages` is the already-computed pipeline, for the one caller that has it.
    `/api/predict` runs `preprocess_stages` to produce its answer, and `docs/05` §3
    forbids running it a second time to measure the same image; ingest passes nothing
    and the gate computes its own. It is a shortcut, never a different measurement.
    """
    params = gate if gate is not None else load_gate()
    mode = _roi_mode(calib, config)
    metrics: dict[str, Any] = {"n_markers": int(n_markers), "roi_mode": mode}

    # Rule 2 first — see the module docstring. Never a silent fallback to center_crop:
    # `docs/02` §6 requires such a frame to be rejected as `partial`.
    if mode == ROI_MODE_HOMOGRAPHY and n_markers < int(params["markers_required"]):
        return GateResult(QUALITY_PARTIAL, "partial", metrics)

    stages = preprocess_stages(image_rgba, calib, config) if stages is None else stages
    gray = stages.gray

    metrics["laplacian_variance"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    metrics["mean_luminance"] = float(gray.mean())

    margin = _interior_margin_px(stages.edges, params["edge_density"]["interior_margin_frac"])
    metrics["interior_margin_px"] = margin
    metrics["edge_density"] = _edge_density(stages.edges, margin)
    # Kept for the TDR: the difference between the two readings is what justifies the
    # interior crop, and `test_empty_interior_full_border_is_rejected` asserts on it.
    metrics["edge_density_full_frame"] = _edge_density(stages.edges, 0)

    reason = _first_failing_rule(metrics, params)
    if reason is None:
        return GateResult(QUALITY_OK, None, metrics)
    return GateResult(REASON_QUALITY[reason], reason, metrics)


# --------------------------------------------------------------------------- #
# rules
# --------------------------------------------------------------------------- #

def _first_failing_rule(metrics: Mapping[str, Any], gate: Mapping[str, Any]) -> str | None:
    if metrics["laplacian_variance"] < float(gate["laplacian_variance_min"]):
        return "blurred"

    luminance = gate["mean_luminance"]
    if not float(luminance["min"]) <= metrics["mean_luminance"] <= float(luminance["max"]):
        return "luminance"

    density = gate["edge_density"]
    if not float(density["min"]) <= metrics["edge_density"] <= float(density["max"]):
        return "edge_density"

    return None


def _interior_margin_px(edges: np.ndarray, margin_frac: float) -> int:
    """The band excluded per side, in pixels of the ROI."""
    side = min(edges.shape[:2])
    return round(side * float(margin_frac))


def _edge_density(edges: np.ndarray, margin: int) -> float:
    """Fraction of lit pixels in `edges`, ignoring `margin` pixels on every side.

    With `margin` at the configured 2 %, an image whose only content is a bright frame
    at the very edge reads 0.0 here and would read well above the 0.5 % floor at
    `margin=0`. That gap is the whole point: the frame is rig, not pattern.
    """
    interior = edges[margin: edges.shape[0] - margin, margin: edges.shape[1] - margin]
    if interior.size == 0:
        return 0.0
    return float(np.count_nonzero(interior) / interior.size)


def _roi_mode(calib: Mapping[str, Any] | None, config: Mapping[str, Any] | None) -> str:
    """Mirror of `preprocess._roi_mode`: the calibration wins over the config default."""
    if calib is not None and calib.get("roi_mode") is not None:
        return str(calib["roi_mode"])
    from preprocess import load_config

    params = load_config() if config is None else config
    return str(params["roi"]["mode"])
