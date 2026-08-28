"""Contract tests for `src/quality.py` — the ingest quality gate, docs/02-dataset-spec.md §6.

The gate decides `quality` for every raw capture: `ok`, `blurred`, `partial` or `reject`.
Rejected images stay in `data/raw/` and are counted; nothing here deletes anything.

The load-bearing test in this file is `test_empty_interior_full_border_is_rejected`:
it pins the rule that edge density is measured on the ROI *interior*, because the
whole-frame reading of the same image passes.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from quality import assess, load_gate

ROI = 512


# --------------------------------------------------------------------------- #
# builders — every input is synthetic; no test reads from data/
# --------------------------------------------------------------------------- #

def _rgba(gray: np.ndarray) -> np.ndarray:
    """Grayscale uint8 -> the RGBA uint8 array the pipeline expects."""
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGBA)


def chladni_like(size: int = ROI, level: int = 128) -> np.ndarray:
    """A sharp, mid-grey, patterned plate: the image that must pass every check."""
    g = np.full((size, size), level, np.uint8)
    xs = np.linspace(-np.pi, np.pi, size)
    field = np.outer(np.sin(3 * xs), np.sin(3 * xs))
    g[np.abs(field) < 0.12] = 245           # sand ridges on the nodal lines
    return _rgba(g)


def border_only(size: int = ROI, thickness: int = 6) -> np.ndarray:
    """Empty interior, bright frame hugging the edge — no nodal pattern whatsoever.

    `thickness` is deliberately smaller than the 2 % margin (10 px at 512) so the
    frame falls entirely inside the excluded band.
    """
    g = np.full((size, size), 128, np.uint8)
    g[:thickness, :] = 255
    g[-thickness:, :] = 255
    g[:, :thickness] = 255
    g[:, -thickness:] = 255
    return _rgba(g)


CALIB_CENTER_CROP = {"roi_mode": "center_crop"}
CALIB_HOMOGRAPHY = {
    "roi_mode": "homography",
    "corners": [[0, 0], [ROI - 1, 0], [ROI - 1, ROI - 1], [0, ROI - 1]],
}


@pytest.fixture
def gate() -> dict:
    return load_gate()


# --------------------------------------------------------------------------- #
# the happy path
# --------------------------------------------------------------------------- #

def test_sharp_lit_patterned_plate_passes(gate):
    r = assess(chladni_like(), CALIB_CENTER_CROP, n_markers=4, gate=gate)
    assert r.quality == "ok"
    assert r.reason is None


def test_result_carries_every_measured_metric(gate):
    r = assess(chladni_like(), CALIB_CENTER_CROP, n_markers=4, gate=gate)
    for key in ("laplacian_variance", "mean_luminance", "edge_density", "n_markers"):
        assert key in r.metrics, key


# --------------------------------------------------------------------------- #
# §6 rule 1 — blur
# --------------------------------------------------------------------------- #

def test_blurred_plate_is_blurred(gate):
    src = chladni_like()
    blurred = cv2.GaussianBlur(src, (31, 31), 12.0)
    r = assess(blurred, CALIB_CENTER_CROP, n_markers=4, gate=gate)
    assert r.quality == "blurred"
    assert r.metrics["laplacian_variance"] < gate["laplacian_variance_min"]


# --------------------------------------------------------------------------- #
# §6 rule 2 — markers. Never a silent fallback to center_crop.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_fewer_than_four_markers_is_partial(gate, n):
    r = assess(chladni_like(), CALIB_HOMOGRAPHY, n_markers=n, gate=gate)
    assert r.quality == "partial"
    assert r.reason == "partial"


def test_missing_markers_never_falls_back_to_center_crop(gate):
    """A homography session with 3 markers must be rejected, not quietly re-cropped."""
    r = assess(chladni_like(), CALIB_HOMOGRAPHY, n_markers=3, gate=gate)
    assert r.quality == "partial"
    assert r.metrics.get("roi_mode") == "homography"


def test_center_crop_session_does_not_need_markers(gate):
    r = assess(chladni_like(), CALIB_CENTER_CROP, n_markers=0, gate=gate)
    assert r.quality == "ok"


# --------------------------------------------------------------------------- #
# §6 rule 3 — luminance
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("level", [5, 20, 235, 250])
def test_luminance_outside_range_is_rejected(gate, level):
    """Under-exposed and clipped frames are rejected even when they are perfectly sharp.

    The stripes keep Laplacian variance well above the blur floor so this test isolates
    rule 3; they are sparse enough (1 row in 16) not to drag the mean back into range.
    """
    g = np.full((ROI, ROI), level, np.uint8)
    g[::16, :] = 185 if level < 128 else 55
    r = assess(_rgba(g), CALIB_CENTER_CROP, n_markers=4, gate=gate)
    assert r.metrics["laplacian_variance"] >= gate["laplacian_variance_min"]
    assert r.quality == "reject"
    assert r.reason == "luminance"


# --------------------------------------------------------------------------- #
# §6 rule 4 — edge density, measured on the INTERIOR
# --------------------------------------------------------------------------- #

def test_empty_interior_full_border_is_rejected(gate):
    """The rule this file exists for.

    An image with no pattern at all, but a bright frame at the very edge, passes the
    0.5 % floor when the whole frame is counted and fails it when only the interior
    is. The gate must take the second reading.
    """
    img = border_only()
    r = assess(img, CALIB_CENTER_CROP, n_markers=4, gate=gate)

    assert r.quality == "reject"
    assert r.reason == "edge_density"
    assert r.metrics["edge_density"] < gate["edge_density"]["min"]

    # ...and the whole-frame reading of the very same image would have passed,
    # which is exactly why the interior crop is not cosmetic.
    assert r.metrics["edge_density_full_frame"] > gate["edge_density"]["min"]


def test_interior_margin_is_two_percent_per_side(gate):
    assert gate["edge_density"]["interior_margin_frac"] == 0.02
    img = border_only(thickness=6)
    r = assess(img, CALIB_CENTER_CROP, n_markers=4, gate=gate)
    assert r.metrics["interior_margin_px"] == round(ROI * 0.02)


def test_pure_noise_exceeds_the_edge_ceiling(gate):
    rng = np.random.default_rng(42)
    g = rng.integers(60, 200, size=(ROI, ROI), dtype=np.uint8)
    r = assess(_rgba(g), CALIB_CENTER_CROP, n_markers=4, gate=gate)
    assert r.quality == "reject"
    assert r.reason == "edge_density"
    assert r.metrics["edge_density"] > gate["edge_density"]["max"]


# --------------------------------------------------------------------------- #
# behaviour of the gate itself
# --------------------------------------------------------------------------- #

def test_gate_never_mutates_its_input(gate):
    img = chladni_like()
    before = img.copy()
    assess(img, CALIB_CENTER_CROP, n_markers=4, gate=gate)
    assert np.array_equal(img, before)


def test_gate_is_deterministic(gate):
    img = chladni_like()
    a = assess(img, CALIB_CENTER_CROP, n_markers=4, gate=gate)
    b = assess(img, CALIB_CENTER_CROP, n_markers=4, gate=gate)
    assert a.quality == b.quality
    assert a.metrics == b.metrics


def test_gate_config_is_json_and_versioned(repo_root):
    cfg = json.loads((repo_root / "config" / "quality_gate.json").read_text("utf-8"))
    assert cfg["version"]
    assert cfg["markers_required"] == 4


# --------------------------------------------------------------------------- #
# `docs/05` §3 — the gate on an upload, where the stages already exist
# --------------------------------------------------------------------------- #

def test_assess_uses_the_stages_it_is_given_instead_of_recomputing(gate, monkeypatch):
    """One call, one pipeline run.

    `/api/predict` runs `preprocess_stages` to produce its answer, and §3 forbids running
    it a second time to measure the same image. The gate takes what the caller already
    has; monkeypatching the module-level name is what proves it is not called again.
    """
    import quality as quality_module

    image = chladni_like()
    stages = quality_module.preprocess_stages(image, CALIB_CENTER_CROP, None)

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("preprocess_stages must not run a second time")

    monkeypatch.setattr(quality_module, "preprocess_stages", fail)
    result = assess(image, CALIB_CENTER_CROP, n_markers=4, gate=gate, stages=stages)

    assert result.quality == "ok"
    assert result.reason is None


def test_the_given_stages_produce_the_same_verdict_as_recomputing(gate):
    """The parameter is a shortcut, never a different measurement."""
    image = chladni_like()
    from quality import preprocess_stages as ps

    recomputed = assess(image, CALIB_CENTER_CROP, n_markers=4, gate=gate)
    handed = assess(
        image,
        CALIB_CENTER_CROP,
        n_markers=4,
        gate=gate,
        stages=ps(image, CALIB_CENTER_CROP, None),
    )
    assert handed.quality == recomputed.quality
    assert handed.reason == recomputed.reason
    assert handed.metrics == recomputed.metrics
