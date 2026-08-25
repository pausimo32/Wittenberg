"""Contract tests for `src/calibrate.py` — ArUco calibration at ingest.

Two rules carry the weight here.

**The innermost corner.** Each marker is taped at a plate corner, so of its four
corners exactly one touches the plate. That is the one the homography must use. Taking
the marker centre instead would pull half a marker of rig into the ROI on every side;
taking the outermost corner would pull a whole one. The tests below assert the chosen
point *is* the inner corner and explicitly *is not* the centre or the outer corner, so
a regression to either is caught by name.

**No silent fallback.** Fewer than four markers is a `partial` rejection
(`docs/02` §6). A calibration that quietly degraded to `center_crop` would produce a
plausible-looking image from an uncalibrated frame, and nothing downstream would know.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from calibrate import (
    Calibration,
    calibrate,
    draw_marker,
    load_calibration_config,
    read_calibration,
    write_calibration,
)
from preprocess import preprocess_stages

BOARD = 400
MARKER = 80
#: Where each marker's top-left pixel goes, by id: 0 TL, 1 TR, 2 BR, 3 BL.
PLACEMENT = {0: (20, 20), 1: (300, 20), 2: (300, 300), 3: (20, 300)}

#: The corner of each marker that touches the plate, given PLACEMENT above.
INNER = {0: (99.0, 99.0), 1: (300.0, 99.0), 2: (300.0, 300.0), 3: (99.0, 300.0)}
#: The opposite corner, and the centre — the two plausible wrong answers.
OUTER = {0: (20.0, 20.0), 1: (379.0, 20.0), 2: (379.0, 379.0), 3: (20.0, 379.0)}


@pytest.fixture
def cfg() -> dict:
    return load_calibration_config()


def board(ids=(0, 1, 2, 3), size: int = BOARD) -> np.ndarray:
    """A synthetic calibration frame: white plate, one ArUco marker per corner."""
    canvas = np.full((size, size), 255, np.uint8)
    for i in ids:
        x, y = PLACEMENT[i]
        canvas[y: y + MARKER, x: x + MARKER] = draw_marker(i, MARKER)
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGBA)


# --------------------------------------------------------------------------- #
# detection and ordering
# --------------------------------------------------------------------------- #

def test_four_markers_are_found(cfg):
    c = calibrate(board(), session_id="SYNTH-20260101-01", cfg=cfg)
    assert isinstance(c, Calibration)
    assert c.n_markers == 4
    assert c.corners is not None


def test_corners_are_ordered_tl_tr_br_bl(cfg):
    """`preprocess._calibrated_quad` requires this order and will not re-derive it."""
    c = calibrate(board(), session_id="SYNTH-20260101-01", cfg=cfg)
    expected = [INNER[0], INNER[1], INNER[2], INNER[3]]
    assert np.allclose(np.array(c.corners), np.array(expected), atol=1.5)


def test_ordering_follows_marker_id_not_geometry(cfg):
    """Ids are the contract: the markers are taped by id, so ingest orders by id."""
    c = calibrate(board(), session_id="SYNTH-20260101-01", cfg=cfg)
    ids = cfg["marker_ids"]
    assert [ids["top_left"], ids["top_right"], ids["bottom_right"], ids["bottom_left"]] == [0, 1, 2, 3]
    assert np.allclose(c.corners[0], INNER[0], atol=1.5)


# --------------------------------------------------------------------------- #
# the innermost-corner rule
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("marker_id", sorted(PLACEMENT))
def test_chosen_corner_is_the_one_touching_the_plate(cfg, marker_id):
    c = calibrate(board(), session_id="SYNTH-20260101-01", cfg=cfg)
    got = np.array(c.corners[marker_id])
    assert np.allclose(got, INNER[marker_id], atol=1.5)


@pytest.mark.parametrize("marker_id", sorted(PLACEMENT))
def test_chosen_corner_is_not_the_marker_centre(cfg, marker_id):
    c = calibrate(board(), session_id="SYNTH-20260101-01", cfg=cfg)
    x, y = PLACEMENT[marker_id]
    centre = np.array([x + MARKER / 2, y + MARKER / 2])
    assert not np.allclose(np.array(c.corners[marker_id]), centre, atol=5.0)


@pytest.mark.parametrize("marker_id", sorted(PLACEMENT))
def test_chosen_corner_is_not_the_outer_corner(cfg, marker_id):
    c = calibrate(board(), session_id="SYNTH-20260101-01", cfg=cfg)
    assert not np.allclose(np.array(c.corners[marker_id]), OUTER[marker_id], atol=5.0)


def test_quad_is_the_plate_not_the_marker_envelope(cfg):
    """The calibrated square spans the gap between markers, not their outer bounds."""
    c = calibrate(board(), session_id="SYNTH-20260101-01", cfg=cfg)
    quad = np.array(c.corners, np.float32)
    side = np.linalg.norm(quad[1] - quad[0])
    assert side == pytest.approx(300 - 99, abs=2.0)      # inner span
    assert side < 379 - 20                                # not the outer envelope


# --------------------------------------------------------------------------- #
# missing markers — partial, never a silent center_crop
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("present", [(), (0,), (0, 1), (0, 1, 2)])
def test_fewer_than_four_markers_yields_no_quad(cfg, present):
    c = calibrate(board(ids=present), session_id="SYNTH-20260101-01", cfg=cfg)
    assert c.n_markers == len(present)
    assert c.corners is None


def test_missing_markers_never_downgrade_the_roi_mode(cfg):
    c = calibrate(board(ids=(0, 1, 2)), session_id="SYNTH-20260101-01", cfg=cfg)
    assert c.roi_mode == "homography"
    assert c.corners is None


def test_center_crop_session_needs_no_markers(cfg):
    c = calibrate(board(ids=()), session_id="SYNTH-20260101-01", cfg=cfg, roi_mode="center_crop")
    assert c.roi_mode == "center_crop"
    assert c.n_markers == 0


# --------------------------------------------------------------------------- #
# scale
# --------------------------------------------------------------------------- #

def test_px_per_mm_is_derived_from_the_plate_size(cfg):
    c = calibrate(board(), session_id="SYNTH-20260101-01", cfg=cfg)
    assert c.px_per_mm == pytest.approx((300 - 99) / cfg["plate_size_mm"], rel=0.02)


# --------------------------------------------------------------------------- #
# the file, and its consumer
# --------------------------------------------------------------------------- #

def test_round_trips_through_the_calib_file(cfg, tmp_path):
    c = calibrate(board(), session_id="SYNTH-20260101-01", cfg=cfg)
    path = write_calibration(c, tmp_path)
    assert path.name == "SYNTH-20260101-01.json"

    back = read_calibration(path)
    assert back["session_id"] == "SYNTH-20260101-01"
    assert back["roi_mode"] == "homography"
    assert np.allclose(np.array(back["corners"]), np.array(c.corners))
    assert back["px_per_mm"] == pytest.approx(c.px_per_mm)


def test_written_calibration_records_the_corner_rule(cfg, tmp_path):
    """Auditability: the file says which convention produced these four points."""
    c = calibrate(board(), session_id="SYNTH-20260101-01", cfg=cfg)
    back = json.loads(write_calibration(c, tmp_path).read_text("utf-8"))
    assert back["corner_rule"] == "innermost"
    assert back["dictionary"] == cfg["dictionary"]


def test_preprocess_accepts_what_calibrate_produces(cfg, tmp_path):
    """The contract that matters: `src/preprocess.py` consumes this dict untouched."""
    c = calibrate(board(), session_id="SYNTH-20260101-01", cfg=cfg)
    calib = read_calibration(write_calibration(c, tmp_path))
    stages = preprocess_stages(board(), calib)
    assert stages.roi.shape == (512, 512, 4)
    assert stages.tensor.shape == (128, 128, 1)


# --------------------------------------------------------------------------- #
# the nominal table — an audit of the rig, not the selection rule
# --------------------------------------------------------------------------- #

def test_a_nominal_board_agrees_with_the_table(cfg):
    """`docs/02` §1.1 tabulates what the innermost rule evaluates to on the real rig.

    On a correctly taped board the geometric answer and the table are the same answer,
    so there is nothing to report.
    """
    c = calibrate(board(), session_id="SYNTH-20260101-01", cfg=cfg)
    assert c.nominal_mismatch == []
    assert c.flags == []


def test_a_marker_taped_on_rotated_still_calibrates_correctly(cfg):
    """This is why the geometry decides and the table only audits.

    Rotating a marker 180° moves its touching corner to a different index of its own
    quad. Selecting by the tabulated index would silently take the wrong point and warp
    a plate that still looks plausible; selecting by proximity to the centroid still
    lands on the corner that physically touches the plate.
    """
    c = calibrate(board_with_rotated_marker(0), session_id="SYNTH-20260101-01", cfg=cfg)
    assert c.corners is not None
    assert tuple(c.corners[0]) == pytest.approx(INNER[0], abs=1.5)


def test_a_marker_taped_on_rotated_is_reported(cfg):
    """...and it is still worth hearing about on the day, not three weeks later."""
    c = calibrate(board_with_rotated_marker(0), session_id="SYNTH-20260101-01", cfg=cfg)
    assert c.nominal_mismatch == [0]
    assert len(c.flags) == 1
    assert "rotated" in c.flags[0]


def test_the_divergence_is_a_flag_and_never_an_error(cfg):
    """A plate deliberately turned within the frame produces one legitimately, and only
    the operator can tell that from a mis-taped marker."""
    c = calibrate(board_with_rotated_marker(2), session_id="SYNTH-20260101-01", cfg=cfg)
    assert c.usable
    assert c.px_per_mm is not None


def test_the_divergence_reaches_the_calib_file(tmp_path, cfg):
    """`src/ingest.py` reads it from there into `data/ingest_report.json`."""
    c = calibrate(board_with_rotated_marker(1), session_id="SYNTH-20260101-01", cfg=cfg)
    written = read_calibration(write_calibration(c, tmp_path))
    assert written["nominal_mismatch"] == [1]
    assert written["flags"]


def board_with_rotated_marker(marker_id: int, size: int = BOARD) -> np.ndarray:
    """The same board with one marker taped on rotated 180°."""
    canvas = np.full((size, size), 255, np.uint8)
    for i in (0, 1, 2, 3):
        x, y = PLACEMENT[i]
        tile = draw_marker(i, MARKER)
        if i == marker_id:
            tile = cv2.rotate(tile, cv2.ROTATE_180)
        canvas[y: y + MARKER, x: x + MARKER] = tile
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGBA)
