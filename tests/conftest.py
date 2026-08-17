"""Shared fixtures for the training-side test suite.

Puts `src/` on the import path so tests import `preprocess` directly, and provides
the synthetic RGBA inputs used by the preprocessing tests. No test reads a capture
from `data/` — the pipeline is pure, so synthetic input exercises it fully.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

SEED = 42


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def config() -> dict:
    """A fresh copy of the real config, so a test may mutate it safely."""
    raw = (REPO_ROOT / "config" / "preprocess.json").read_text(encoding="utf-8")
    return copy.deepcopy(json.loads(raw))


@pytest.fixture
def plate_photo() -> np.ndarray:
    """A synthetic 800x600 RGBA capture: noisy background, a bright skewed plate
    carrying a Chladni-like line pattern. Deterministic (seed 42)."""
    rng = np.random.default_rng(SEED)
    image = rng.integers(20, 60, size=(600, 800, 3), dtype=np.uint8)
    image = np.dstack([image, np.full((600, 800, 1), 255, np.uint8)])

    cv2.fillConvexPoly(image, PLATE_QUAD, (210, 210, 210, 255))
    for offset in range(-160, 200, 70):
        cv2.line(image, (260 + offset, 90), (300 + offset, 450), (25, 25, 25, 255), 6)
    for offset in range(-120, 160, 90):
        cv2.line(image, (110, 240 + offset), (560, 220 + offset), (25, 25, 25, 255), 6)
    return image


#: Corners of the plate inside `plate_photo`, ordered top-left, top-right,
#: bottom-right, bottom-left — the order `data/calib/<session_id>.json` stores.
PLATE_QUAD = np.array([[120, 90], [520, 70], [560, 430], [90, 470]], dtype=np.int32)


@pytest.fixture
def calib_homography() -> dict:
    return {
        "session_id": "20260817-01",
        "roi_mode": "homography",
        "corners": PLATE_QUAD.astype(float).tolist(),
        "px_per_mm": 14.5,
    }


@pytest.fixture
def calib_center_crop() -> dict:
    return {"session_id": "20260817-02", "roi_mode": "center_crop"}


@pytest.fixture
def white_quad_photo() -> np.ndarray:
    """Black RGBA frame with a solid white quadrilateral at `PLATE_QUAD`.

    Warping that quad to the ROI must fill the ROI with white; any other geometry
    leaves black in the frame. This is the geometric assertion for step 2.
    """
    image = np.zeros((600, 800, 4), dtype=np.uint8)
    image[..., 3] = 255
    cv2.fillConvexPoly(image, PLATE_QUAD, (255, 255, 255, 255))
    return image
