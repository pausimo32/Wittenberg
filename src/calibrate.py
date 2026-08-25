"""ArUco calibration — turns a capture into the four points `preprocess` warps from.

`src/preprocess.py` states in its docstring that `corners` arrive "already ordered by
ingest (the markers carry ids, so ordering is ingest's job, not a geometric guess made
here)". This module is that job.

**Which corner.** Four markers are taped at the plate corners, so each marker has
exactly one corner touching the plate. That corner is the calibration point: of the
four, the one nearest the centroid of all markers. The two tempting alternatives are
both wrong by a knowable amount — at the nominal 210 mm working distance (`docs/02`
§1, ~14.5 px/mm) a 25 mm marker is ~360 px, so using the marker *centre* drags ~12 mm
of rig into the ROI on every side and the *outer* corner drags ~25 mm. Either shrinks
the plate in the warped image and shifts every nodal line the model is asked to read.

**The table in `docs/02` §1.1 audits that choice; it does not make it.** On the nominal
zenithal rig, marker `i`'s touching corner is at a fixed index into its `detectMarkers`
quad — `{0: 2, 1: 3, 2: 0, 3: 1}`, carried as `nominal_corner_index` in
`config/calibration.json`. Selecting by that index instead would be *less* robust:
a marker taped on rotated 90° still has a well-defined innermost corner, and its index
is no longer the tabulated one. So geometry decides and the table is compared against
the result. A divergence is reported — not raised — because a plate legitimately turned
within the frame produces one, and the operator is the only one who can tell that from
a mis-taped marker. What matters is hearing about it on the day, not three weeks later.

**Which marker is which.** By id, never by geometry: `config/calibration.json` maps
id -> plate corner. Sorting detected markers by position instead would silently
mis-order a frame shot at an angle, and a mis-ordered quad warps to a mirrored or
rotated plate that still looks perfectly plausible.

**Fewer than four markers is not a fallback.** `Calibration.corners` is `None` and
`roi_mode` stays as configured; the caller rejects the frame as `partial` per
`docs/02` §6. Degrading to `center_crop` here would hide an uncalibrated capture
inside an otherwise clean dataset.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

import cv2
import numpy as np

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "calibration.json"

ROI_MODE_HOMOGRAPHY = "homography"
ROI_MODE_CENTER_CROP = "center_crop"

#: The order `preprocess._calibrated_quad` expects.
CORNER_ORDER = ("top_left", "top_right", "bottom_right", "bottom_left")


class Calibration(NamedTuple):
    """One session's geometry.

    `corners` is `None` when fewer than four markers were found — the caller must
    treat that as a `partial` rejection, not as a reason to change `roi_mode`.

    `nominal_mismatch` lists the marker ids whose geometrically selected corner is not
    the one `config/calibration.json`'s `nominal_corner_index` predicts. Empty is the
    normal case. Non-empty is a rig report, not an error — see the module docstring.
    """

    session_id: str
    roi_mode: str
    corners: list[list[float]] | None
    px_per_mm: float | None
    n_markers: int
    marker_ids: list[int]
    nominal_mismatch: list[int]

    @property
    def usable(self) -> bool:
        return self.roi_mode == ROI_MODE_CENTER_CROP or self.corners is not None

    @property
    def flags(self) -> list[str]:
        """Human-readable warnings for the session log and `data/ingest_report.json`."""
        if not self.nominal_mismatch:
            return []
        ids = ", ".join(str(i) for i in self.nominal_mismatch)
        message = (
            f"marker(s) {ids}: the corner touching the plate is not the one the nominal "
            f"zenithal rig produces (docs/02 §1.1). The calibration used the geometric "
            f"corner and is correct either way, but this means a marker is taped on "
            f"rotated, or the plate is turned within the frame. Check the rig before the "
            f"next session; if the plate is deliberately turned, say so in session.json."
        )
        return [message]


def load_calibration_config(path: Path | str | None = None) -> dict[str, Any]:
    return json.loads(Path(CONFIG_PATH if path is None else path).read_text("utf-8"))


def draw_marker(marker_id: int, size: int, cfg: Mapping[str, Any] | None = None) -> np.ndarray:
    """Render one marker. Shared by the synthetic capture tool and the tests, so both
    speak the same dictionary as the detector."""
    params = cfg if cfg is not None else load_calibration_config()
    return cv2.aruco.generateImageMarker(_dictionary(params), int(marker_id), int(size))


def calibrate(
    image_rgba: np.ndarray,
    session_id: str,
    cfg: Mapping[str, Any] | None = None,
    roi_mode: str = ROI_MODE_HOMOGRAPHY,
) -> Calibration:
    """Detect the markers in one frame and build the session calibration."""
    params = cfg if cfg is not None else load_calibration_config()
    found = detect_markers(image_rgba, params)

    wanted = [int(params["marker_ids"][name]) for name in CORNER_ORDER]
    if not all(i in found for i in wanted):
        return Calibration(session_id, roi_mode, None, None, len(found), sorted(found), [])

    centroid = np.mean([found[i].mean(axis=0) for i in wanted], axis=0)
    chosen = [_innermost_index(found[i], centroid) for i in wanted]
    quad = [found[i][k] for i, k in zip(wanted, chosen)]

    return Calibration(
        session_id=session_id,
        roi_mode=roi_mode,
        corners=[[float(x), float(y)] for x, y in quad],
        px_per_mm=_px_per_mm(np.array(quad, np.float32), float(params["plate_size_mm"])),
        n_markers=len(found),
        marker_ids=sorted(found),
        nominal_mismatch=_nominal_mismatch(wanted, chosen, params),
    )


def detect_markers(
    image_rgba: np.ndarray, cfg: Mapping[str, Any] | None = None
) -> dict[int, np.ndarray]:
    """`{marker_id: (4, 2) corner array}` for every marker in the frame."""
    params = cfg if cfg is not None else load_calibration_config()
    gray = cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2GRAY) if image_rgba.ndim == 3 else image_rgba

    detector = cv2.aruco.ArucoDetector(_dictionary(params), cv2.aruco.DetectorParameters())
    corners, ids, _rejected = detector.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).astype(np.float64) for c, i in zip(corners, ids.ravel())}


# --------------------------------------------------------------------------- #
# the calib file — the schema `preprocess` documents
# --------------------------------------------------------------------------- #

def write_calibration(calib: Calibration, directory: Path | str) -> Path:
    """Write `data/calib/<session_id>.json`.

    The rig convention is written alongside the points, so a calibration can still be
    audited years later without knowing which version of the config produced it.
    """
    cfg = load_calibration_config()
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{calib.session_id}.json"

    path.write_text(
        json.dumps(
            {
                "session_id": calib.session_id,
                "roi_mode": calib.roi_mode,
                "corners": calib.corners,
                "px_per_mm": calib.px_per_mm,
                "n_markers": calib.n_markers,
                "marker_ids": calib.marker_ids,
                "dictionary": cfg["dictionary"],
                "corner_rule": cfg["corner_rule"],
                "nominal_mismatch": calib.nominal_mismatch,
                "flags": calib.flags,
                "calibration_config_version": cfg["version"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def read_calibration(path: Path | str) -> dict[str, Any]:
    """Read a calib file into the mapping `preprocess.preprocess` takes as `calib`."""
    return json.loads(Path(path).read_text("utf-8"))


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #

def _dictionary(cfg: Mapping[str, Any]):
    name = str(cfg["dictionary"])
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"unknown ArUco dictionary {name!r} in config/calibration.json")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def _innermost_index(marker: np.ndarray, centroid: np.ndarray) -> int:
    """Index of the corner of this marker that touches the plate.

    The plate lies between the markers, so its corner is the marker corner closest to
    the centre of the marker constellation. This holds under perspective too: the warp
    is projective, and projective maps preserve which corner is nearest the interior.
    """
    return int(np.argmin(np.linalg.norm(marker - centroid, axis=1)))


def _innermost_corner(marker: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """The corner itself. Kept as the module's stated rule in point form."""
    return marker[_innermost_index(marker, centroid)]


def _nominal_mismatch(
    wanted: list[int], chosen: list[int], cfg: Mapping[str, Any]
) -> list[int]:
    """Marker ids where geometry and `docs/02` §1.1's table disagree.

    The table describes the nominal rig, so a disagreement is a statement about the
    *mounting*, not about the calibration: the geometric corner is used regardless.
    """
    nominal = cfg.get("nominal_corner_index")
    if not nominal:
        return []
    return [
        marker_id
        for marker_id, index in zip(wanted, chosen)
        if str(marker_id) in nominal and int(nominal[str(marker_id)]) != index
    ]


def _px_per_mm(quad: np.ndarray, plate_size_mm: float) -> float:
    """Mean of the four sides, in pixels, over the plate's physical size.

    Averaging all four rather than taking one absorbs the small keystone left by a
    mount that is not perfectly zenithal.
    """
    sides = [float(np.linalg.norm(quad[(i + 1) % 4] - quad[i])) for i in range(4)]
    return float(np.mean(sides) / plate_size_mm)
