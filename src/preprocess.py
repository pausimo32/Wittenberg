"""Training-side preprocessing — the Python half of docs/03-preprocessing-spec.md.

Implements §2 steps 2 to 8. Step 1 (decode, EXIF orientation, Display P3 → sRGB, HEIC)
belongs to `src/imageio.py` per §5.1/§5.4 and is *not* done here: this module takes an
already-decoded 8-bit sRGB RGBA array.

The browser half is `web/lib/preprocess.ts`. Both read `config/preprocess.json`; this
file contains no transformation constant of its own. Any change here is a change to the
contract — see §6 of the spec before touching it.

The transform is a pure function: no file I/O, no globals, no logging, and the input
array is never mutated. `load_config` is the single I/O boundary and sits outside the
transform path; `preprocess`/`preprocess_stages` accept the config as an argument and
only fall back to it when the caller omits one.

Calibration (`data/calib/<session_id>.json`, see docs/02-dataset-spec.md §4)::

    {
      "session_id": "20260817-01",
      "roi_mode": "homography",             # optional; defaults to config roi.mode
      "corners": [[x, y], [x, y], [x, y], [x, y]],   # top-left, top-right,
                                                     # bottom-right, bottom-left
      "px_per_mm": 14.5                     # recorded by ingest, unused here
    }

`corners` are the four ArUco marker positions, already ordered by ingest (the markers
carry ids, so ordering is ingest's job, not a geometric guess made here). A calibration
with fewer than four of them raises `CalibrationError` rather than silently degrading to
`center_crop`: `roi_mode` has to stay auditable per §2, and docs/02 §6 requires such an
image to be rejected as `partial` by the caller.

For `roi_mode: "center_crop"` no calibration data is needed — pass
`{"roi_mode": "center_crop"}` or `None` when the config default is already `center_crop`.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any, NamedTuple

import cv2
import numpy as np

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "preprocess.json"

ROI_MODE_HOMOGRAPHY = "homography"
ROI_MODE_CENTER_CROP = "center_crop"

# Geometry, not tuning: a homography is defined by four point correspondences and an
# RGBA pixel has four channels. Every tunable value comes from the config.
_QUAD_CORNERS = 4
_RGBA_CHANNELS = 4


class CalibrationError(ValueError):
    """The calibration cannot define the requested ROI — e.g. fewer than four markers."""


class Stages(NamedTuple):
    """The array entering the pipeline and the one produced by each of §2 steps 2 to 8.

    Used by the parity gate, by the tests, when debugging a bad prediction, and by
    TDR annex 7. With a 512 px ROI and a 128x128 output:

    | field     | step | shape         | dtype   | range     |
    |-----------|------|---------------|---------|-----------|
    | `source`  | 1    | (h, w, 4)     | uint8   | 0-255     |
    | `roi`     | 2    | (512, 512, 4) | uint8   | 0-255     |
    | `gray`    | 3    | (512, 512)    | uint8   | 0-255     |
    | `clahe`   | 4    | (512, 512)    | uint8   | 0-255     |
    | `blur`    | 5    | (512, 512)    | uint8   | 0-255     |
    | `edges`   | 6    | (512, 512)    | uint8   | {0, 255}  |
    | `resized` | 7    | (128, 128)    | uint8   | 0-255     |
    | `tensor`  | 8    | (128, 128, 1) | float32 | [0, 1]    |

    `source` is the caller's own array, returned as a view rather than a copy: the
    transform never mutates it, and copying a full-resolution capture on every call
    would cost more than the stage is worth. **Whoever receives it must not modify
    it** — doing so mutates the caller's image, and any downstream re-run of the
    pipeline on the same array would then disagree with `roi` and the later stages.
    Copy it yourself if you need to draw on it (TDR figures, debug overlays).
    """

    source: np.ndarray
    roi: np.ndarray
    gray: np.ndarray
    clahe: np.ndarray
    blur: np.ndarray
    edges: np.ndarray
    resized: np.ndarray
    tensor: np.ndarray


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Read `config/preprocess.json`. The only I/O in this module, and never called
    from inside the transform when the caller supplies a config."""
    return copy.deepcopy(_read_config(Path(path) if path is not None else CONFIG_PATH))


@cache
def _read_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def preprocess(
    image_rgba: np.ndarray,
    calib: Mapping[str, Any] | None,
    config: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Run §2 steps 2 to 8 and return the model input tensor.

    `image_rgba` is 8-bit RGBA, `calib` is the session calibration (see module docstring).
    Returns a `float32` array of the configured output shape — `(128, 128, 1)`, in `[0, 1]`.
    """
    return preprocess_stages(image_rgba, calib, config).tensor


def preprocess_stages(
    image_rgba: np.ndarray,
    calib: Mapping[str, Any] | None,
    config: Mapping[str, Any] | None = None,
) -> Stages:
    """Same pipeline as `preprocess`, keeping its input and the array after each of
    steps 2 to 8. See `Stages` — `source` is a view of `image_rgba`, not a copy."""
    params = load_config() if config is None else config
    _validate_image(image_rgba)

    roi = _region_of_interest(image_rgba, calib, params)  # step 2
    gray = _grayscale(roi, params["grayscale"])  # step 3
    clahe = _clahe(gray, params["clahe"])  # step 4
    blur = _blur(clahe, params["blur"])  # step 5
    edges = _canny(blur, params["canny"])  # step 6
    resized = _resize(edges, params["resize"])  # step 7
    tensor = _scale(resized, params["output"])  # step 8

    return Stages(image_rgba, roi, gray, clahe, blur, edges, resized, tensor)


# --- step 2: ROI / geometric normalisation -----------------------------------


def _region_of_interest(
    image: np.ndarray, calib: Mapping[str, Any] | None, params: Mapping[str, Any]
) -> np.ndarray:
    size = params["roi"]["output_size"]
    mode = _roi_mode(calib, params)

    if mode == ROI_MODE_HOMOGRAPHY:
        source = _calibrated_quad(calib)
    elif mode == ROI_MODE_CENTER_CROP:
        source = _centred_square_quad(image)
    else:
        raise ValueError(f"unknown roi_mode: {mode!r}")

    # Both modes go through the same warp so the two ROI paths cannot drift apart.
    # Flags are fixed by spec §2 step 2 and deliberately absent from the config.
    matrix = cv2.getPerspectiveTransform(source, _destination_quad(size))
    return cv2.warpPerspective(
        image,
        matrix,
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _roi_mode(calib: Mapping[str, Any] | None, params: Mapping[str, Any]) -> str:
    """The calibration wins over the config default, so a dataset of mixed provenance
    can carry both modes at once."""
    if calib is not None and calib.get("roi_mode") is not None:
        return str(calib["roi_mode"])
    return str(params["roi"]["mode"])


def _calibrated_quad(calib: Mapping[str, Any] | None) -> np.ndarray:
    corners = None if calib is None else calib.get("corners")
    if corners is None:
        raise CalibrationError(
            "roi_mode 'homography' needs the four calibrated marker corners; "
            "the calibration has none"
        )

    quad = np.asarray(corners, dtype=np.float32)
    if quad.ndim != 2 or quad.shape != (_QUAD_CORNERS, 2):
        raise CalibrationError(
            f"roi_mode 'homography' needs four marker corners as (x, y) pairs, "
            f"got {quad.shape}"
        )
    if cv2.contourArea(quad) <= 0.0:
        raise CalibrationError(
            "the four calibrated marker corners are degenerate (zero area); "
            "they must form a convex quadrilateral ordered top-left, top-right, "
            "bottom-right, bottom-left"
        )
    return quad


def _centred_square_quad(image: np.ndarray) -> np.ndarray:
    """The largest centred square of the frame, as a source quad."""
    height, width = image.shape[:2]
    side = min(height, width)
    left = (width - side) // 2
    top = (height - side) // 2
    right = left + side - 1
    bottom = top + side - 1
    return np.array(
        [[left, top], [right, top], [right, bottom], [left, bottom]], dtype=np.float32
    )


def _destination_quad(size: int) -> np.ndarray:
    """The ROI square, corners in the same order as the calibrated ones."""
    last = size - 1
    return np.array(
        [[0.0, 0.0], [last, 0.0], [last, last], [0.0, last]], dtype=np.float32
    )


# --- steps 3 to 8 ------------------------------------------------------------


def _grayscale(image: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    return cv2.cvtColor(image, _cv_flag(params["code"]))


def _clahe(gray: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    if not params["enabled"]:
        return gray.copy()
    equaliser = cv2.createCLAHE(
        clipLimit=params["clip_limit"],
        tileGridSize=tuple(params["tile_grid_size"]),
    )
    return equaliser.apply(gray)


def _blur(gray: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    # The only smoothing in the pipeline — cv2.Canny does none of its own (spec §2.5).
    return cv2.GaussianBlur(
        gray,
        tuple(params["ksize"]),
        sigmaX=params["sigma_x"],
        sigmaY=params["sigma_y"],
        borderType=_cv_flag(params["border"]),
    )


def _canny(gray: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    return cv2.Canny(
        gray,
        params["threshold1"],
        params["threshold2"],
        apertureSize=params["aperture_size"],
        L2gradient=params["l2_gradient"],
    )


def _resize(edges: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    width, height = params["size"]
    # INTER_AREA anti-aliases the binary edges into greyscale on purpose (spec §2.7).
    return cv2.resize(
        edges, (width, height), interpolation=_cv_flag(params["interpolation"])
    )


def _scale(resized: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    # Last operation, on the full uint8 array, never fused into the resize (spec §5.6).
    dtype = np.dtype(params["dtype"])
    scaled = resized.astype(dtype) * params["scale"]
    return scaled.astype(dtype).reshape(tuple(params["shape"]))


# --- helpers -----------------------------------------------------------------


def _cv_flag(name: str) -> int:
    """Resolve an OpenCV enum by the name the config uses, so no flag is hardcoded."""
    flag = getattr(cv2, name, None)
    if not isinstance(flag, int):
        raise ValueError(f"unknown OpenCV constant: {name!r}")
    return flag


def _validate_image(image: Any) -> None:
    if not isinstance(image, np.ndarray):
        raise ValueError(f"expected an ndarray, got {type(image).__name__}")
    if image.dtype != np.uint8:
        raise ValueError(f"expected an 8-bit image, got dtype {image.dtype}")
    if image.ndim != 3 or image.shape[-1] != _RGBA_CHANNELS:
        raise ValueError(
            f"expected a decoded RGBA image (h, w, 4), got shape {image.shape}; "
            "decoding to sRGB RGBA is src/imageio.py's job (spec §2 step 1)"
        )
