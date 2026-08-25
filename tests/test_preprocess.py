"""Contract tests for `src/preprocess.py` — docs/03-preprocessing-spec.md §2.

Covers: the shape and dtype of every stage, determinism, both `roi_mode` values,
and the fewer-than-four-markers failure path.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from preprocess import (
    CalibrationError,
    load_config,
    preprocess,
    preprocess_stages,
)

STAGE_NAMES = (
    "source",
    "roi",
    "gray",
    "clahe",
    "blur",
    "edges",
    "resized",
    "tensor",
)


# --- output contract (step 8) ------------------------------------------------


def test_preprocess_returns_float32_tensor_of_the_configured_shape(
    plate_photo, calib_homography, config
):
    tensor = preprocess(plate_photo, calib_homography, config)

    assert tensor.shape == tuple(config["output"]["shape"])
    assert tensor.dtype == np.float32


def test_preprocess_output_is_bounded_to_the_unit_interval(
    plate_photo, calib_homography, config
):
    tensor = preprocess(plate_photo, calib_homography, config)

    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_preprocess_output_is_the_last_stage(plate_photo, calib_homography, config):
    tensor = preprocess(plate_photo, calib_homography, config)
    stages = preprocess_stages(plate_photo, calib_homography, config)

    assert np.array_equal(tensor, stages.tensor)


# --- stage contract (steps 1 to 8) -------------------------------------------


def test_preprocess_stages_returns_eight_arrays(plate_photo, calib_homography, config):
    stages = preprocess_stages(plate_photo, calib_homography, config)

    assert len(stages) == 8
    assert stages._fields == STAGE_NAMES


def test_source_stage_is_the_input_view_not_a_defensive_copy(
    plate_photo, calib_homography, config
):
    """Step 1's output is handed back as-is; the caller must not modify it."""
    stages = preprocess_stages(plate_photo, calib_homography, config)

    assert stages.source is plate_photo


def test_roi_stage_is_the_rgba_warp_that_enters_the_grayscale_step(
    plate_photo, calib_homography, config
):
    """`roi` and `source` must be the arrays the pipeline actually used, not a
    re-run: converting `roi` reproduces `gray` exactly."""
    stages = preprocess_stages(plate_photo, calib_homography, config)

    size = config["roi"]["output_size"]
    assert stages.roi.shape == (size, size, 4)
    assert stages.roi.dtype == np.uint8
    assert np.array_equal(
        cv2.cvtColor(stages.roi, getattr(cv2, config["grayscale"]["code"])), stages.gray
    )


@pytest.mark.parametrize("name", ["gray", "clahe", "blur", "edges"])
def test_roi_sized_stages_are_uint8_squares_of_the_configured_roi_size(
    name, plate_photo, calib_homography, config
):
    stages = preprocess_stages(plate_photo, calib_homography, config)
    stage = getattr(stages, name)

    size = config["roi"]["output_size"]
    assert stage.shape == (size, size)
    assert stage.dtype == np.uint8


def test_resize_stage_is_uint8_of_the_configured_size(
    plate_photo, calib_homography, config
):
    stages = preprocess_stages(plate_photo, calib_homography, config)

    width, height = config["resize"]["size"]
    assert stages.resized.shape == (height, width)
    assert stages.resized.dtype == np.uint8


def test_canny_stage_is_binary(plate_photo, calib_homography, config):
    stages = preprocess_stages(plate_photo, calib_homography, config)

    assert set(np.unique(stages.edges)).issubset({0, 255})


def test_canny_stage_finds_the_nodal_lines(plate_photo, calib_homography, config):
    stages = preprocess_stages(plate_photo, calib_homography, config)

    assert (stages.edges == 255).any()


def test_resize_stage_is_antialiased_greyscale_not_binary(
    plate_photo, calib_homography, config
):
    """INTER_AREA on a binary image must yield intermediate levels — spec §2.7."""
    stages = preprocess_stages(plate_photo, calib_homography, config)

    intermediate = (stages.resized > 0) & (stages.resized < 255)
    assert intermediate.any()


def test_clahe_stage_passes_through_when_disabled(
    plate_photo, calib_homography, config
):
    config["clahe"]["enabled"] = False
    stages = preprocess_stages(plate_photo, calib_homography, config)

    assert np.array_equal(stages.clahe, stages.gray)


def test_blur_stage_smooths_the_image(plate_photo, calib_homography, config):
    """Step 5 is the only smoothing in the pipeline — Canny adds none (spec §2.5)."""
    stages = preprocess_stages(plate_photo, calib_homography, config)

    assert not np.array_equal(stages.blur, stages.clahe)
    assert cv2.Laplacian(stages.blur, cv2.CV_64F).var() < cv2.Laplacian(
        stages.clahe, cv2.CV_64F
    ).var()


def test_clahe_stage_changes_the_image_when_enabled(
    plate_photo, calib_homography, config
):
    stages = preprocess_stages(plate_photo, calib_homography, config)

    assert not np.array_equal(stages.clahe, stages.gray)


# --- every parameter comes from the config -----------------------------------


def test_roi_size_is_driven_by_the_config(plate_photo, calib_homography, config):
    config["roi"]["output_size"] = 256
    stages = preprocess_stages(plate_photo, calib_homography, config)

    assert stages.gray.shape == (256, 256)


def test_output_size_is_driven_by_the_config(plate_photo, calib_homography, config):
    config["resize"]["size"] = [64, 64]
    config["output"]["shape"] = [64, 64, 1]
    tensor = preprocess(plate_photo, calib_homography, config)

    assert tensor.shape == (64, 64, 1)


def test_output_dtype_is_driven_by_the_config(plate_photo, calib_homography, config):
    config["output"]["dtype"] = "float64"

    assert preprocess(plate_photo, calib_homography, config).dtype == np.float64


def test_scale_is_driven_by_the_config(plate_photo, calib_homography, config):
    config["output"]["scale"] = 1.0
    stages = preprocess_stages(plate_photo, calib_homography, config)

    assert np.array_equal(stages.tensor[..., 0], stages.resized.astype(np.float32))


#: (config section, key, a value differing from the committed one, affected stage).
#: Every tunable of §2 steps 3 to 7 — none of them may be hardcoded in the module.
TUNABLES = [
    ("roi", "interpolation", "INTER_NEAREST", "roi"),
    ("grayscale", "code", "COLOR_BGRA2GRAY", "gray"),
    ("clahe", "clip_limit", 40.0, "clahe"),
    ("clahe", "tile_grid_size", [2, 2], "clahe"),
    ("blur", "ksize", [11, 11], "blur"),
    ("blur", "sigma_x", 4.0, "blur"),
    ("blur", "sigma_y", 4.0, "blur"),
    ("blur", "border", "BORDER_CONSTANT", "blur"),
    ("canny", "threshold1", 200, "edges"),
    ("canny", "threshold2", 250, "edges"),
    ("canny", "aperture_size", 5, "edges"),
    ("canny", "l2_gradient", True, "edges"),
    ("resize", "interpolation", "INTER_NEAREST", "resized"),
]


@pytest.mark.parametrize(
    ("section", "key", "value", "stage"),
    TUNABLES,
    ids=[f"{section}.{key}" for section, key, _, stage in TUNABLES],
)
def test_every_tunable_is_read_from_the_config(
    section, key, value, stage, plate_photo, calib_homography, config
):
    reference = getattr(preprocess_stages(plate_photo, calib_homography, config), stage)
    config[section][key] = value

    mutated = getattr(preprocess_stages(plate_photo, calib_homography, config), stage)
    assert not np.array_equal(mutated, reference)


def test_canny_thresholds_are_driven_by_the_config(plate_photo, calib_homography, config):
    reference = preprocess_stages(plate_photo, calib_homography, config).edges
    config["canny"]["threshold1"] = 200
    config["canny"]["threshold2"] = 250

    edges = preprocess_stages(plate_photo, calib_homography, config).edges
    assert not np.array_equal(edges, reference)
    assert (edges == 255).sum() < (reference == 255).sum()


def test_unknown_opencv_constant_in_the_config_raises(plate_photo, calib_homography, config):
    config["resize"]["interpolation"] = "INTER_MAGIC"

    with pytest.raises(ValueError, match="INTER_MAGIC"):
        preprocess(plate_photo, calib_homography, config)


#: A quad deliberately hanging off the top-left of `plate_photo` (800x600), so the warp
#: has to sample outside the frame and the border mode is observable at all. With a quad
#: entirely inside the image every border mode gives the same pixels.
_OVERHANGING_QUAD = [[-120.0, -90.0], [520.0, 70.0], [560.0, 430.0], [90.0, 470.0]]


def test_roi_border_mode_is_read_from_the_config(plate_photo, config):
    calib = {"roi_mode": "homography", "corners": _OVERHANGING_QUAD}
    replicate = preprocess_stages(plate_photo, calib, config).roi

    config["roi"]["border_mode"] = "BORDER_CONSTANT"
    constant = preprocess_stages(plate_photo, calib, config).roi

    # BORDER_CONSTANT fills the overhang with zeros; BORDER_REPLICATE extends the edge.
    assert not np.array_equal(constant, replicate)
    assert constant[0, 0].max() == 0


def test_unknown_roi_border_mode_in_the_config_raises(plate_photo, calib_homography, config):
    config["roi"]["border_mode"] = "BORDER_MAGIC"

    with pytest.raises(ValueError, match="BORDER_MAGIC"):
        preprocess(plate_photo, calib_homography, config)


def test_load_config_reads_the_committed_config(repo_root):
    expected = json.loads(
        (repo_root / "config" / "preprocess.json").read_text(encoding="utf-8")
    )

    assert load_config() == expected


def test_config_is_optional_and_defaults_to_the_committed_one(
    plate_photo, calib_homography
):
    assert np.array_equal(
        preprocess(plate_photo, calib_homography),
        preprocess(plate_photo, calib_homography, load_config()),
    )


# --- determinism (hard rule 5) -----------------------------------------------


def test_preprocess_is_deterministic_across_ten_runs(
    plate_photo, calib_homography, config
):
    reference = preprocess(plate_photo, calib_homography, config).tobytes()

    for _ in range(10):
        assert preprocess(plate_photo, calib_homography, config).tobytes() == reference


def test_preprocess_stages_are_deterministic_across_ten_runs(
    plate_photo, calib_center_crop, config
):
    reference = [s.tobytes() for s in preprocess_stages(plate_photo, calib_center_crop, config)]

    for _ in range(10):
        stages = preprocess_stages(plate_photo, calib_center_crop, config)
        assert [s.tobytes() for s in stages] == reference


def test_preprocess_does_not_mutate_its_input(plate_photo, calib_homography, config):
    before = plate_photo.copy()
    preprocess(plate_photo, calib_homography, config)

    assert np.array_equal(plate_photo, before)


# --- roi_mode: homography ----------------------------------------------------


def test_homography_warps_the_marked_quad_onto_the_whole_roi(
    white_quad_photo, calib_homography, config
):
    stages = preprocess_stages(white_quad_photo, calib_homography, config)

    interior = stages.gray[4:-4, 4:-4]
    assert interior.min() == 255


def test_homography_interpolates_linearly(white_quad_photo, calib_homography, config):
    """Spec §2 step 2 fixes INTER_LINEAR: a skewed quad must produce intermediate
    levels along its border, which a nearest-neighbour warp cannot."""
    stages = preprocess_stages(white_quad_photo, calib_homography, config)

    assert ((stages.gray > 0) & (stages.gray < 255)).any()


def test_homography_replicates_the_border_outside_the_frame(config):
    """Spec §2 step 2 fixes BORDER_REPLICATE: a quad reaching past the frame must be
    filled from the edge pixels, not with a constant."""
    frame = np.full((600, 800, 4), 255, dtype=np.uint8)
    corners = [[-60, -60], [860, -60], [860, 660], [-60, 660]]

    stages = preprocess_stages(frame, {"roi_mode": "homography", "corners": corners}, config)

    assert stages.gray.min() == 255


def test_homography_uses_the_calibrated_corners_not_the_frame(
    white_quad_photo, calib_homography, calib_center_crop, config
):
    warped = preprocess_stages(white_quad_photo, calib_homography, config).gray
    cropped = preprocess_stages(white_quad_photo, calib_center_crop, config).gray

    assert (cropped == 0).any()
    assert not np.array_equal(warped, cropped)


@pytest.mark.parametrize(
    "calib",
    [
        {"roi_mode": "homography", "corners": [[0, 0], [511, 0], [511, 511], [0, 511]]},
        {"roi_mode": "center_crop"},
    ],
)
def test_roi_of_a_frame_already_at_roi_size_is_an_identity_warp(calib, config):
    """Pins the destination quad to `[0, size-1]`, not `[0, size]`. The browser side
    must use the same convention or the parity gate drifts by a sub-pixel shift."""
    config["roi"]["output_size"] = 512
    rng = np.random.default_rng(42)
    frame = np.dstack(
        [
            rng.integers(0, 256, (512, 512, 3), dtype=np.uint8),
            np.full((512, 512, 1), 255, np.uint8),
        ]
    )

    stages = preprocess_stages(frame, calib, config)

    assert np.array_equal(stages.gray, cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY))


def test_homography_on_the_frame_corners_equals_a_full_frame_warp(config):
    """Corner ordering and the destination quad must be the same in both ROI modes."""
    square = np.zeros((400, 400, 4), dtype=np.uint8)
    square[..., 3] = 255
    square[100:300, 120:260] = 255

    corners = [[0, 0], [399, 0], [399, 399], [0, 399]]
    warped = preprocess_stages(square, {"roi_mode": "homography", "corners": corners}, config)
    cropped = preprocess_stages(square, {"roi_mode": "center_crop"}, config)

    assert np.array_equal(warped.gray, cropped.gray)


# --- roi_mode: center_crop ---------------------------------------------------


def test_center_crop_keeps_the_largest_centred_square(config):
    """The crop must be centred: a mark outside the centred square is dropped."""
    frame = np.zeros((400, 800, 4), dtype=np.uint8)
    frame[..., 3] = 255
    frame[:, :180] = 255  # left band, outside the centred 400x400 square

    stages = preprocess_stages(frame, {"roi_mode": "center_crop"}, config)

    assert stages.gray.max() == 0


def test_center_crop_needs_no_calibration(plate_photo, config):
    config["roi"]["mode"] = "center_crop"
    tensor = preprocess(plate_photo, None, config)

    assert tensor.shape == tuple(config["output"]["shape"])


def test_calib_roi_mode_overrides_the_config_default(plate_photo, config):
    config["roi"]["mode"] = "homography"
    tensor = preprocess(plate_photo, {"roi_mode": "center_crop"}, config)

    assert tensor.shape == tuple(config["output"]["shape"])


def test_both_roi_modes_are_supported_and_differ(
    plate_photo, calib_homography, calib_center_crop, config
):
    homography = preprocess(plate_photo, calib_homography, config)
    center_crop = preprocess(plate_photo, calib_center_crop, config)

    assert homography.shape == center_crop.shape
    assert not np.array_equal(homography, center_crop)


# --- failure paths -----------------------------------------------------------


@pytest.mark.parametrize("found", [0, 1, 2, 3])
def test_fewer_than_four_markers_raises_calibration_error(
    found, plate_photo, calib_homography, config
):
    calib_homography["corners"] = calib_homography["corners"][:found]

    with pytest.raises(CalibrationError, match="four"):
        preprocess(plate_photo, calib_homography, config)


def test_fewer_than_four_markers_never_falls_back_silently(
    plate_photo, calib_homography, config
):
    """A partial calibration must not be downgraded to center_crop: roi_mode has to
    stay auditable, so the caller decides (docs/02 §6 marks the image `partial`)."""
    calib_homography["corners"] = calib_homography["corners"][:3]

    with pytest.raises(CalibrationError):
        preprocess_stages(plate_photo, calib_homography, config)


def test_missing_calibration_in_homography_mode_raises(plate_photo, config):
    with pytest.raises(CalibrationError):
        preprocess(plate_photo, None, config)


def test_calibration_without_corners_raises(plate_photo, config):
    with pytest.raises(CalibrationError):
        preprocess(plate_photo, {"session_id": "20260817-01"}, config)


def test_degenerate_corners_raise_calibration_error(plate_photo, calib_homography, config):
    calib_homography["corners"] = [[10, 10]] * 4

    with pytest.raises(CalibrationError):
        preprocess(plate_photo, calib_homography, config)


def test_unknown_roi_mode_raises(plate_photo, config):
    with pytest.raises(ValueError, match="roi_mode"):
        preprocess(plate_photo, {"roi_mode": "magic"}, config)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((600, 800, 3), dtype=np.uint8),
        np.zeros((600, 800), dtype=np.uint8),
        np.zeros((600, 800, 4), dtype=np.float32),
    ],
)
def test_non_rgba_uint8_input_raises(bad, calib_center_crop, config):
    with pytest.raises(ValueError):
        preprocess(bad, calib_center_crop, config)
