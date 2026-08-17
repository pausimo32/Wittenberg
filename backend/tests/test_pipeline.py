"""Le prétraitement doit être déterministe et produire le tenseur exact attendu."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.conftest import manifest_dict, write_manifest
from tests.support import images
from wittenberg.imaging.pipeline import (
    PreprocessingUnavailable,
    check_pipeline_requirements,
    preprocess_bytes,
    preprocess_image,
)
from wittenberg.imaging.plate import center_crop, opencv_available, order_corners
from wittenberg.model.manifest import ModelManifest, load_manifest

LIMITS = {
    "max_bytes": 10 * 1024 * 1024,
    "max_pixels": 40_000_000,
    "max_dimension": 12_000,
    "min_dimension": 32,
}


def build(tmp_path: Path, **overrides: Any) -> ModelManifest:
    return load_manifest(write_manifest(tmp_path, manifest_dict(**overrides)))


def test_tensor_shape_dtype_and_range(tmp_path: Path) -> None:
    manifest = build(tmp_path)
    result = preprocess_image(images.chladni_like(256), manifest)

    assert result.tensor.shape == (1, 32, 32, 1)
    assert result.tensor.dtype == np.float32
    assert result.tensor.min() >= 0.0
    assert result.tensor.max() <= 1.0


def test_pipeline_is_deterministic(tmp_path: Path) -> None:
    """Deux exécutions sur les mêmes octets donnent un tenseur identique au bit près."""
    manifest = build(tmp_path)
    payload = images.encode(images.chladni_like(256), "PNG")

    first, _ = preprocess_bytes(payload, manifest, **LIMITS)  # type: ignore[arg-type]
    second, _ = preprocess_bytes(payload, manifest, **LIMITS)  # type: ignore[arg-type]

    np.testing.assert_array_equal(first.tensor, second.tensor)
    assert first.steps == second.steps
    assert first.quality.score == second.quality.score


def test_nchw_layout(tmp_path: Path) -> None:
    manifest = build(tmp_path, input={"layout": "NCHW"})
    result = preprocess_image(images.chladni_like(256), manifest)

    assert result.tensor.shape == (1, 1, 32, 32)


def test_three_channel_input_is_replicated(tmp_path: Path) -> None:
    manifest = build(tmp_path, input={"channels": 3})
    result = preprocess_image(images.chladni_like(256), manifest)

    assert result.tensor.shape == (1, 32, 32, 3)
    # Le prétraitement passe en niveaux de gris : les 3 canaux sont identiques.
    np.testing.assert_array_equal(result.tensor[0, :, :, 0], result.tensor[0, :, :, 2])


def test_signed_normalization(tmp_path: Path) -> None:
    manifest = build(
        tmp_path,
        input={"value_range": [-1.0, 1.0], "normalization": {"mode": "scale_signed"}},
    )
    result = preprocess_image(images.chladni_like(256), manifest)

    assert result.tensor.min() >= -1.0
    assert result.tensor.max() <= 1.0
    assert result.tensor.min() < 0.0


def test_mean_std_normalization(tmp_path: Path) -> None:
    manifest = build(
        tmp_path,
        input={
            "value_range": [-5.0, 5.0],
            "normalization": {"mode": "mean_std", "mean": [0.5], "std": [0.25]},
        },
    )
    result = preprocess_image(images.chladni_like(256), manifest)

    assert result.tensor.dtype == np.float32
    assert result.tensor.min() >= -5.0


def test_raw_uint8_tensor(tmp_path: Path) -> None:
    manifest = build(
        tmp_path,
        input={
            "dtype": "uint8",
            "value_range": [0.0, 255.0],
            "normalization": {"mode": "raw"},
        },
    )
    result = preprocess_image(images.chladni_like(256), manifest)

    assert result.tensor.dtype == np.uint8
    assert result.tensor.max() > 1


def test_pad_preserves_aspect_ratio(tmp_path: Path) -> None:
    """Un motif rectangulaire ne doit pas être déformé en mode 'pad'."""
    manifest = build(
        tmp_path,
        preprocessing={"plate_detection": {"enabled": False, "fallback_center_crop": 1.0}},
    )
    wide = images.chladni_like(256).resize((256, 64))
    result = preprocess_image(wide, manifest)

    assert result.tensor.shape == (1, 32, 32, 1)
    assert "resize:pad(32, 32)" in result.steps


def test_stretch_mode(tmp_path: Path) -> None:
    manifest = build(tmp_path, preprocessing={"resize_mode": "stretch"})
    result = preprocess_image(images.chladni_like(256), manifest)

    assert any(step.startswith("resize:stretch") for step in result.steps)


def test_center_crop_fallback_when_plate_detection_disabled(tmp_path: Path) -> None:
    manifest = build(tmp_path, preprocessing={"plate_detection": {"enabled": False}})
    result = preprocess_image(images.chladni_like(256), manifest)

    assert result.plate.method == "center_crop"
    assert result.plate.detected is False
    assert any(w.code == "plate_not_detected" for w in result.quality.warnings)


@pytest.mark.skipif(not opencv_available(), reason="OpenCV absent")
def test_plate_is_detected_and_rectified(tmp_path: Path) -> None:
    """Une plaque claire nettement délimitée doit être localisée."""
    manifest = build(tmp_path)
    result = preprocess_image(images.chladni_like(400, plate_ratio=0.75), manifest)

    assert result.plate.detected is True
    assert result.plate.method == "homography"
    assert result.plate.skew is not None


@pytest.mark.skipif(not opencv_available(), reason="OpenCV absent")
def test_canny_produces_binary_edges(tmp_path: Path) -> None:
    manifest = build(
        tmp_path,
        preprocessing={
            "edge_detection": {"enabled": True, "low_threshold": 80, "high_threshold": 160}
        },
    )
    result = preprocess_image(images.chladni_like(256), manifest)

    assert any(step.startswith("canny:") for step in result.steps)
    unique = set(np.unique(result.tensor).tolist())
    assert unique.issubset({0.0, 1.0})


def test_canny_without_opencv_is_refused_at_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sans OpenCV, le service refuse de servir plutôt que de changer le tenseur."""
    manifest = build(tmp_path, preprocessing={"edge_detection": {"enabled": True}})
    monkeypatch.setattr("wittenberg.imaging.pipeline._import_cv2", lambda: None)

    with pytest.raises(PreprocessingUnavailable, match="OpenCV"):
        check_pipeline_requirements(manifest.preprocessing)


def test_requirements_pass_when_no_opencv_feature_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = build(tmp_path, preprocessing={"contrast": {"mode": "minmax"}})
    monkeypatch.setattr("wittenberg.imaging.pipeline._import_cv2", lambda: None)

    check_pipeline_requirements(manifest.preprocessing)


def test_denoise_median_runs(tmp_path: Path) -> None:
    manifest = build(tmp_path, preprocessing={"denoise": {"mode": "median", "kernel_size": 3}})
    result = preprocess_image(images.noise_image(256), manifest)

    assert "denoise:median(3)" in result.steps


def test_flat_image_skips_contrast_stretch(tmp_path: Path) -> None:
    """Étirer le contraste d'une image uniforme n'amplifierait que le bruit."""
    manifest = build(tmp_path, preprocessing={"plate_detection": {"enabled": False}})
    result = preprocess_image(images.flat_image(128, value=40), manifest)

    assert "contrast:minmax(skipped-flat)" in result.steps


def test_steps_are_reported_in_order(tmp_path: Path) -> None:
    manifest = build(tmp_path)
    result = preprocess_image(images.chladni_like(256), manifest)

    assert result.steps[0].startswith("plate:")
    assert "grayscale" in result.steps
    assert result.steps[-1].startswith("tensor:")


def test_debug_preview_is_absent_by_default(tmp_path: Path) -> None:
    manifest = build(tmp_path)

    assert preprocess_image(images.chladni_like(128), manifest).debug_preview is None
    assert preprocess_image(images.chladni_like(128), manifest, debug_preview=True).debug_preview


def test_exif_rotated_photo_yields_expected_tensor(tmp_path: Path) -> None:
    """Une photo de téléphone pivotée traverse la pipeline sans erreur."""
    manifest = build(tmp_path)
    payload = images.jpeg_with_exif_orientation(orientation=6, size=200)

    result, decoded = preprocess_bytes(payload, manifest, **LIMITS)  # type: ignore[arg-type]

    assert decoded.exif_transposed is True
    assert result.tensor.shape == (1, 32, 32, 1)


def test_center_crop_is_square_and_centred() -> None:
    source = np.zeros((100, 200, 3), dtype=np.uint8)
    source[40:60, 90:110] = 255
    cropped = center_crop(source, 0.5)

    assert cropped.shape == (50, 50, 3)
    assert cropped[25, 25].tolist() == [255, 255, 255]


def test_order_corners_normalises_orientation() -> None:
    quad = np.array([[10, 90], [90, 90], [90, 10], [10, 10]], dtype=np.float32)
    ordered = order_corners(quad)

    assert ordered[0].tolist() == [10, 10]
    assert ordered[2].tolist() == [90, 90]
