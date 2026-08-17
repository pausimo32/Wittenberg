"""Les indicateurs de qualité doivent orienter l'utilisateur, sans jamais bloquer."""

from __future__ import annotations

import numpy as np

from tests.support import images
from wittenberg.imaging.quality import assess_quality, edge_density, laplacian_variance


def gray(image_size: int = 256, **kwargs: int) -> np.ndarray:
    return np.asarray(images.chladni_like(image_size, **kwargs).convert("L"), dtype=np.uint8)


def test_sharp_pattern_scores_well() -> None:
    report = assess_quality(gray(), coverage=0.8, skew=0.05, plate_detected=True)

    assert report.score > 0.7
    assert not [w for w in report.warnings if w.code in {"blurry", "no_pattern_detected"}]


def test_blurred_image_is_flagged() -> None:
    from PIL import ImageFilter

    blurred = images.chladni_like(256).convert("L").filter(ImageFilter.GaussianBlur(radius=8))
    report = assess_quality(
        np.asarray(blurred, dtype=np.uint8), coverage=0.8, skew=0.0, plate_detected=True
    )

    assert any(w.code == "blurry" for w in report.warnings)


def test_uniform_image_reports_no_pattern() -> None:
    flat = np.full((128, 128), 30, dtype=np.uint8)
    report = assess_quality(flat, coverage=0.8, skew=0.0, plate_detected=True)

    codes = {w.code for w in report.warnings}
    assert "no_pattern_detected" in codes
    assert "low_contrast" in codes
    assert report.score < 0.5


def test_overexposed_image_is_flagged() -> None:
    bright = np.full((128, 128), 252, dtype=np.uint8)
    report = assess_quality(bright, coverage=0.8, skew=0.0, plate_detected=True)

    assert any(w.code == "overexposed" for w in report.warnings)


def test_underexposed_image_is_flagged() -> None:
    dark = np.zeros((128, 128), dtype=np.uint8)
    report = assess_quality(dark, coverage=0.8, skew=0.0, plate_detected=True)

    assert any(w.code == "underexposed" for w in report.warnings)


def test_low_coverage_is_flagged() -> None:
    report = assess_quality(gray(), coverage=0.05, skew=0.0, plate_detected=True)

    assert any(w.code == "low_coverage" for w in report.warnings)


def test_strong_perspective_is_flagged() -> None:
    report = assess_quality(gray(), coverage=0.8, skew=0.6, plate_detected=True)

    assert any(w.code == "perspective" for w in report.warnings)


def test_undetected_plate_is_informational_not_a_warning() -> None:
    report = assess_quality(gray(), coverage=0.8, skew=None, plate_detected=False)
    entry = next(w for w in report.warnings if w.code == "plate_not_detected")

    assert entry.severity == "info"


def test_score_is_always_within_bounds() -> None:
    for array in (gray(), np.zeros((64, 64), np.uint8), np.full((64, 64), 255, np.uint8)):
        for coverage in (0.0, 0.5, 1.0):
            report = assess_quality(array, coverage=coverage, skew=None, plate_detected=False)
            assert 0.0 <= report.score <= 1.0


def test_warnings_carry_actionable_french_messages() -> None:
    report = assess_quality(
        np.full((64, 64), 5, dtype=np.uint8), coverage=0.1, skew=0.9, plate_detected=True
    )

    assert report.warnings
    for warning in report.warnings:
        assert warning.message.endswith(".")
        assert len(warning.message) > 20


def test_laplacian_variance_ranks_sharpness() -> None:
    from PIL import ImageFilter

    sharp = gray()
    blurred = np.asarray(
        images.chladni_like(256).convert("L").filter(ImageFilter.GaussianBlur(radius=6)),
        dtype=np.uint8,
    )

    assert laplacian_variance(sharp) > laplacian_variance(blurred)


def test_edge_density_is_zero_on_flat_image() -> None:
    assert edge_density(np.full((64, 64), 100, dtype=np.uint8)) == 0.0


def test_metrics_are_exposed() -> None:
    report = assess_quality(gray(), coverage=0.8, skew=0.1, plate_detected=True)

    assert {"sharpness", "contrast", "edge_density", "coverage", "skew"} <= set(report.metrics)
