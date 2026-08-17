"""Validation des fichiers reçus : contenu réel, jamais l'extension annoncée."""

from __future__ import annotations

import pytest
from PIL import Image

from tests.support import images
from wittenberg.imaging.validation import (
    ImageFormat,
    ImageValidationError,
    decode_image,
    sniff_format,
)

LIMITS = {
    "max_bytes": 10 * 1024 * 1024,
    "max_pixels": 40_000_000,
    "max_dimension": 12_000,
    "min_dimension": 32,
}


def decode(data: bytes, **overrides: int) -> object:
    return decode_image(data, **{**LIMITS, **overrides})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (images.encode(images.flat_image(64), "JPEG"), ImageFormat.JPEG),
        (images.encode(images.flat_image(64), "PNG"), ImageFormat.PNG),
        (images.encode(images.flat_image(64), "WEBP"), ImageFormat.WEBP),
        (images.encode(images.flat_image(64), "BMP"), ImageFormat.BMP),
        (images.encode(images.flat_image(64), "TIFF"), ImageFormat.TIFF),
        (b"GIF89a" + b"\x00" * 32, ImageFormat.GIF),
        (b"\x00\x00\x00\x20ftypheic" + b"\x00" * 16, ImageFormat.HEIF),
        (b"pas une image du tout, seulement du texte", ImageFormat.UNKNOWN),
    ],
)
def test_sniff_format(payload: bytes, expected: ImageFormat) -> None:
    assert sniff_format(payload) is expected


def test_valid_png_decodes() -> None:
    result = decode(images.encode(images.chladni_like(128), "PNG"))

    assert result.detected_format is ImageFormat.PNG  # type: ignore[attr-defined]
    assert result.image.mode == "RGB"  # type: ignore[attr-defined]
    assert result.image.size == (128, 128)  # type: ignore[attr-defined]


def test_valid_jpeg_decodes() -> None:
    result = decode(images.encode(images.chladni_like(128), "JPEG"))

    assert result.detected_format is ImageFormat.JPEG  # type: ignore[attr-defined]


def test_empty_file_is_rejected() -> None:
    with pytest.raises(ImageValidationError) as excinfo:
        decode(b"")

    assert excinfo.value.code == "empty_file"
    assert excinfo.value.status_code == 400


def test_text_file_with_image_extension_is_rejected() -> None:
    """L'extension ment : seul le contenu compte."""
    with pytest.raises(ImageValidationError) as excinfo:
        decode(b"Ceci est un fichier texte renomme en photo.jpg" * 4)

    assert excinfo.value.code == "unsupported_format"
    assert excinfo.value.status_code == 415


def test_truncated_png_is_rejected() -> None:
    payload = images.encode(images.chladni_like(128), "PNG")
    with pytest.raises(ImageValidationError) as excinfo:
        decode(payload[: len(payload) // 2])

    assert excinfo.value.code == "corrupt_image"


def test_corrupt_jpeg_body_is_rejected() -> None:
    """En-tête JPEG valide mais corps illisible."""
    payload = bytearray(images.encode(images.chladni_like(128), "JPEG"))
    for position in range(20, len(payload) - 4):
        payload[position] = 0
    with pytest.raises(ImageValidationError) as excinfo:
        decode(bytes(payload))

    assert excinfo.value.code == "corrupt_image"


def test_file_too_large_is_rejected() -> None:
    payload = images.encode(images.noise_image(256), "PNG")
    with pytest.raises(ImageValidationError) as excinfo:
        decode(payload, max_bytes=len(payload) - 1)

    assert excinfo.value.code == "file_too_large"
    assert excinfo.value.status_code == 413


def test_dimension_limit_is_enforced() -> None:
    payload = images.huge_declared_png(3000, 100)
    with pytest.raises(ImageValidationError) as excinfo:
        decode(payload, max_dimension=1000)

    assert excinfo.value.code == "image_too_large"
    assert excinfo.value.status_code == 413


def test_pixel_bomb_is_rejected_before_decoding() -> None:
    """Une image déclarant trop de pixels est refusée sur son en-tête."""
    payload = images.huge_declared_png(4000, 4000)
    with pytest.raises(ImageValidationError) as excinfo:
        decode(payload, max_pixels=1_000_000)

    assert excinfo.value.code == "image_too_many_pixels"
    assert excinfo.value.status_code == 413


def test_tiny_image_is_rejected() -> None:
    payload = images.encode(images.flat_image(8), "PNG")
    with pytest.raises(ImageValidationError) as excinfo:
        decode(payload)

    assert excinfo.value.code == "image_too_small"


def test_gif_returns_actionable_415() -> None:
    payload = images.encode(images.flat_image(64), "GIF")
    with pytest.raises(ImageValidationError) as excinfo:
        decode(payload)

    assert excinfo.value.status_code == 415
    assert "GIF" in excinfo.value.message


def test_heif_returns_actionable_415_when_unsupported() -> None:
    """Le cas du téléphone : message explicite, pas d'erreur générique."""
    payload = b"\x00\x00\x00\x20ftypheic" + b"\x00" * 64
    with pytest.raises(ImageValidationError) as excinfo:
        decode(payload)

    assert excinfo.value.status_code == 415
    assert excinfo.value.code == "unsupported_heif"
    assert "JPEG" in excinfo.value.message


def test_exif_orientation_is_applied() -> None:
    """Photo de téléphone en mode portrait : les dimensions sont redressées."""
    payload = images.jpeg_with_exif_orientation(orientation=6, size=200)
    result = decode(payload)

    assert result.exif_transposed is True  # type: ignore[attr-defined]
    # L'original est 200x100 ; l'orientation 6 le redresse en 100x200.
    assert result.image.size == (100, 200)  # type: ignore[attr-defined]


def test_rgba_is_composited_on_black() -> None:
    payload = images.encode(images.rgba_image(128), "PNG")
    result = decode(payload)

    assert result.had_alpha is True  # type: ignore[attr-defined]
    assert result.image.mode == "RGB"  # type: ignore[attr-defined]
    # La bande transparente du haut devient noire, jamais blanche.
    assert result.image.getpixel((64, 5)) == (0, 0, 0)  # type: ignore[attr-defined]


def test_grayscale_is_converted_to_rgb() -> None:
    payload = images.encode(images.grayscale_image(128), "PNG")
    result = decode(payload)

    assert result.original_mode == "L"  # type: ignore[attr-defined]
    assert result.image.mode == "RGB"  # type: ignore[attr-defined]


def test_cmyk_is_converted_to_rgb() -> None:
    payload = images.encode(images.cmyk_image(128), "JPEG")
    result = decode(payload)

    assert result.image.mode == "RGB"  # type: ignore[attr-defined]


def test_palette_with_transparency_is_handled() -> None:
    source = images.chladni_like(128).convert("P", palette=Image.Palette.ADAPTIVE)
    source.info["transparency"] = 0
    payload = images.encode(source, "PNG")
    result = decode(payload)

    assert result.image.mode == "RGB"  # type: ignore[attr-defined]


def test_metadata_is_stripped() -> None:
    """Aucune métadonnée EXIF ou GPS ne survit au décodage."""
    payload = images.jpeg_with_exif_orientation(orientation=3, size=200)
    result = decode(payload)

    assert not dict(result.image.getexif())  # type: ignore[attr-defined]
    assert "exif" not in result.image.info  # type: ignore[attr-defined]
