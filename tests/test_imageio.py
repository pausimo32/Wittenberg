"""Contract tests for `src/imageio.py` — step 1 of the pipeline.

Decode, apply and erase EXIF orientation, convert Display P3 to sRGB, drop every
embedded profile. This is the module `src/preprocess.py` explicitly does *not* do
(see its docstring), and the one the ingest and the serving API must share: a photo
analysed through the app has to reach `preprocess` as the exact array ingest built
at training time.

Orientation is exercised on PNG because the HEIF encoder normalises orientation to 1
when it writes a file; the expected images are computed with numpy, not with Pillow,
so the test does not simply restate the implementation.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from icc import display_p3_profile
from PIL import Image

from imageio import (
    SUPPORTED_SUFFIXES,
    Decoded,
    UnsupportedImageError,
    decode,
    decode_bytes,
)

pillow_heif = pytest.importorskip("pillow_heif")
pillow_heif.register_heif_opener()


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #

def asymmetric() -> np.ndarray:
    """An image with no symmetry at all, so every orientation is distinguishable."""
    a = np.zeros((6, 10, 3), np.uint8)
    a[:, :, 0] = np.arange(10, dtype=np.uint8) * 20
    a[:, :, 1] = np.arange(6, dtype=np.uint8)[:, None] * 30
    a[0, 0] = (255, 255, 255)
    a[-1, -1] = (7, 11, 13)
    return a


#: stored pixels -> displayed pixels, per the EXIF orientation tag. Written in numpy
#: so the assertion is independent of Pillow's own transposition code.
DISPLAY = {
    1: lambda s: s,
    2: lambda s: np.fliplr(s),
    3: lambda s: np.rot90(s, 2),
    4: lambda s: np.flipud(s),
    5: lambda s: np.swapaxes(s, 0, 1),
    6: lambda s: np.rot90(s, -1),
    7: lambda s: np.rot90(np.swapaxes(s, 0, 1), 2),
    8: lambda s: np.rot90(s, 1),
}


def png_bytes(rgb: np.ndarray, orientation: int | None = None, icc: bytes | None = None) -> bytes:
    im = Image.fromarray(rgb, "RGB")
    kw: dict = {}
    if orientation is not None:
        exif = Image.Exif()
        exif[274] = orientation
        kw["exif"] = exif
    if icc is not None:
        kw["icc_profile"] = icc
    buf = io.BytesIO()
    im.save(buf, format="PNG", **kw)
    return buf.getvalue()


def heic_bytes(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(buf, format="HEIF", quality=100)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# decoding
# --------------------------------------------------------------------------- #

def test_png_decodes_to_rgba_uint8():
    d = decode_bytes(png_bytes(asymmetric()), filename="a.png")
    assert isinstance(d, Decoded)
    assert d.pixels.dtype == np.uint8
    assert d.pixels.shape == (6, 10, 4)
    assert (d.pixels[:, :, 3] == 255).all()


def test_heic_decodes_to_rgba_uint8():
    """The real capture format — `docs/02` §4 stores raw frames as `.HEIC`."""
    d = decode_bytes(heic_bytes(asymmetric()), filename="IMG_0001.HEIC")
    assert d.pixels.dtype == np.uint8
    assert d.pixels.shape[2] == 4
    assert d.pixels.shape[:2] == (6, 10)


def test_heic_and_png_are_both_supported():
    assert {".heic", ".heif", ".png"} <= set(SUPPORTED_SUFFIXES)


def test_grayscale_input_is_promoted_to_rgba():
    g = np.arange(48, dtype=np.uint8).reshape(6, 8)
    buf = io.BytesIO()
    Image.fromarray(g, "L").save(buf, format="PNG")
    d = decode_bytes(buf.getvalue(), filename="g.png")
    assert d.pixels.shape == (6, 8, 4)
    assert (d.pixels[:, :, 0] == d.pixels[:, :, 1]).all()


def test_path_and_bytes_callers_agree(tmp_path):
    """One implementation, two callers: ingest decodes a path, the API decodes an upload."""
    raw = png_bytes(asymmetric(), orientation=6)
    p = tmp_path / "IMG_1.png"
    p.write_bytes(raw)
    assert np.array_equal(decode(p).pixels, decode_bytes(raw, filename=p.name).pixels)


def test_unsupported_suffix_is_refused(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_bytes(b"not an image")
    with pytest.raises(UnsupportedImageError):
        decode(p)


def test_corrupt_payload_is_refused():
    with pytest.raises(UnsupportedImageError):
        decode_bytes(b"\x89PNG\r\n\x1a\n garbage", filename="broken.png")


# --------------------------------------------------------------------------- #
# EXIF orientation — applied, then erased
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("orientation", sorted(DISPLAY))
def test_exif_orientation_is_applied(orientation):
    stored = asymmetric()
    d = decode_bytes(png_bytes(stored, orientation=orientation), filename="a.png")
    expected = DISPLAY[orientation](stored)
    assert np.array_equal(d.pixels[:, :, :3], expected)


@pytest.mark.parametrize("orientation", sorted(DISPLAY))
def test_orientation_that_was_applied_is_reported(orientation):
    d = decode_bytes(png_bytes(asymmetric(), orientation=orientation), filename="a.png")
    assert d.orientation == orientation


def test_orientation_is_erased_not_carried():
    """Re-decoding what we produced must be a no-op.

    If the tag survived into the output it would be applied a second time here and the
    image would come back rotated. This is the property the whole step exists for: the
    orientation is consumed once, at ingest, and never again.
    """
    once = decode_bytes(png_bytes(asymmetric(), orientation=6), filename="a.png")
    twice = decode_bytes(png_bytes(once.pixels[:, :, :3]), filename="a.png")
    assert np.array_equal(once.pixels, twice.pixels)


def test_missing_orientation_tag_is_treated_as_upright():
    stored = asymmetric()
    d = decode_bytes(png_bytes(stored), filename="a.png")
    assert d.orientation == 1
    assert np.array_equal(d.pixels[:, :, :3], stored)


# --------------------------------------------------------------------------- #
# colour — Display P3 to sRGB, profile dropped
# --------------------------------------------------------------------------- #

def test_display_p3_is_converted_to_srgb():
    """A saturated in-gamut green must move. Pure primaries clip and would prove nothing."""
    rgb = np.full((4, 4, 3), (0, 200, 80), np.uint8)
    tagged = decode_bytes(png_bytes(rgb, icc=display_p3_profile()), filename="p3.png")
    untagged = decode_bytes(png_bytes(rgb), filename="plain.png")

    assert not np.array_equal(tagged.pixels[:, :, :3], untagged.pixels[:, :, :3])
    assert tagged.source_profile is not None
    assert "P3" in tagged.source_profile


def test_converted_pixels_are_the_srgb_equivalent():
    rgb = np.full((4, 4, 3), (0, 200, 80), np.uint8)
    d = decode_bytes(png_bytes(rgb, icc=display_p3_profile()), filename="p3.png")
    # The same colour expressed in sRGB: greener and less blue than the P3 triple.
    assert d.pixels[0, 0, 1] > 200
    assert d.pixels[0, 0, 2] < 80


def test_untagged_image_is_assumed_srgb_and_left_alone():
    rgb = np.full((4, 4, 3), (0, 200, 80), np.uint8)
    d = decode_bytes(png_bytes(rgb), filename="plain.png")
    assert np.array_equal(d.pixels[:, :, :3], rgb)
    assert d.source_profile is None


def test_profile_is_dropped_from_the_result():
    """Nothing downstream may re-manage colour: the array is plain sRGB, no profile."""
    rgb = np.full((4, 4, 3), (0, 200, 80), np.uint8)
    d = decode_bytes(png_bytes(rgb, icc=display_p3_profile()), filename="p3.png")
    round_tripped = decode_bytes(png_bytes(d.pixels[:, :, :3]), filename="out.png")
    assert round_tripped.source_profile is None
    assert np.array_equal(round_tripped.pixels, d.pixels)


def test_decode_is_deterministic():
    raw = png_bytes(asymmetric(), orientation=6, icc=display_p3_profile())
    a = decode_bytes(raw, filename="a.png")
    b = decode_bytes(raw, filename="a.png")
    assert np.array_equal(a.pixels, b.pixels)
    assert a.orientation == b.orientation
