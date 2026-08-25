"""Step 1 of the pipeline: bytes on disk -> an 8-bit sRGB RGBA array.

`src/preprocess.py` starts at step 2 and says so in its own docstring: it takes an
"already-decoded 8-bit sRGB RGBA array". This module is what produces that array, and
it does the three things that must happen exactly once in the life of a capture:

1. **Decode** — HEIC/HEIF straight off the iPhone, and PNG/JPEG for anything else
   (the synthetic sessions are PNG).
2. **Apply the EXIF orientation, then erase it.** An orientation applied twice is a
   rotated plate, and a rotated plate is a different nodal pattern. The tag is consumed
   here and never travels further: the return value is a bare array with no metadata
   attached, so there is nothing downstream that *could* apply it again.
3. **Convert Display P3 to sRGB and drop the profile.** An iPhone photographs in
   Display P3. Feeding P3 numbers to a pipeline that assumes sRGB shifts every
   luminance the quality gate measures. After this step the array carries no profile,
   so no later stage can colour-manage it a second time.

**One implementation, two callers.** Ingest decodes a path; the serving API decodes an
uploaded byte string. Both land in `_decode_pil`, so a photo analysed in the app is
transformed exactly as the training frames were. Forking this is the same mistake
`CLAUDE.md` rule 1 forbids for `preprocess`, one step earlier in the pipeline.

The transform is pure: no globals, no logging, nothing written to disk.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image, ImageCms, UnidentifiedImageError

#: Suffixes ingest will open. `.heic`/`.heif` are the capture format (`docs/02` §4);
#: `.png` covers the synthetic sessions; `.jpg` is accepted because a phone set to
#: "Most Compatible" writes JPEG without telling anyone.
SUPPORTED_SUFFIXES = (".heic", ".heif", ".png", ".jpg", ".jpeg")

_EXIF_ORIENTATION = 274
_ORIENTATION_UPRIGHT = 1

#: stored -> displayed, per EXIF tag 274. Applied with numpy rather than
#: `ImageOps.exif_transpose` so the mapping is visible and unit-testable here.
_TRANSPOSE = {
    1: lambda a: a,
    2: lambda a: np.fliplr(a),
    3: lambda a: np.rot90(a, 2),
    4: lambda a: np.flipud(a),
    5: lambda a: np.swapaxes(a, 0, 1),
    6: lambda a: np.rot90(a, -1),
    7: lambda a: np.rot90(np.swapaxes(a, 0, 1), 2),
    8: lambda a: np.rot90(a, 1),
}

_heif_registered = False


class UnsupportedImageError(ValueError):
    """The file is not an image this project reads, or it is corrupt."""


class Decoded(NamedTuple):
    """The array, plus what had to be undone to get it.

    `orientation` and `source_profile` are recorded, not discarded, because ingest
    writes them into `data/ingest_report.json`: a session where every frame needed a
    90-degree rotation says something about how the rig was mounted, and it is cheaper
    to read that from the report than to re-derive it from the raw captures later.

    `pixels` is `(h, w, 4)` uint8 RGBA in sRGB, with no profile and no EXIF.
    """

    pixels: np.ndarray
    orientation: int
    source_profile: str | None
    format: str


def decode(path: Path | str) -> Decoded:
    """Decode a capture from disk. The ingest-side caller."""
    p = Path(path)
    if p.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise UnsupportedImageError(
            f"{p.name}: suffix {p.suffix!r} is not one of {', '.join(SUPPORTED_SUFFIXES)}"
        )
    return decode_bytes(p.read_bytes(), filename=p.name)


def decode_bytes(data: bytes, filename: str | None = None) -> Decoded:
    """Decode a capture already in memory. The API-side caller.

    `filename` is used only to improve the error message; the format is detected from
    the payload, so an upload with a wrong or missing extension still decodes.
    """
    _register_heif()
    name = filename or "<bytes>"
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            return _decode_pil(im)
    except UnsupportedImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise UnsupportedImageError(f"{name}: cannot decode ({exc})") from exc


# --------------------------------------------------------------------------- #
# the single implementation both callers reach
# --------------------------------------------------------------------------- #

def _decode_pil(im: Image.Image) -> Decoded:
    fmt = im.format or "?"
    orientation = _orientation_of(im)
    profile_bytes = im.info.get("icc_profile")

    rgb, profile_name = _to_srgb(im, profile_bytes)
    rgb = _apply_orientation(rgb, orientation)

    alpha = np.full(rgb.shape[:2] + (1,), 255, np.uint8)
    pixels = np.ascontiguousarray(np.concatenate([rgb, alpha], axis=2))
    return Decoded(pixels, orientation, profile_name, fmt)


def _orientation_of(im: Image.Image) -> int:
    try:
        value = im.getexif().get(_EXIF_ORIENTATION, _ORIENTATION_UPRIGHT)
    except Exception:  # noqa: BLE001 - a malformed EXIF block must not cost us the capture
        return _ORIENTATION_UPRIGHT
    value = int(value or _ORIENTATION_UPRIGHT)
    return value if value in _TRANSPOSE else _ORIENTATION_UPRIGHT


def _apply_orientation(rgb: np.ndarray, orientation: int) -> np.ndarray:
    """Apply the tag and return a fresh, contiguous array — the tag itself is gone.

    Erasure is structural: `Decoded.pixels` is an ndarray, so there is no EXIF block
    left for anything downstream to read, let alone re-apply.
    """
    return np.ascontiguousarray(_TRANSPOSE[orientation](rgb))


def _to_srgb(im: Image.Image, profile_bytes: bytes | None) -> tuple[np.ndarray, str | None]:
    """Convert to sRGB if the image is tagged with anything else, and drop the profile.

    An untagged image is assumed to be sRGB already and is left byte-for-byte alone;
    guessing a profile for it would be a silent, unrecorded transform.
    """
    if profile_bytes is None:
        return _as_rgb_array(im), None

    try:
        source = ImageCms.getOpenProfile(io.BytesIO(profile_bytes))
        name = (ImageCms.getProfileDescription(source) or "").strip() or "unnamed profile"
        converted = ImageCms.profileToProfile(
            im.convert("RGB"),
            source,
            ImageCms.createProfile("sRGB"),
            renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
            outputMode="RGB",
        )
    except Exception:  # noqa: BLE001 - littleCMS raises a wide and undocumented range
        # A profile we cannot read is reported as unusable rather than guessed at: the
        # pixels pass through untouched and `source_profile` stays None, so the report
        # shows the image was not colour-managed instead of implying it was.
        return _as_rgb_array(im), None

    if converted is None:
        return _as_rgb_array(im), name
    return np.asarray(converted, np.uint8), name


def _as_rgb_array(im: Image.Image) -> np.ndarray:
    """Grayscale, palette and RGBA inputs all normalise to three 8-bit channels."""
    return np.asarray(im.convert("RGB"), np.uint8)


def _register_heif() -> None:
    """Teach Pillow the capture format, once per process.

    `pillow-heif` is a hard dependency of ingest (`requirements.txt`), but the failure
    is raised at decode time rather than at import so the rest of the module — and the
    PNG path the synthetic sessions use — still works on a machine without it.
    """
    global _heif_registered
    if _heif_registered:
        return
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except ImportError:  # pragma: no cover - exercised only on an incomplete install
        pass
    _heif_registered = True
