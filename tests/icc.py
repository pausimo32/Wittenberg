"""Build a minimal Display P3 ICC v2 profile in memory.

Windows ships only sRGB/SWOP profiles, and Pillow's `ImageCms.createProfile` can make
sRGB but not P3. The colour-management step of `src/imageio.py` is only meaningfully
tested against a real wide-gamut profile, so this builds one: a matrix/TRC profile with
the Display P3 primaries (Bradford-adapted to D50, as in Apple's own profile) and a
gamma 2.2 curve.
"""

from __future__ import annotations

import struct

# Display P3 primaries and D50 white, chromatically adapted — the numbers Apple's
# "Display P3" profile carries. Red's Z is genuinely negative there; s15Fixed16 signs it.
_PRIMARIES = {
    b"rXYZ": (0.51512, 0.24119, -0.00105),
    b"gXYZ": (0.29198, 0.69225, 0.04189),
    b"bXYZ": (0.15706, 0.06657, 0.78407),
}
_D50 = (0.96420, 1.00000, 0.82491)
_GAMMA = 2.2


def _s15f16(x: float) -> bytes:
    return struct.pack(">i", round(x * 65536.0))


def _xyz_tag(xyz) -> bytes:
    return b"XYZ " + b"\x00" * 4 + b"".join(_s15f16(v) for v in xyz)


def _curve_tag(gamma: float) -> bytes:
    return b"curv" + b"\x00" * 4 + struct.pack(">I", 1) + struct.pack(">H", round(gamma * 256))


def _desc_tag(text: str) -> bytes:
    raw = text.encode("ascii") + b"\x00"
    return (
        b"desc" + b"\x00" * 4
        + struct.pack(">I", len(raw)) + raw
        + b"\x00" * 12                       # unicode language code + count
        + b"\x00" * 67                       # scriptcode description
    )


def display_p3_profile() -> bytes:
    tags: list[tuple[bytes, bytes]] = [
        (b"desc", _desc_tag("Display P3 (synthetic, tests only)")),
        (b"wtpt", _xyz_tag(_D50)),
        (b"cprt", _desc_tag("no copyright, generated for tests")),
    ]
    for sig, xyz in _PRIMARIES.items():
        tags.append((sig, _xyz_tag(xyz)))
    for sig in (b"rTRC", b"gTRC", b"bTRC"):
        tags.append((sig, _curve_tag(_GAMMA)))

    header_size = 128
    table_size = 4 + 12 * len(tags)
    offset = header_size + table_size

    table, blob = b"", b""
    for sig, data in tags:
        padded = data + b"\x00" * (-len(data) % 4)
        table += sig + struct.pack(">II", offset + len(blob), len(data))
        blob += padded

    body = struct.pack(">I", len(tags)) + table + blob
    total = header_size + len(body)

    header = (
        struct.pack(">I", total)
        + b"lcms"
        + struct.pack(">I", 0x02100000)
        + b"mntr" + b"RGB " + b"XYZ "
        + b"\x00" * 12
        + b"acsp" + b"MSFT"
        + b"\x00" * 4 * 3
        + b"\x00" * 8
        + struct.pack(">I", 0)
        + b"".join(_s15f16(v) for v in _D50)
        + b"\x00" * 4
        + b"\x00" * 44
    )
    assert len(header) == header_size, len(header)
    return header + body
