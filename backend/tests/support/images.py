"""Générateurs d'images de test.

Toutes les images sont produites à la volée : le dépôt ne contient aucun
binaire de test.
"""

from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image


def chladni_like(size: int = 256, m: int = 3, n: int = 4, plate_ratio: float = 0.8) -> Image.Image:
    """Plaque claire sur fond sombre, avec des lignes nodales marquées.

    Reproduit la structure visuelle d'une photographie réelle : un carré clair
    centré, portant des lignes sombres là où la fonction de Chladni s'annule.
    """
    canvas = np.zeros((size, size), dtype=np.float64)
    side = int(size * plate_ratio)
    offset = (size - side) // 2

    coords = np.linspace(0, math.pi, side)
    x, y = np.meshgrid(coords, coords, indexing="xy")
    field = np.cos(m * x) * np.cos(n * y) - np.cos(n * x) * np.cos(m * y)
    # Les grains s'accumulent sur les lignes nodales : clair partout, sombre
    # là où le champ s'annule.
    plate = 235.0 - 215.0 * np.exp(-((field / 0.06) ** 2))

    canvas[offset : offset + side, offset : offset + side] = plate
    return Image.fromarray(canvas.astype(np.uint8), mode="L").convert("RGB")


def flat_image(size: int = 128, value: int = 10) -> Image.Image:
    """Image parfaitement uniforme : aucun motif."""
    return Image.new("RGB", (size, size), (value, value, value))


def noise_image(size: int = 128, seed: int = 7) -> Image.Image:
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    return Image.fromarray(data, mode="RGB")


def encode(image: Image.Image, fmt: str = "PNG", **kwargs: object) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


def jpeg_with_exif_orientation(orientation: int = 6, size: int = 200) -> bytes:
    """JPEG portant une balise EXIF Orientation.

    L'image source est un rectangle : la rotation EXIF est donc observable via
    l'inversion des dimensions après redressement.
    """
    image = chladni_like(size=size).resize((size, size // 2))
    exif = image.getexif()
    exif[0x0112] = orientation
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif, quality=92)
    return buffer.getvalue()


def rgba_image(size: int = 128) -> Image.Image:
    """Image avec canal alpha partiellement transparent."""
    base = chladni_like(size=size).convert("RGBA")
    alpha = Image.new("L", base.size, 255)
    for y in range(size // 4):
        for x in range(size):
            alpha.putpixel((x, y), 0)
    base.putalpha(alpha)
    return base


def grayscale_image(size: int = 128) -> Image.Image:
    return chladni_like(size=size).convert("L")


def cmyk_image(size: int = 128) -> Image.Image:
    return chladni_like(size=size).convert("CMYK")


def huge_declared_png(width: int, height: int) -> bytes:
    """PNG réel aux dimensions demandées, sans contenu coûteux à générer."""
    return encode(Image.new("RGB", (width, height), (0, 0, 0)), "PNG")
