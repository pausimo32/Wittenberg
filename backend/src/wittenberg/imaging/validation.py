"""Validation et décodage sûrs des images téléversées.

Le type est déterminé par le contenu réel (signature binaire), jamais par
l'extension ni par l'en-tête ``Content-Type`` fourni par le client.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from enum import StrEnum

from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError

# Une image tronquée doit échouer franchement plutôt que produire un tenseur
# à moitié rempli, qui donnerait une prédiction silencieusement fausse.
ImageFile.LOAD_TRUNCATED_IMAGES = False

# La garde de Pillow est désactivée : la limite réelle vient de la
# configuration et est appliquée explicitement dans `decode_image`.
Image.MAX_IMAGE_PIXELS = None


class ImageFormat(StrEnum):
    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"
    GIF = "gif"
    BMP = "bmp"
    TIFF = "tiff"
    HEIF = "heif"
    UNKNOWN = "unknown"


#: Formats décodés sans dépendance supplémentaire.
NATIVELY_SUPPORTED = frozenset(
    {
        ImageFormat.JPEG,
        ImageFormat.PNG,
        ImageFormat.WEBP,
        ImageFormat.BMP,
        ImageFormat.TIFF,
    }
)

#: Marques ISO-BMFF correspondant à HEIC/HEIF/AVIF.
_HEIF_BRANDS = frozenset(
    {
        b"heic",
        b"heix",
        b"heim",
        b"heis",
        b"hevc",
        b"hevm",
        b"hevs",
        b"mif1",
        b"msf1",
        b"avif",
        b"avis",
    }
)


class ImageValidationError(Exception):
    """Entrée invalide. ``code`` est stable et exploitable par le frontend."""

    def __init__(self, code: str, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class DecodedImage:
    """Image décodée, normalisée en RGB et débarrassée de ses métadonnées."""

    image: Image.Image
    detected_format: ImageFormat
    original_width: int
    original_height: int
    original_mode: str
    exif_transposed: bool
    had_alpha: bool


def sniff_format(data: bytes) -> ImageFormat:
    """Identifie le format d'après les premiers octets."""
    if len(data) < 12:
        return ImageFormat.UNKNOWN
    if data.startswith(b"\xff\xd8\xff"):
        return ImageFormat.JPEG
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ImageFormat.PNG
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ImageFormat.WEBP
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return ImageFormat.GIF
    if data[:2] == b"BM":
        return ImageFormat.BMP
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return ImageFormat.TIFF
    if data[4:8] == b"ftyp" and data[8:12] in _HEIF_BRANDS:
        return ImageFormat.HEIF
    return ImageFormat.UNKNOWN


def _heif_available() -> bool:
    try:
        import pillow_heif  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    return True


def _register_heif() -> None:
    import pillow_heif  # type: ignore[import-not-found]

    pillow_heif.register_heif_opener()


def decode_image(
    data: bytes,
    *,
    max_bytes: int,
    max_pixels: int,
    max_dimension: int,
    min_dimension: int,
) -> DecodedImage:
    """Valide puis décode des octets en une image RGB exploitable.

    Les contrôles sont ordonnés du moins coûteux au plus coûteux : taille,
    signature, en-tête, puis décodage complet. Une image « bombe » est donc
    rejetée avant d'être décompressée.
    """
    if not data:
        raise ImageValidationError("empty_file", "Le fichier reçu est vide.")

    if len(data) > max_bytes:
        limit_mb = max_bytes / (1024 * 1024)
        raise ImageValidationError(
            "file_too_large",
            f"L'image dépasse la taille maximale autorisée ({limit_mb:.0f} Mo).",
            status_code=413,
        )

    detected = sniff_format(data)

    if detected is ImageFormat.HEIF and not _heif_available():
        raise ImageValidationError(
            "unsupported_heif",
            "Les images HEIC/HEIF ne sont pas prises en charge par ce serveur. "
            "Exportez la photo en JPEG ou PNG depuis votre téléphone, ou "
            "installez la dépendance optionnelle `pillow-heif` côté serveur.",
            status_code=415,
        )
    if detected is ImageFormat.HEIF:
        _register_heif()
    elif detected is ImageFormat.UNKNOWN:
        raise ImageValidationError(
            "unsupported_format",
            "Le fichier n'est pas une image reconnue. Formats acceptés : "
            "JPEG, PNG, WebP, BMP, TIFF.",
            status_code=415,
        )
    elif detected is ImageFormat.GIF:
        raise ImageValidationError(
            "unsupported_format",
            "Le format GIF n'est pas accepté. Utilisez une photographie JPEG, PNG ou WebP.",
            status_code=415,
        )

    # ── Inspection de l'en-tête avant tout décodage complet ──────────────
    try:
        with Image.open(io.BytesIO(data)) as probe:
            width, height = probe.size
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError(
            "corrupt_image",
            "L'image ne peut pas être lue : le fichier est corrompu ou incomplet.",
        ) from exc

    if width <= 0 or height <= 0:
        raise ImageValidationError("corrupt_image", "L'image déclare des dimensions nulles.")

    if width < min_dimension or height < min_dimension:
        raise ImageValidationError(
            "image_too_small",
            f"L'image est trop petite ({width}×{height} px). "
            f"Minimum requis : {min_dimension}×{min_dimension} px.",
        )

    if width > max_dimension or height > max_dimension:
        raise ImageValidationError(
            "image_too_large",
            f"L'image est trop grande ({width}×{height} px). "
            f"Maximum autorisé : {max_dimension} px par côté.",
            status_code=413,
        )

    if width * height > max_pixels:
        raise ImageValidationError(
            "image_too_many_pixels",
            f"L'image contient trop de pixels ({width * height:,} px). "
            f"Maximum autorisé : {max_pixels:,} px.".replace(",", " "),
            status_code=413,
        )

    # ── Décodage complet ─────────────────────────────────────────────────
    try:
        with Image.open(io.BytesIO(data)) as opened:
            opened.load()
            original_mode = opened.mode
            had_alpha = original_mode in ("RGBA", "LA", "PA") or "transparency" in opened.info
            # 0x0112 = balise EXIF Orientation ; 1 signifie « déjà droite ».
            orientation = opened.getexif().get(0x0112)
            exif_applied = orientation not in (None, 1)
            transposed = ImageOps.exif_transpose(opened)
            rgb = _to_rgb(transposed if transposed is not None else opened)
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError(
            "corrupt_image",
            "L'image ne peut pas être décodée : le fichier est corrompu ou tronqué.",
        ) from exc

    # Reconstruire l'image à partir des seuls pixels : toute métadonnée
    # (EXIF, GPS, ICC, commentaires) est ainsi abandonnée.
    stripped = Image.frombytes("RGB", rgb.size, rgb.tobytes())

    return DecodedImage(
        image=stripped,
        detected_format=detected,
        original_width=width,
        original_height=height,
        original_mode=original_mode,
        exif_transposed=exif_applied,
        had_alpha=had_alpha,
    )


def _to_rgb(image: Image.Image) -> Image.Image:
    """Convertit tout mode d'entrée en RGB de façon déterministe.

    Les zones transparentes sont composées sur du noir : la plaque est
    photographiée sur fond sombre et le remplissage du redimensionnement est
    lui aussi noir, ce qui évite d'introduire un bord clair artificiel qui
    serait interprété comme une ligne nodale.
    """
    mode = image.mode
    if mode == "RGB":
        return image
    if mode in ("RGBA", "LA", "PA") or (mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    if mode in ("L", "1", "I", "I;16", "F", "P", "CMYK", "YCbCr", "LAB", "HSV"):
        if mode in ("I", "I;16", "F"):
            # Les modes à haute profondeur doivent d'abord être ramenés à 8 bits.
            return image.convert("L").convert("RGB")
        return image.convert("RGB")
    return image.convert("RGB")
