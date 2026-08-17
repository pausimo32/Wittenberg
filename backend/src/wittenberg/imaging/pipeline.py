"""Pipeline de prétraitement déterministe — implémentation unique.

Ce module est la SEULE implémentation du prétraitement. L'API, le script
d'évaluation et l'outil en ligne de commande l'appellent tous ; un futur script
d'entraînement doit faire de même, afin qu'entraînement et inférence ne
puissent jamais diverger.

Le comportement est entièrement piloté par la section ``preprocessing`` du
manifeste et versionné par ``preprocessing.version``. Toute modification du
comportement impose d'incrémenter cette version.

Ordre des opérations (fixe) :

1. décodage sûr, redressement EXIF, suppression des métadonnées (``validation``)
2. localisation de la plaque, correction de perspective ou repli central
3. mesure de la qualité sur la région utile
4. conversion en niveaux de gris
5. normalisation de contraste
6. réduction de bruit
7. redimensionnement à la géométrie du modèle
8. extraction de contours (Canny) si le modèle a été entraîné ainsi
9. normalisation numérique et construction du tenseur
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from wittenberg.imaging.plate import PlateResult, locate_plate
from wittenberg.imaging.quality import QualityReport, assess_quality
from wittenberg.imaging.validation import DecodedImage, decode_image
from wittenberg.model.manifest import (
    InputSection,
    Layout,
    ModelManifest,
    NormalizationMode,
    PreprocessingSection,
    ResizeMode,
)


class PreprocessingUnavailable(Exception):
    """Le prétraitement exigé par le manifeste ne peut pas être exécuté.

    Cas typique : le manifeste active Canny ou CLAHE alors qu'OpenCV n'est pas
    installé. Détecté au démarrage, ce défaut rend le service non disponible
    plutôt que de produire silencieusement un tenseur différent de celui vu à
    l'entraînement.
    """


@dataclass(frozen=True)
class PreprocessResult:
    """Sortie du pipeline."""

    #: tenseur prêt pour le modèle, lot de taille 1 inclus
    tensor: np.ndarray
    quality: QualityReport
    plate: PlateResult
    #: trace ordonnée des opérations réellement appliquées
    steps: list[str] = field(default_factory=list)
    #: aperçu PNG encodé en base64, uniquement si demandé explicitement
    debug_preview: str | None = None


def _import_cv2() -> Any | None:
    try:
        import cv2
    except ImportError:  # pragma: no cover - dépend de l'installation
        return None
    return cv2


def check_pipeline_requirements(preprocessing: PreprocessingSection) -> None:
    """Vérifie que les dépendances du prétraitement demandé sont présentes.

    Appelé au démarrage : mieux vaut refuser de servir que servir un tenseur
    qui ne correspond pas à l'entraînement.

    :raises PreprocessingUnavailable: si une dépendance manque.
    """
    if _import_cv2() is not None:
        return
    missing = []
    if preprocessing.edge_detection.enabled:
        missing.append("edge_detection.enabled=true (Canny)")
    if preprocessing.contrast.mode == "clahe":
        missing.append("contrast.mode='clahe'")
    if missing:
        raise PreprocessingUnavailable(
            "OpenCV est requis par le manifeste pour : "
            + ", ".join(missing)
            + ". Installez `opencv-python-headless`."
        )


# ── Points d'entrée ───────────────────────────────────────────────────────


def preprocess_bytes(
    data: bytes,
    manifest: ModelManifest,
    *,
    max_bytes: int,
    max_pixels: int,
    max_dimension: int,
    min_dimension: int,
    debug_preview: bool = False,
) -> tuple[PreprocessResult, DecodedImage]:
    """Valide, décode et prétraite des octets d'image."""
    decoded = decode_image(
        data,
        max_bytes=max_bytes,
        max_pixels=max_pixels,
        max_dimension=max_dimension,
        min_dimension=min_dimension,
    )
    result = preprocess_image(decoded.image, manifest, debug_preview=debug_preview)
    return result, decoded


def preprocess_image(
    image: Image.Image,
    manifest: ModelManifest,
    *,
    debug_preview: bool = False,
) -> PreprocessResult:
    """Prétraite une image PIL déjà décodée et convertie en RGB."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    config = manifest.preprocessing
    spec = manifest.input
    steps: list[str] = []
    cv2 = _import_cv2()

    rgb = np.asarray(image, dtype=np.uint8)

    # ── 2. Localisation de la plaque ─────────────────────────────────────
    plate = locate_plate(rgb, config.plate_detection)
    steps.append(f"plate:{plate.method}")

    # ── 3. Qualité, mesurée sur la région utile avant toute altération ───
    region_gray = _to_gray(plate.region)
    quality = assess_quality(
        region_gray,
        coverage=plate.coverage,
        skew=plate.skew,
        plate_detected=plate.detected,
    )

    # ── 4. Niveaux de gris ───────────────────────────────────────────────
    if config.grayscale:
        working = region_gray
        steps.append("grayscale")
    else:
        working = plate.region

    # ── 5. Contraste ─────────────────────────────────────────────────────
    working = _apply_contrast(working, config, cv2, steps)

    # ── 6. Débruitage ────────────────────────────────────────────────────
    working = _apply_denoise(working, config, steps)

    # ── 7. Redimensionnement ─────────────────────────────────────────────
    working = _resize(working, spec, config, steps)

    # ── 8. Contours ──────────────────────────────────────────────────────
    if config.edge_detection.enabled:
        if cv2 is None:
            raise PreprocessingUnavailable(
                "le manifeste active la détection de contours mais OpenCV est absent"
            )
        gray_for_edges = working if working.ndim == 2 else _to_gray(working)
        working = cv2.Canny(
            gray_for_edges,
            config.edge_detection.low_threshold,
            config.edge_detection.high_threshold,
            apertureSize=config.edge_detection.aperture_size,
        )
        steps.append(
            f"canny:{config.edge_detection.low_threshold}-{config.edge_detection.high_threshold}"
        )

    preview = _encode_preview(working) if debug_preview else None

    # ── 9. Tenseur ───────────────────────────────────────────────────────
    tensor = build_tensor(working, spec)
    steps.append(f"tensor:{spec.dtype}{tuple(tensor.shape)}")

    return PreprocessResult(
        tensor=tensor,
        quality=quality,
        plate=plate,
        steps=steps,
        debug_preview=preview,
    )


# ── Étapes ────────────────────────────────────────────────────────────────


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    """Luminance perceptuelle ITU-R BT.601, identique au frontend historique."""
    if rgb.ndim == 2:
        return rgb
    values = rgb.astype(np.float64)
    gray = 0.299 * values[..., 0] + 0.587 * values[..., 1] + 0.114 * values[..., 2]
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)


def _apply_contrast(
    array: np.ndarray, config: PreprocessingSection, cv2: Any | None, steps: list[str]
) -> np.ndarray:
    mode = config.contrast.mode
    if mode == "none":
        return array

    if mode == "minmax":
        values = array.astype(np.float64)
        low, high = float(values.min()), float(values.max())
        span = high - low
        if span < 1e-6:
            # Image uniforme : l'étirement amplifierait uniquement le bruit.
            steps.append("contrast:minmax(skipped-flat)")
            return array
        stretched = (values - low) * (255.0 / span)
        steps.append("contrast:minmax")
        return np.clip(np.rint(stretched), 0, 255).astype(np.uint8)

    # CLAHE : égalisation locale, robuste à un éclairage non uniforme.
    if cv2 is None:
        raise PreprocessingUnavailable("le manifeste demande CLAHE mais OpenCV est absent")
    clahe = cv2.createCLAHE(
        clipLimit=config.contrast.clip_limit,
        tileGridSize=(config.contrast.tile_grid, config.contrast.tile_grid),
    )
    steps.append(f"contrast:clahe({config.contrast.clip_limit})")
    if array.ndim == 2:
        return clahe.apply(array)
    channels = [clahe.apply(array[..., i]) for i in range(array.shape[2])]
    return np.stack(channels, axis=-1)


def _apply_denoise(array: np.ndarray, config: PreprocessingSection, steps: list[str]) -> np.ndarray:
    mode = config.denoise.mode
    if mode == "none":
        return array
    size = config.denoise.kernel_size
    image = _to_pil(array)
    if mode == "median":
        filtered = image.filter(ImageFilter.MedianFilter(size=size))
    else:
        # Rayon choisi pour qu'un noyau de taille n couvre ±(n-1)/2 pixels.
        filtered = image.filter(ImageFilter.GaussianBlur(radius=(size - 1) / 2))
    steps.append(f"denoise:{mode}({size})")
    return np.asarray(filtered, dtype=np.uint8)


def _resize(
    array: np.ndarray,
    spec: InputSection,
    config: PreprocessingSection,
    steps: list[str],
) -> np.ndarray:
    target = (spec.width, spec.height)
    image = _to_pil(array)

    if config.resize_mode is ResizeMode.STRETCH:
        resized = image.resize(target, Image.Resampling.LANCZOS)
        steps.append(f"resize:stretch{target}")
        return np.asarray(resized, dtype=np.uint8)

    # Mode PAD : conserve la géométrie du motif, complète les bords.
    scale = min(spec.width / image.width, spec.height / image.height)
    new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resized = image.resize(new_size, Image.Resampling.LANCZOS)

    if array.ndim == 2:
        background = Image.new("L", target, config.pad_value)
    else:
        fill = (config.pad_value,) * 3
        background = Image.new("RGB", target, fill)
    offset = ((spec.width - new_size[0]) // 2, (spec.height - new_size[1]) // 2)
    background.paste(resized, offset)
    steps.append(f"resize:pad{target}")
    return np.asarray(background, dtype=np.uint8)


def build_tensor(array: np.ndarray, spec: InputSection) -> np.ndarray:
    """Construit le tenseur final : canaux, normalisation, disposition, dtype."""
    values = array

    # ── Canaux ───────────────────────────────────────────────────────────
    if values.ndim == 2:
        values = values[..., np.newaxis]
    if values.shape[2] != spec.channels:
        if spec.channels == 1:
            values = _to_gray(values)[..., np.newaxis]
        elif values.shape[2] == 1:
            values = np.repeat(values, 3, axis=2)
        else:
            raise PreprocessingUnavailable(
                f"impossible de produire {spec.channels} canaux depuis une image "
                f"à {values.shape[2]} canaux"
            )

    if values.shape[0] != spec.height or values.shape[1] != spec.width:
        raise PreprocessingUnavailable(
            f"géométrie inattendue après redimensionnement : "
            f"{values.shape[:2]} au lieu de ({spec.height}, {spec.width})"
        )

    # ── Normalisation ────────────────────────────────────────────────────
    normalized = _normalize(values.astype(np.float64), spec)

    low, high = spec.value_range
    normalized = np.clip(normalized, low, high)

    # ── Disposition et dtype ─────────────────────────────────────────────
    tensor = normalized[np.newaxis, ...]  # NHWC avec lot de 1
    if spec.layout is Layout.NCHW:
        tensor = np.transpose(tensor, (0, 3, 1, 2))
    return np.ascontiguousarray(tensor.astype(np.dtype(spec.dtype)))


def _normalize(values: np.ndarray, spec: InputSection) -> np.ndarray:
    mode = spec.normalization.mode
    if mode is NormalizationMode.RAW:
        return values
    if mode is NormalizationMode.SCALE_01:
        return values / 255.0
    if mode is NormalizationMode.SCALE_SIGNED:
        return values / 127.5 - 1.0
    mean = np.asarray(spec.normalization.mean, dtype=np.float64)
    std = np.asarray(spec.normalization.std, dtype=np.float64)
    return (values / 255.0 - mean) / std


def _to_pil(array: np.ndarray) -> Image.Image:
    if array.ndim == 2:
        return Image.fromarray(array, mode="L")
    return Image.fromarray(array, mode="RGB")


def _encode_preview(array: np.ndarray) -> str:
    """Encode l'image prétraitée en PNG base64 (développement uniquement)."""
    buffer = io.BytesIO()
    _to_pil(array).save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
