"""Indicateurs de qualité de l'image d'entrée.

Ces mesures ne sont pas des certitudes scientifiques : elles servent
uniquement à indiquer à l'utilisateur qu'une photographie mérite d'être
reprise. Elles n'influencent jamais la prédiction elle-même.

Toutes les mesures sont calculées en NumPy pur, sans OpenCV, afin que le
diagnostic reste disponible quelle que soit l'installation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ── Seuils ────────────────────────────────────────────────────────────────
# Calibrés pour des photographies de plaque sur fond sombre. Ils sont
# volontairement permissifs : un avertissement de trop est moins gênant
# qu'une bonne image refusée.
BLUR_VARIANCE_MIN = 80.0
DARK_PIXEL_VALUE = 16
BRIGHT_PIXEL_VALUE = 240
UNDEREXPOSED_RATIO_MAX = 0.80
OVEREXPOSED_RATIO_MAX = 0.25
CONTRAST_STD_MIN = 22.0
COVERAGE_MIN = 0.25
SKEW_MAX = 0.20
EDGE_DENSITY_MIN = 0.010

SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"


@dataclass(frozen=True)
class QualityWarning:
    """Avertissement adressé à l'utilisateur, avec un code stable."""

    code: str
    message: str
    severity: str = SEVERITY_WARNING


@dataclass(frozen=True)
class QualityReport:
    """Synthèse de la qualité perçue de l'image."""

    score: float
    warnings: list[QualityWarning] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


def laplacian_variance(gray: np.ndarray) -> float:
    """Variance du laplacien : proxy classique de la netteté.

    Utilise le noyau à 4 voisins appliqué par décalages de tableau, ce qui
    évite toute dépendance à une bibliothèque de convolution.
    """
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    values = gray.astype(np.float64)
    laplacian = (
        values[:-2, 1:-1]
        + values[2:, 1:-1]
        + values[1:-1, :-2]
        + values[1:-1, 2:]
        - 4.0 * values[1:-1, 1:-1]
    )
    return float(np.var(laplacian))


def edge_density(gray: np.ndarray) -> float:
    """Fraction de pixels appartenant à un contour marqué.

    Un motif de Chladni possède des lignes nodales nettes ; une plaque nue ou
    une photographie sans motif produit une densité très faible.
    """
    if gray.shape[0] < 2 or gray.shape[1] < 2:
        return 0.0
    values = gray.astype(np.float64)
    grad_y = np.abs(np.diff(values, axis=0))[:, :-1]
    grad_x = np.abs(np.diff(values, axis=1))[:-1, :]
    magnitude = np.hypot(grad_x, grad_y)
    if magnitude.size == 0:
        return 0.0
    # Seuil relatif à la dynamique réelle : robuste à l'exposition globale.
    threshold = max(12.0, float(np.percentile(magnitude, 90)))
    return float(np.mean(magnitude >= threshold))


def assess_quality(
    gray: np.ndarray,
    *,
    coverage: float,
    skew: float | None,
    plate_detected: bool,
) -> QualityReport:
    """Évalue la région utile en niveaux de gris (uint8)."""
    warnings: list[QualityWarning] = []

    sharpness = laplacian_variance(gray)
    contrast = float(np.std(gray.astype(np.float64)))
    dark_ratio = float(np.mean(gray <= DARK_PIXEL_VALUE))
    bright_ratio = float(np.mean(gray >= BRIGHT_PIXEL_VALUE))
    density = edge_density(gray)

    if sharpness < BLUR_VARIANCE_MIN:
        warnings.append(
            QualityWarning(
                "blurry",
                "L'image semble floue. Stabilisez l'appareil et vérifiez la mise au point.",
            )
        )
    if bright_ratio > OVEREXPOSED_RATIO_MAX:
        warnings.append(
            QualityWarning(
                "overexposed",
                "L'image paraît surexposée. Réduisez l'intensité de l'éclairage "
                "ou évitez les reflets directs sur la plaque.",
            )
        )
    if dark_ratio > UNDEREXPOSED_RATIO_MAX:
        warnings.append(
            QualityWarning(
                "underexposed",
                "L'image paraît sous-exposée. Augmentez l'éclairage de la plaque.",
            )
        )
    if contrast < CONTRAST_STD_MIN:
        warnings.append(
            QualityWarning(
                "low_contrast",
                "Le contraste entre les grains et la plaque est faible. "
                "Un éclairage plus rasant améliore la lisibilité du motif.",
            )
        )
    if not plate_detected:
        warnings.append(
            QualityWarning(
                "plate_not_detected",
                "Les bords de la plaque n'ont pas pu être localisés ; un recadrage "
                "central a été appliqué. Cadrez la plaque entière sur un fond contrasté.",
                severity=SEVERITY_INFO,
            )
        )
    if coverage < COVERAGE_MIN:
        warnings.append(
            QualityWarning(
                "low_coverage",
                "La plaque n'occupe qu'une petite partie du cadre. "
                "Rapprochez l'appareil pour que la plaque remplisse l'image.",
            )
        )
    if skew is not None and skew > SKEW_MAX:
        warnings.append(
            QualityWarning(
                "perspective",
                "La plaque est vue de biais. Placez l'appareil perpendiculairement "
                "à la plaque pour limiter la déformation.",
            )
        )
    if density < EDGE_DENSITY_MIN:
        warnings.append(
            QualityWarning(
                "no_pattern_detected",
                "Aucun motif net n'est visible. Vérifiez la présence de sel sur la "
                "plaque et attendez la stabilisation des lignes nodales.",
            )
        )

    metrics = {
        "sharpness": round(sharpness, 3),
        "contrast": round(contrast, 3),
        "dark_ratio": round(dark_ratio, 4),
        "bright_ratio": round(bright_ratio, 4),
        "edge_density": round(density, 4),
        "coverage": round(coverage, 4),
    }
    if skew is not None:
        metrics["skew"] = round(skew, 4)

    return QualityReport(score=_score(metrics, plate_detected), warnings=warnings, metrics=metrics)


def _ratio(value: float, target: float) -> float:
    """Sous-score dans [0, 1] : 1 dès que ``value`` atteint ``target``."""
    if target <= 0:
        return 1.0
    return float(min(1.0, max(0.0, value / target)))


def _score(metrics: dict[str, float], plate_detected: bool) -> float:
    """Agrège les mesures en une note unique dans [0, 1]."""
    sub_scores = [
        (_ratio(metrics["sharpness"], BLUR_VARIANCE_MIN), 0.25),
        (_ratio(metrics["contrast"], CONTRAST_STD_MIN), 0.20),
        (_ratio(metrics["edge_density"], EDGE_DENSITY_MIN), 0.20),
        (_ratio(metrics["coverage"], COVERAGE_MIN), 0.15),
        (1.0 - min(1.0, metrics["bright_ratio"] / OVEREXPOSED_RATIO_MAX), 0.10),
        (1.0 - min(1.0, metrics["dark_ratio"] / UNDEREXPOSED_RATIO_MAX), 0.05),
    ]
    skew = metrics.get("skew")
    skew_score = (
        1.0 - min(1.0, skew / SKEW_MAX) if skew is not None else (1.0 if plate_detected else 0.5)
    )
    sub_scores.append((skew_score, 0.05))

    total_weight = sum(weight for _, weight in sub_scores)
    weighted = sum(value * weight for value, weight in sub_scores)
    return round(float(weighted / total_weight), 4)
