"""Localisation de la plaque et correction de perspective.

La plaque est un carré d'aluminium de 20 × 20 cm photographié par une caméra
fixe, approximativement perpendiculaire. Lorsqu'un quadrilatère plausible est
détecté, l'image est redressée par homographie vers un carré.

Si la détection échoue — ou si OpenCV n'est pas installé — aucune homographie
n'est inventée : on applique un recadrage central déterministe et on signale
l'incertitude par un avertissement de qualité.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from wittenberg.model.manifest import PlateDetectionSection


def opencv_available() -> bool:
    """Indique si OpenCV est importable."""
    return _import_cv2() is not None


def _import_cv2() -> Any | None:
    try:
        import cv2
    except ImportError:  # pragma: no cover - dépend de l'installation
        return None
    return cv2


@dataclass(frozen=True)
class PlateResult:
    """Résultat de la localisation de la plaque."""

    #: image RGB redressée ou recadrée, en uint8
    region: np.ndarray
    #: 'homography' si la plaque a été détectée et redressée, 'center_crop' sinon
    method: str
    #: fraction de l'image d'origine occupée par la région retenue
    coverage: float
    #: écart au rectangle parfait, dans [0, 1] ; None si non mesurable
    skew: float | None
    detected: bool


def locate_plate(rgb: np.ndarray, config: PlateDetectionSection) -> PlateResult:
    """Extrait la région utile de l'image.

    :param rgb: image RGB uint8 de forme ``(H, W, 3)``.
    """
    if config.enabled:
        cv2 = _import_cv2()
        if cv2 is not None:
            quad = _find_plate_quad(cv2, rgb, config)
            if quad is not None:
                skew = _quad_skew(quad)
                coverage = _quad_area(quad) / float(rgb.shape[0] * rgb.shape[1])
                if skew <= config.max_skew and coverage >= config.min_area_ratio:
                    return PlateResult(
                        region=_warp_to_square(cv2, rgb, quad),
                        method="homography",
                        coverage=min(coverage, 1.0),
                        skew=skew,
                        detected=True,
                    )

    cropped = center_crop(rgb, config.fallback_center_crop)
    coverage = (cropped.shape[0] * cropped.shape[1]) / float(rgb.shape[0] * rgb.shape[1])
    return PlateResult(
        region=cropped,
        method="center_crop",
        coverage=coverage,
        skew=None,
        detected=False,
    )


def center_crop(rgb: np.ndarray, ratio: float) -> np.ndarray:
    """Recadrage central carré déterministe.

    Le côté vaut ``ratio`` fois la plus petite dimension de l'image.
    """
    height, width = rgb.shape[:2]
    side = max(1, round(min(height, width) * ratio))
    top = (height - side) // 2
    left = (width - side) // 2
    return rgb[top : top + side, left : left + side]


def _find_plate_quad(cv2: Any, rgb: np.ndarray, config: PlateDetectionSection) -> np.ndarray | None:
    """Cherche le plus grand quadrilatère convexe plausible.

    La plaque est claire sur fond sombre : un seuillage d'Otsu suivi d'une
    recherche de contours suffit dans les conditions expérimentales décrites.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Refermer les lignes nodales sombres pour obtenir une plaque pleine.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    image_area = float(rgb.shape[0] * rgb.shape[1])
    min_area = config.min_area_ratio * image_area

    best: np.ndarray | None = None
    best_area = 0.0
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        area = float(cv2.contourArea(contour))
        if area < min_area or area <= best_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        best = approx.reshape(4, 2).astype(np.float32)
        best_area = area

    return best


def order_corners(quad: np.ndarray) -> np.ndarray:
    """Ordonne les sommets : haut-gauche, haut-droit, bas-droit, bas-gauche."""
    points = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    total = points.sum(axis=1)
    diff = np.diff(points, axis=1).ravel()
    return np.array(
        [
            points[int(np.argmin(total))],  # somme minimale → haut-gauche
            points[int(np.argmin(diff))],  # x - y maximal  → haut-droit
            points[int(np.argmax(total))],  # somme maximale → bas-droit
            points[int(np.argmax(diff))],  # y - x maximal  → bas-gauche
        ],
        dtype=np.float32,
    )


def _quad_area(quad: np.ndarray) -> float:
    """Aire du quadrilatère par la formule du lacet."""
    points = order_corners(quad)
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _quad_skew(quad: np.ndarray) -> float:
    """Mesure l'écart au rectangle, dans [0, 1].

    Combine le déséquilibre des côtés opposés et celui des diagonales. Vaut 0
    pour un rectangle vu de face et croît avec l'inclinaison de la caméra.
    """
    points = order_corners(quad)
    sides = [float(np.linalg.norm(points[(i + 1) % 4] - points[i])) for i in range(4)]
    top, right, bottom, left = sides
    diag1 = float(np.linalg.norm(points[2] - points[0]))
    diag2 = float(np.linalg.norm(points[3] - points[1]))

    def imbalance(a: float, b: float) -> float:
        total = a + b
        return abs(a - b) / total if total > 0 else 1.0

    return float(
        min(
            1.0,
            max(imbalance(top, bottom), imbalance(left, right), imbalance(diag1, diag2)),
        )
    )


def _warp_to_square(cv2: Any, rgb: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Redresse le quadrilatère vers un carré par homographie.

    Le côté cible est la plus grande longueur mesurée, ce qui évite de perdre
    de la résolution ; la plaque étant carrée, le carré est la cible correcte.
    """
    points = order_corners(quad)
    widths = [
        float(np.linalg.norm(points[1] - points[0])),
        float(np.linalg.norm(points[2] - points[3])),
    ]
    heights = [
        float(np.linalg.norm(points[3] - points[0])),
        float(np.linalg.norm(points[2] - points[1])),
    ]
    side = round(max(*widths, *heights))
    side = max(2, side)

    destination = np.array(
        [[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(points, destination)
    warped: np.ndarray = cv2.warpPerspective(
        rgb, matrix, (side, side), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE
    )
    return warped
