"""Exécute le pipeline de prétraitement hors de l'API.

Sert à vérifier visuellement ce que « voit » le modèle, et à réutiliser
exactement la même pipeline pour préparer un jeu de données.

Exemples ::

    python -m wittenberg.tools.preprocess_cli \\
        --manifest models/model_manifest.json photo.jpg

    python -m wittenberg.tools.preprocess_cli \\
        --manifest models/model_manifest.json photo.jpg --save-preview vue.png
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

from wittenberg.imaging.pipeline import preprocess_bytes
from wittenberg.imaging.validation import ImageValidationError
from wittenberg.model.manifest import ManifestError, load_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wittenberg-preprocess",
        description="Applique le pipeline de prétraitement à une image.",
    )
    parser.add_argument("image", type=Path, help="Image à prétraiter.")
    parser.add_argument(
        "--manifest", type=Path, required=True, help="Manifeste décrivant le prétraitement."
    )
    parser.add_argument(
        "--save-preview", type=Path, default=None, help="Écrit l'image prétraitée en PNG."
    )
    parser.add_argument(
        "--save-tensor", type=Path, default=None, help="Écrit le tenseur final au format .npy."
    )
    parser.add_argument("--max-mb", type=float, default=10.0)
    parser.add_argument("--max-pixels", type=int, default=40_000_000)
    parser.add_argument("--max-dimension", type=int, default=12_000)
    parser.add_argument("--min-dimension", type=int, default=32)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)
    except ManifestError as exc:
        print(f"Manifeste invalide : {exc}", file=sys.stderr)
        return 2

    if not args.image.is_file():
        print(f"Image introuvable : {args.image}", file=sys.stderr)
        return 2

    data = args.image.read_bytes()
    try:
        result, decoded = preprocess_bytes(
            data,
            manifest,
            max_bytes=int(args.max_mb * 1024 * 1024),
            max_pixels=args.max_pixels,
            max_dimension=args.max_dimension,
            min_dimension=args.min_dimension,
            debug_preview=args.save_preview is not None,
        )
    except ImageValidationError as exc:
        print(f"Image refusée [{exc.code}] : {exc.message}", file=sys.stderr)
        return 1

    if args.save_preview and result.debug_preview:
        args.save_preview.write_bytes(base64.b64decode(result.debug_preview))
    if args.save_tensor:
        np.save(args.save_tensor, result.tensor)

    report = {
        "source": {
            "format": decoded.detected_format.value,
            "size": [decoded.original_width, decoded.original_height],
            "mode": decoded.original_mode,
            "exif_transposed": decoded.exif_transposed,
            "had_alpha": decoded.had_alpha,
        },
        "preprocessing_version": manifest.preprocessing.version,
        "steps": result.steps,
        "plate": {
            "method": result.plate.method,
            "detected": result.plate.detected,
            "coverage": round(result.plate.coverage, 4),
            "skew": result.plate.skew,
        },
        "quality": {
            "score": result.quality.score,
            "warnings": [
                {"code": w.code, "severity": w.severity, "message": w.message}
                for w in result.quality.warnings
            ],
            "metrics": result.quality.metrics,
        },
        "tensor": {
            "shape": list(result.tensor.shape),
            "dtype": str(result.tensor.dtype),
            "min": float(result.tensor.min()),
            "max": float(result.tensor.max()),
            "mean": round(float(result.tensor.mean()), 6),
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
