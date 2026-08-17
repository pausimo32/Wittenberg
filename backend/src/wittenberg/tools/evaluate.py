"""Évaluation du modèle sur un jeu de validation.

Calcule les métriques qui qualifient le modèle dans son ensemble — à ne jamais
confondre avec la ``confidence`` d'une prédiction isolée.

Structure attendue ::

    validation/
      330Hz/  img_001.jpg  …
      440Hz/  img_014.jpg  …

Le nom du dossier est rapproché du manifeste par libellé exact puis par
fréquence. Aucun ordre alphabétique n'est utilisé pour attribuer un index :
l'index provient toujours du manifeste.

Exemple ::

    python -m wittenberg.tools.evaluate \\
        --manifest models/model_manifest.json --data data/validation
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from wittenberg.config import Settings
from wittenberg.imaging.pipeline import preprocess_bytes
from wittenberg.imaging.validation import ImageValidationError
from wittenberg.model.decision import decide, to_probabilities
from wittenberg.model.manifest import ClassEntry, ManifestError, ModelManifest, load_manifest
from wittenberg.model.service import ModelService, ModelStatus

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"})


@dataclass
class Sample:
    path: Path
    true_index: int


def _normalize(name: str) -> str:
    return re.sub(r"[^0-9a-z]", "", name.lower())


def resolve_class(folder: str, manifest: ModelManifest) -> ClassEntry | None:
    """Associe un nom de dossier à une classe du manifeste.

    Essaie le libellé exact, puis une comparaison normalisée, puis la valeur
    numérique de la fréquence (« 440Hz », « 440 Hz », « f440 » → 440).
    """
    for entry in manifest.ordered_classes:
        if entry.label == folder:
            return entry
    target = _normalize(folder)
    for entry in manifest.ordered_classes:
        if _normalize(entry.label) == target:
            return entry
    numbers = re.findall(r"\d+(?:[.,]\d+)?", folder)
    if len(numbers) == 1:
        value = float(numbers[0].replace(",", "."))
        for entry in manifest.ordered_classes:
            if abs(entry.frequency_hz - value) < 1e-6:
                return entry
    return None


def collect_samples(root: Path, manifest: ModelManifest) -> tuple[list[Sample], list[str]]:
    """Parcourt l'arborescence et renvoie les échantillons et les dossiers ignorés."""
    samples: list[Sample] = []
    unmatched: list[str] = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        entry = resolve_class(folder.name, manifest)
        if entry is None:
            unmatched.append(folder.name)
            continue
        for image in sorted(folder.rglob("*")):
            if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES:
                samples.append(Sample(path=image, true_index=entry.index))
    return samples, unmatched


def session_key(path: Path, pattern: str | None) -> str | None:
    """Extrait un identifiant de session depuis le chemin, si un motif est fourni."""
    if not pattern:
        return None
    match = re.search(pattern, path.as_posix())
    if not match:
        return None
    return match.group(1) if match.groups() else match.group(0)


def check_leakage(
    validation: list[Sample], train_root: Path | None, pattern: str | None
) -> dict[str, Any]:
    """Cherche des indices de fuite entre entraînement et validation.

    Deux signaux : un nom de fichier identique des deux côtés, et une session
    expérimentale présente des deux côtés. Les photographies d'une même session
    se ressemblent fortement ; les répartir entre les deux jeux surestime les
    performances.
    """
    report: dict[str, Any] = {"checked": train_root is not None}
    if train_root is None or not train_root.is_dir():
        return report

    train_files = [
        p for p in train_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    train_names = {p.name for p in train_files}
    shared_names = sorted({s.path.name for s in validation if s.path.name in train_names})

    shared_sessions: list[str] = []
    if pattern:
        train_sessions = {k for k in (session_key(p, pattern) for p in train_files) if k}
        val_sessions = {k for k in (session_key(s.path, pattern) for s in validation) if k}
        shared_sessions = sorted(train_sessions & val_sessions)

    report.update(
        {
            "shared_filenames": shared_names[:50],
            "shared_filenames_count": len(shared_names),
            "shared_sessions": shared_sessions[:50],
            "shared_sessions_count": len(shared_sessions),
            "leakage_suspected": bool(shared_names or shared_sessions),
        }
    )
    return report


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    ranked: list[list[int]],
    confidences: list[float],
    manifest: ModelManifest,
    *,
    top_k: int,
    min_confidence: float,
) -> dict[str, Any]:
    """Accuracy, top-k accuracy, macro-F1, métriques par classe, matrice de confusion."""
    num_classes = manifest.num_classes
    total = len(y_true)
    if total == 0:
        return {"error": "aucun échantillon évalué"}

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_index, pred_index in zip(y_true, y_pred, strict=True):
        confusion[true_index, pred_index] += 1

    correct = int(np.trace(confusion))
    accuracy = correct / total
    topk_correct = sum(
        1 for true_index, candidates in zip(y_true, ranked, strict=True) if true_index in candidates
    )

    per_class: list[dict[str, Any]] = []
    f1_scores: list[float] = []
    for entry in manifest.ordered_classes:
        index = entry.index
        true_positive = int(confusion[index, index])
        false_positive = int(confusion[:, index].sum() - true_positive)
        false_negative = int(confusion[index, :].sum() - true_positive)
        support = int(confusion[index, :].sum())
        precision = (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive)
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative)
            else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        # Une classe absente du jeu de validation ne doit pas tirer le macro-F1
        # vers le bas : elle n'a simplement pas été mesurée.
        if support > 0:
            f1_scores.append(f1)
        per_class.append(
            {
                "index": index,
                "label": entry.label,
                "frequency_hz": entry.frequency_hz,
                "support": support,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }
        )

    low_confidence = [c for c in confidences if c < min_confidence]
    return {
        "samples": total,
        "accuracy": round(accuracy, 4),
        f"top_{top_k}_accuracy": round(topk_correct / total, 4),
        "macro_f1": round(float(np.mean(f1_scores)) if f1_scores else 0.0, 4),
        "classes_evaluated": len(f1_scores),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": [c.label for c in manifest.ordered_classes],
            "rows_are_true_classes": True,
            "matrix": confusion.tolist(),
        },
        "confidence": {
            "mean": round(float(np.mean(confidences)), 4),
            "median": round(float(np.median(confidences)), 4),
            "min": round(float(np.min(confidences)), 4),
            "below_threshold": len(low_confidence),
            "below_threshold_ratio": round(len(low_confidence) / total, 4),
            "threshold": min_confidence,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wittenberg-evaluate",
        description="Évalue le modèle sur un jeu de validation étiqueté.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="Dossier de validation.")
    parser.add_argument("--model", type=Path, default=None, help="Artefact ; sinon le manifeste.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None, help="Écrit le rapport JSON.")
    parser.add_argument(
        "--train-data",
        type=Path,
        default=None,
        help="Dossier d'entraînement, pour contrôler l'absence de fuite de données.",
    )
    parser.add_argument(
        "--session-pattern",
        type=str,
        default=None,
        help=(
            "Expression régulière extrayant l'identifiant de session du chemin, "
            r"par exemple 'session[-_]?(\d+)'."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)
    except ManifestError as exc:
        print(f"Manifeste invalide : {exc}", file=sys.stderr)
        return 2

    if not args.data.is_dir():
        print(f"Dossier de validation introuvable : {args.data}", file=sys.stderr)
        return 2

    settings = Settings(
        model_manifest_path=args.manifest,
        model_path=args.model,
        warmup_on_startup=True,
    )
    service = ModelService(settings)
    state = service.load()
    if state.status is not ModelStatus.READY:
        print(f"Modèle indisponible : {state.reason}", file=sys.stderr)
        return 3

    samples, unmatched = collect_samples(args.data, manifest)
    if not samples:
        print("Aucune image exploitable trouvée dans le jeu de validation.", file=sys.stderr)
        return 4

    y_true: list[int] = []
    y_pred: list[int] = []
    ranked: list[list[int]] = []
    confidences: list[float] = []
    failures: list[dict[str, str]] = []
    per_class_counts: dict[str, int] = defaultdict(int)

    for sample in samples:
        try:
            result, _ = preprocess_bytes(
                sample.path.read_bytes(),
                manifest,
                max_bytes=settings.max_upload_bytes,
                max_pixels=settings.max_image_pixels,
                max_dimension=settings.max_image_dimension,
                min_dimension=settings.min_image_dimension,
            )
            scores = service.infer_sync(result.tensor)
        except (ImageValidationError, OSError, ValueError) as exc:
            failures.append({"file": sample.path.name, "error": type(exc).__name__})
            continue

        probabilities = to_probabilities(scores, manifest.output.activation)
        decision = decide(probabilities, manifest, top_k=args.top_k)
        y_true.append(sample.true_index)
        y_pred.append(decision.best.index)
        ranked.append([entry.index for entry in decision.top_k])
        confidences.append(decision.best.probability)
        per_class_counts[manifest.class_at(sample.true_index).label] += 1

    metrics = compute_metrics(
        y_true,
        y_pred,
        ranked,
        confidences,
        manifest,
        top_k=args.top_k,
        min_confidence=manifest.decision.min_confidence,
    )

    report = {
        "model": {"name": manifest.model.name, "version": manifest.model.version},
        "preprocessing_version": manifest.preprocessing.version,
        "dataset": {
            "root": str(args.data),
            "images_per_class": dict(sorted(per_class_counts.items())),
            "unmatched_folders": unmatched,
            "unreadable_files": failures,
        },
        "leakage_check": check_leakage(samples, args.train_data, args.session_pattern),
        "metrics": metrics,
    }

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)

    if unmatched:
        print(
            f"\nAttention : dossiers ignorés (aucune classe correspondante) : {unmatched}",
            file=sys.stderr,
        )
    leakage = report["leakage_check"]
    if isinstance(leakage, dict) and leakage.get("leakage_suspected"):
        print(
            "\nAttention : fuite de données probable entre entraînement et validation.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
