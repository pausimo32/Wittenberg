"""Valide un manifeste et, optionnellement, le modèle qu'il décrit.

À exécuter après chaque nouvel entraînement, avant de déployer ::

    python -m wittenberg.tools.validate_manifest --manifest models/model_manifest.json
    python -m wittenberg.tools.validate_manifest --manifest models/model_manifest.json --write-hash
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from wittenberg.config import Settings
from wittenberg.model.manifest import ManifestError, compute_sha256, load_manifest
from wittenberg.model.service import ModelService, ModelStatus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wittenberg-validate-manifest",
        description="Contrôle la cohérence d'un manifeste et de son artefact.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument(
        "--write-hash",
        action="store_true",
        help="Calcule le SHA-256 de l'artefact et l'inscrit dans le manifeste.",
    )
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="Valide uniquement le schéma, sans charger le modèle.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)
    except ManifestError as exc:
        print(f"ÉCHEC — {exc}", file=sys.stderr)
        return 2

    print(f"Schéma valide : {manifest.num_classes} classes, version {manifest.model.version}")
    for entry in manifest.ordered_classes:
        print(f"  [{entry.index}] {entry.label} → {entry.frequency_hz} Hz")

    artifact = args.model or manifest.artifact_path
    if not artifact.exists():
        print(f"\nArtefact absent : {artifact}", file=sys.stderr)
        return 3

    if args.write_hash:
        digest = compute_sha256(artifact)
        raw = json.loads(args.manifest.read_text(encoding="utf-8"))
        raw["model"]["sha256"] = digest
        args.manifest.write_text(
            json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"\nSHA-256 inscrit dans le manifeste : {digest}")
        manifest = load_manifest(args.manifest)

    if args.skip_load:
        return 0

    settings = Settings(
        model_manifest_path=args.manifest, model_path=args.model, warmup_on_startup=True
    )
    service = ModelService(settings)
    state = service.load()
    if state.status is not ModelStatus.READY:
        print(f"\nÉCHEC du chargement — {state.reason}", file=sys.stderr)
        return 4

    print(
        f"\nModèle chargé et préchauffé : runtime={service.runtime_version}, "
        f"warmup={service.warmup_ms} ms"
    )
    print("Manifeste et modèle cohérents.")
    service.shutdown()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
