"""Outils hors API : prétraitement en ligne de commande et évaluation."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import manifest_dict, write_manifest
from tests.support import images
from wittenberg.model.manifest import load_manifest
from wittenberg.tools import evaluate, preprocess_cli, validate_manifest

# ── preprocess_cli ───────────────────────────────────────────────────────


def test_preprocess_cli_reports_pipeline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """La même pipeline doit être exécutable hors de l'API."""
    manifest = write_manifest(tmp_path, manifest_dict())
    photo = tmp_path / "motif.png"
    photo.write_bytes(images.encode(images.chladni_like(256), "PNG"))

    code = preprocess_cli.main([str(photo), "--manifest", str(manifest)])
    report = json.loads(capsys.readouterr().out)

    assert code == 0
    assert report["tensor"]["shape"] == [1, 32, 32, 1]
    assert report["tensor"]["dtype"] == "float32"
    assert report["preprocessing_version"] == "prep-test-1"
    assert report["steps"]


def test_preprocess_cli_saves_preview_and_tensor(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path, manifest_dict())
    photo = tmp_path / "motif.png"
    photo.write_bytes(images.encode(images.chladni_like(256), "PNG"))
    preview = tmp_path / "vue.png"
    tensor = tmp_path / "tenseur.npy"

    preprocess_cli.main(
        [
            str(photo),
            "--manifest",
            str(manifest),
            "--save-preview",
            str(preview),
            "--save-tensor",
            str(tensor),
        ]
    )

    assert preview.stat().st_size > 0
    assert np.load(tensor).shape == (1, 32, 32, 1)


def test_preprocess_cli_rejects_non_image(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = write_manifest(tmp_path, manifest_dict())
    bogus = tmp_path / "faux.png"
    bogus.write_bytes(b"pas une image" * 10)

    code = preprocess_cli.main([str(bogus), "--manifest", str(manifest)])

    assert code == 1
    assert "refusée" in capsys.readouterr().err


# ── validate_manifest ────────────────────────────────────────────────────


def test_validate_manifest_accepts_coherent_pair(
    manifest_factory: Callable[..., Path], capsys: pytest.CaptureFixture[str]
) -> None:
    code = validate_manifest.main(["--manifest", str(manifest_factory())])

    assert code == 0
    assert "cohérents" in capsys.readouterr().out


def test_validate_manifest_writes_hash(manifest_factory: Callable[..., Path]) -> None:
    path = manifest_factory(include_hash=False)
    assert json.loads(path.read_text(encoding="utf-8"))["model"].get("sha256") is None

    code = validate_manifest.main(["--manifest", str(path), "--write-hash"])
    digest = json.loads(path.read_text(encoding="utf-8"))["model"]["sha256"]

    assert code == 0
    assert digest and len(digest) == 64


def test_validate_manifest_detects_incoherence(manifest_factory: Callable[..., Path]) -> None:
    path = manifest_factory()
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["classes"] = payload["classes"][:2]
    payload["decision"]["top_k"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert validate_manifest.main(["--manifest", str(path)]) == 4


# ── evaluate ─────────────────────────────────────────────────────────────


def test_resolve_class_matches_label_then_frequency(tmp_path: Path) -> None:
    manifest = load_manifest(write_manifest(tmp_path, manifest_dict()))

    assert evaluate.resolve_class("111 Hz", manifest).index == 0  # type: ignore[union-attr]
    assert evaluate.resolve_class("111Hz", manifest).index == 0  # type: ignore[union-attr]
    assert evaluate.resolve_class("222", manifest).index == 1  # type: ignore[union-attr]
    assert evaluate.resolve_class("f333", manifest).index == 2  # type: ignore[union-attr]
    assert evaluate.resolve_class("inconnu", manifest) is None


def test_resolve_class_never_uses_alphabetical_order(tmp_path: Path) -> None:
    """Un dossier sans correspondance ne doit jamais se voir attribuer un index."""
    manifest = load_manifest(write_manifest(tmp_path, manifest_dict()))

    assert evaluate.resolve_class("aaa_premier_dossier", manifest) is None


def test_collect_samples_walks_class_folders(tmp_path: Path) -> None:
    manifest = load_manifest(write_manifest(tmp_path, manifest_dict()))
    root = tmp_path / "validation"
    for name, count in (("111 Hz", 2), ("222Hz", 3), ("dossier_inconnu", 1)):
        folder = root / name
        folder.mkdir(parents=True)
        for i in range(count):
            (folder / f"img_{i}.png").write_bytes(images.encode(images.chladni_like(64), "PNG"))

    samples, unmatched = evaluate.collect_samples(root, manifest)

    assert len(samples) == 5
    assert unmatched == ["dossier_inconnu"]


def test_metrics_are_computed_correctly(tmp_path: Path) -> None:
    manifest = load_manifest(write_manifest(tmp_path, manifest_dict()))
    y_true = [0, 0, 1, 1, 2, 3]
    y_pred = [0, 1, 1, 1, 2, 2]
    ranked = [[0, 1], [1, 0], [1, 2], [1, 0], [2, 3], [2, 3]]
    confidences = [0.9, 0.4, 0.8, 0.7, 0.95, 0.5]

    metrics = evaluate.compute_metrics(
        y_true, y_pred, ranked, confidences, manifest, top_k=2, min_confidence=0.55
    )

    assert metrics["samples"] == 6
    assert metrics["accuracy"] == pytest.approx(4 / 6, abs=1e-4)
    assert metrics["top_2_accuracy"] == pytest.approx(6 / 6, abs=1e-4)
    assert metrics["confidence"]["below_threshold"] == 2
    assert metrics["classes_evaluated"] == 4
    # Matrice : lignes = classes réelles.
    assert metrics["confusion_matrix"]["matrix"][0] == [1, 1, 0, 0]


def test_macro_f1_ignores_classes_absent_from_validation(tmp_path: Path) -> None:
    manifest = load_manifest(write_manifest(tmp_path, manifest_dict()))
    metrics = evaluate.compute_metrics(
        [0, 0], [0, 0], [[0], [0]], [0.9, 0.9], manifest, top_k=1, min_confidence=0.5
    )

    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["classes_evaluated"] == 1


def test_leakage_detection_flags_shared_filenames(tmp_path: Path) -> None:
    train = tmp_path / "train" / "111 Hz"
    train.mkdir(parents=True)
    (train / "session3_img7.png").write_bytes(b"x")
    validation = [evaluate.Sample(path=Path("validation/111 Hz/session3_img7.png"), true_index=0)]

    report = evaluate.check_leakage(validation, tmp_path / "train", None)

    assert report["leakage_suspected"] is True
    assert report["shared_filenames_count"] == 1


def test_leakage_detection_flags_shared_sessions(tmp_path: Path) -> None:
    """Deux clichés d'une même session ne doivent pas se répartir entre les jeux."""
    train = tmp_path / "train" / "111 Hz"
    train.mkdir(parents=True)
    (train / "session_12_a.png").write_bytes(b"x")
    validation = [evaluate.Sample(path=Path("validation/111 Hz/session_12_b.png"), true_index=0)]

    report = evaluate.check_leakage(validation, tmp_path / "train", r"session_(\d+)")

    assert report["shared_sessions"] == ["12"]
    assert report["leakage_suspected"] is True


def test_leakage_check_is_clean_when_splits_are_disjoint(tmp_path: Path) -> None:
    train = tmp_path / "train" / "111 Hz"
    train.mkdir(parents=True)
    (train / "session_01_a.png").write_bytes(b"x")
    validation = [evaluate.Sample(path=Path("validation/111 Hz/session_09_b.png"), true_index=0)]

    report = evaluate.check_leakage(validation, tmp_path / "train", r"session_(\d+)")

    assert report["leakage_suspected"] is False


def test_evaluate_end_to_end(
    manifest_factory: Callable[..., Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Le script complet doit produire un rapport exploitable."""
    manifest_path = manifest_factory()
    root = tmp_path / "validation"
    for label in ("111 Hz", "222 Hz"):
        folder = root / label
        folder.mkdir(parents=True)
        for i in range(2):
            (folder / f"img_{i}.png").write_bytes(
                images.encode(images.chladni_like(128, m=2 + i, n=3), "PNG")
            )

    code = evaluate.main(["--manifest", str(manifest_path), "--data", str(root)])
    report = json.loads(capsys.readouterr().out)

    assert code == 0
    assert report["metrics"]["samples"] == 4
    assert set(report["metrics"]) >= {
        "accuracy",
        "macro_f1",
        "per_class",
        "confusion_matrix",
        "confidence",
    }
    assert report["dataset"]["images_per_class"] == {"111 Hz": 2, "222 Hz": 2}


def test_evaluate_refuses_without_model(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path, manifest_dict())
    root = tmp_path / "validation" / "111 Hz"
    root.mkdir(parents=True)

    assert evaluate.main(["--manifest", str(manifest), "--data", str(root.parent)]) == 3
