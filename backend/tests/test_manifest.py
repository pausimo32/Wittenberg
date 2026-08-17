"""Le manifeste est la source de vérité : il doit refuser toute incohérence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conftest import manifest_dict, write_manifest
from wittenberg.model.manifest import ManifestError, compute_sha256, load_manifest


def test_valid_manifest_loads(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, manifest_dict())
    manifest = load_manifest(path)

    assert manifest.num_classes == 4
    assert manifest.model.version == "test-1.0.0"
    assert manifest.input.tensor_shape == (1, 32, 32, 1)
    assert manifest.source_path == path.resolve()


def test_class_order_comes_from_index_not_file_order(tmp_path: Path) -> None:
    """L'ordre du tableau JSON ne doit rien décider : seul `index` compte."""
    shuffled = manifest_dict(
        classes=[
            {"index": 2, "label": "333 Hz", "frequency_hz": 333.0},
            {"index": 0, "label": "111 Hz", "frequency_hz": 111.0},
            {"index": 3, "label": "444 Hz", "frequency_hz": 444.0},
            {"index": 1, "label": "222 Hz", "frequency_hz": 222.0},
        ]
    )
    manifest = load_manifest(write_manifest(tmp_path, shuffled))

    assert [c.label for c in manifest.ordered_classes] == [
        "111 Hz",
        "222 Hz",
        "333 Hz",
        "444 Hz",
    ]
    assert manifest.class_at(2).frequency_hz == 333.0


def test_duplicate_index_is_rejected(tmp_path: Path) -> None:
    payload = manifest_dict(
        classes=[
            {"index": 0, "label": "a", "frequency_hz": 1.0},
            {"index": 0, "label": "b", "frequency_hz": 2.0},
        ]
    )
    with pytest.raises(ManifestError, match="index"):
        load_manifest(write_manifest(tmp_path, payload))


def test_missing_index_is_rejected(tmp_path: Path) -> None:
    payload = manifest_dict(
        classes=[
            {"index": 0, "label": "a", "frequency_hz": 1.0},
            {"index": 2, "label": "b", "frequency_hz": 2.0},
        ]
    )
    with pytest.raises(ManifestError, match="index"):
        load_manifest(write_manifest(tmp_path, payload))


def test_duplicate_frequency_is_rejected(tmp_path: Path) -> None:
    payload = manifest_dict(
        classes=[
            {"index": 0, "label": "a", "frequency_hz": 440.0},
            {"index": 1, "label": "b", "frequency_hz": 440.0},
        ]
    )
    with pytest.raises(ManifestError, match="fréquences"):
        load_manifest(write_manifest(tmp_path, payload))


def test_framework_format_mismatch_is_rejected(tmp_path: Path) -> None:
    payload = manifest_dict(model={"framework": "onnx", "format": "keras"})
    with pytest.raises(ManifestError, match="incompatible"):
        load_manifest(write_manifest(tmp_path, payload))


def test_top_k_cannot_exceed_class_count(tmp_path: Path) -> None:
    payload = manifest_dict(decision={"top_k": 99})
    with pytest.raises(ManifestError, match="top_k"):
        load_manifest(write_manifest(tmp_path, payload))


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    """Une faute de frappe ne doit pas passer silencieusement."""
    payload = manifest_dict()
    payload["preprocessing"]["greyscale"] = True
    with pytest.raises(ManifestError):
        load_manifest(write_manifest(tmp_path, payload))


def test_mean_std_requires_both(tmp_path: Path) -> None:
    payload = manifest_dict(input={"normalization": {"mode": "mean_std", "mean": [0.5]}})
    with pytest.raises(ManifestError, match="mean_std"):
        load_manifest(write_manifest(tmp_path, payload))


def test_future_manifest_version_is_rejected(tmp_path: Path) -> None:
    payload = manifest_dict(manifest_version=99)
    with pytest.raises(ManifestError, match="manifest_version"):
        load_manifest(write_manifest(tmp_path, payload))


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="introuvable"):
        load_manifest(tmp_path / "absent.json")


def test_invalid_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "model_manifest.json"
    path.write_text("{ pas du json", encoding="utf-8")
    with pytest.raises(ManifestError, match="JSON invalide"):
        load_manifest(path)


def test_artifact_path_is_relative_to_manifest(tmp_path: Path) -> None:
    nested = tmp_path / "models"
    nested.mkdir()
    path = write_manifest(nested, manifest_dict())
    manifest = load_manifest(path)

    assert manifest.artifact_path == (nested / "model.onnx").resolve()


def test_sha256_is_stable(tmp_path: Path) -> None:
    target = tmp_path / "artifact.bin"
    target.write_bytes(b"wittenberg")

    assert compute_sha256(target) == compute_sha256(target)
    assert len(compute_sha256(target)) == 64


def test_shipped_example_manifest_is_valid() -> None:
    """Le gabarit livré doit rester conforme au schéma."""
    example = Path(__file__).resolve().parents[1] / "models" / "model_manifest.example.json"
    manifest = load_manifest(example)

    assert manifest.num_classes >= 2
    # Le gabarit ne doit jamais être confondu avec un modèle réel.
    assert "exemple" in manifest.model.version


def test_example_manifest_does_not_hardcode_project_frequencies() -> None:
    """Aucune fréquence « plausible » ne doit être livrée par défaut.

    Les classes doivent provenir des mesures réelles, jamais d'un gabarit ni
    d'un article de référence.
    """
    example = Path(__file__).resolve().parents[1] / "models" / "model_manifest.example.json"
    payload = json.loads(example.read_text(encoding="utf-8"))
    frequencies = {c["frequency_hz"] for c in payload["classes"]}

    assert frequencies.isdisjoint({330.0, 440.0, 587.0, 660.0, 880.0, 1100.0})
