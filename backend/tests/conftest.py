"""Fixtures partagées.

Un adaptateur factice n'existe que dans ces tests. La production ne peut jamais
démarrer sur des prédictions simulées : ``ModelService`` n'accepte que les
adaptateurs construits à partir d'un manifeste et d'un artefact réels.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.support.onnx_factory import make_cnn
from wittenberg.app import create_app
from wittenberg.config import Settings
from wittenberg.model.adapters.base import ModelAdapter
from wittenberg.model.manifest import compute_sha256

DEFAULT_CLASSES = [
    {"index": 0, "label": "111 Hz", "frequency_hz": 111.0},
    {"index": 1, "label": "222 Hz", "frequency_hz": 222.0},
    {"index": 2, "label": "333 Hz", "frequency_hz": 333.0},
    {"index": 3, "label": "444 Hz", "frequency_hz": 444.0},
]


def manifest_dict(**overrides: Any) -> dict[str, Any]:
    """Manifeste minimal valide, surchargeable section par section."""
    base: dict[str, Any] = {
        "manifest_version": 1,
        "model": {
            "name": "test-cnn",
            "version": "test-1.0.0",
            "framework": "onnx",
            "format": "onnx",
            "artifact": "model.onnx",
        },
        "input": {
            "height": 32,
            "width": 32,
            "channels": 1,
            "layout": "NHWC",
            "dtype": "float32",
            "value_range": [0.0, 1.0],
            "normalization": {"mode": "scale_01"},
        },
        "preprocessing": {
            "version": "prep-test-1",
            "grayscale": True,
            "plate_detection": {"enabled": True, "fallback_center_crop": 0.9},
            "contrast": {"mode": "minmax"},
            "denoise": {"mode": "none"},
            "edge_detection": {"enabled": False},
            "resize_mode": "pad",
        },
        "output": {"activation": "softmax"},
        "classes": [dict(entry) for entry in DEFAULT_CLASSES],
        "decision": {"min_confidence": 0.55, "min_margin": 0.0, "top_k": 3},
        "dataset": {"id": "test", "images_per_class": 10},
    }
    for section, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(section), dict):
            base[section] = {**base[section], **value}
        else:
            base[section] = value
    return base


def write_manifest(directory: Path, payload: dict[str, Any]) -> Path:
    path = directory / "model_manifest.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


@pytest.fixture
def manifest_factory(tmp_path: Path) -> Callable[..., Path]:
    """Crée un couple (modèle ONNX, manifeste) cohérent dans un dossier isolé."""

    def factory(
        *,
        num_classes: int = 4,
        height: int = 32,
        width: int = 32,
        channels: int = 1,
        with_softmax: bool = True,
        include_hash: bool = True,
        build_model: bool = True,
        manifest_overrides: dict[str, Any] | None = None,
        directory: Path | None = None,
    ) -> Path:
        target = directory or tmp_path
        target.mkdir(parents=True, exist_ok=True)
        artifact = target / "model.onnx"
        if build_model:
            make_cnn(
                artifact,
                num_classes=num_classes,
                height=height,
                width=width,
                channels=channels,
                with_softmax=with_softmax,
            )
        payload = manifest_dict(
            input={"height": height, "width": width, "channels": channels},
            output={"activation": "softmax" if with_softmax else "logits"},
            classes=[dict(entry) for entry in DEFAULT_CLASSES[:num_classes]],
        )
        if include_hash and artifact.exists():
            payload["model"]["sha256"] = compute_sha256(artifact)
        for section, value in (manifest_overrides or {}).items():
            if isinstance(value, dict) and isinstance(payload.get(section), dict):
                payload[section] = {**payload[section], **value}
            else:
                payload[section] = value
        return write_manifest(target, payload)

    return factory


@pytest.fixture
def settings_factory() -> Callable[..., Settings]:
    """Construit des réglages explicites, sans lecture du fichier .env local."""

    def factory(**overrides: Any) -> Settings:
        defaults: dict[str, Any] = {
            "model_manifest_path": None,
            "model_path": None,
            "allowed_origins": "http://localhost:5500",
            "log_level": "WARNING",
            "warmup_on_startup": True,
            "strict_startup": False,
            "_env_file": None,
        }
        defaults.update(overrides)
        return Settings(**defaults)

    return factory


@pytest.fixture
def client_factory(
    settings_factory: Callable[..., Settings],
) -> Callable[..., Iterator[TestClient]]:
    """Fabrique un client de test à partir de réglages donnés."""

    def factory(**overrides: Any) -> TestClient:
        return TestClient(create_app(settings_factory(**overrides)))

    return factory


@pytest.fixture
def ready_client(
    manifest_factory: Callable[..., Path], client_factory: Callable[..., TestClient]
) -> Iterator[TestClient]:
    """Client dont le modèle est chargé et prêt."""
    manifest = manifest_factory()
    with client_factory(model_manifest_path=manifest) as client:
        yield client


@pytest.fixture
def empty_client(client_factory: Callable[..., TestClient]) -> Iterator[TestClient]:
    """Client sans modèle configuré."""
    with client_factory() as client:
        yield client


class FakeAdapter(ModelAdapter):
    """Adaptateur déterministe, réservé aux tests.

    Permet de contrôler exactement le vecteur de scores afin de vérifier les
    règles de décision sans dépendre des poids d'un modèle.
    """

    thread_safe = False

    def __init__(self, scores: np.ndarray, input_shape: tuple[int | None, ...]) -> None:
        self._scores = np.asarray(scores, dtype=np.float64)
        self._input_shape = input_shape
        self.calls = 0

    @property
    def input_shape(self) -> tuple[int | None, ...]:
        return self._input_shape

    @property
    def output_size(self) -> int | None:
        return int(self._scores.size)

    @property
    def runtime_version(self) -> str:
        return "fake-adapter"

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        self.calls += 1
        return self._scores.copy()
