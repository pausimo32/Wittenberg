"""Chargement du modèle : une seule fois, validé, ou clairement indisponible."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.conftest import FakeAdapter
from wittenberg.config import Settings
from wittenberg.model.service import ModelService, ModelStatus, ModelUnavailableError


def make_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {"log_level": "WARNING", "_env_file": None}
    base.update(overrides)
    return Settings(**base)


def test_without_configuration_service_is_not_configured() -> None:
    service = ModelService(make_settings())
    state = service.load()

    assert state.status is ModelStatus.NOT_CONFIGURED
    assert service.is_ready is False
    assert state.missing
    assert any("MANIFEST" in item for item in state.missing)


def test_accessing_model_when_unavailable_raises() -> None:
    service = ModelService(make_settings())
    service.load()

    with pytest.raises(ModelUnavailableError):
        _ = service.manifest


def test_ready_after_successful_load(manifest_factory: Callable[..., Path]) -> None:
    service = ModelService(make_settings(model_manifest_path=manifest_factory()))
    state = service.load()

    assert state.status is ModelStatus.READY
    assert service.is_ready
    assert service.manifest.num_classes == 4
    assert service.runtime_version is not None
    assert service.runtime_version.startswith("onnxruntime")
    assert service.warmup_ms is not None
    service.shutdown()


def test_warmup_runs_at_startup(manifest_factory: Callable[..., Path]) -> None:
    """Le préchauffage valide le nombre réel de sorties, pas seulement le déclaré."""
    service = ModelService(make_settings(model_manifest_path=manifest_factory()))
    service.load()

    assert service.warmup_ms is not None and service.warmup_ms >= 0.0
    service.shutdown()


def test_missing_artifact_reports_error(
    manifest_factory: Callable[..., Path], tmp_path: Path
) -> None:
    manifest = manifest_factory(build_model=False, include_hash=False)
    service = ModelService(make_settings(model_manifest_path=manifest))
    state = service.load()

    assert state.status is ModelStatus.ERROR
    assert state.reason is not None and "introuvable" in state.reason


def test_hash_mismatch_blocks_loading(manifest_factory: Callable[..., Path]) -> None:
    """Un artefact modifié après signature ne doit jamais être servi."""
    manifest_path = manifest_factory()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["model"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    state = ModelService(make_settings(model_manifest_path=manifest_path)).load()

    assert state.status is ModelStatus.ERROR
    assert state.reason is not None and "SHA-256" in state.reason


def test_hash_check_can_be_disabled(manifest_factory: Callable[..., Path]) -> None:
    manifest_path = manifest_factory()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["model"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    service = ModelService(
        make_settings(model_manifest_path=manifest_path, verify_model_hash=False)
    )

    assert service.load().status is ModelStatus.READY
    service.shutdown()


def test_class_count_mismatch_is_detected(manifest_factory: Callable[..., Path]) -> None:
    """Modèle à 4 sorties, manifeste à 3 classes : incompatible."""
    manifest_path = manifest_factory(num_classes=4)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["classes"] = payload["classes"][:3]
    payload["decision"]["top_k"] = 3
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    state = ModelService(make_settings(model_manifest_path=manifest_path)).load()

    assert state.status is ModelStatus.ERROR
    assert state.reason is not None and "sorties" in state.reason


def test_input_geometry_mismatch_is_detected(manifest_factory: Callable[..., Path]) -> None:
    manifest_path = manifest_factory(height=32, width=32)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["input"]["height"] = 64
    payload["input"]["width"] = 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    state = ModelService(make_settings(model_manifest_path=manifest_path)).load()

    assert state.status is ModelStatus.ERROR
    assert state.reason is not None and "incompatible" in state.reason


def test_corrupt_artifact_reports_error(manifest_factory: Callable[..., Path]) -> None:
    manifest_path = manifest_factory()
    (manifest_path.parent / "model.onnx").write_bytes(b"ce n'est pas un modele onnx")

    state = ModelService(
        make_settings(model_manifest_path=manifest_path, verify_model_hash=False)
    ).load()

    assert state.status is ModelStatus.ERROR


def test_explicit_model_path_overrides_manifest(
    manifest_factory: Callable[..., Path], tmp_path: Path
) -> None:
    other = tmp_path / "ailleurs"
    manifest_factory(directory=other)
    manifest_path = manifest_factory(include_hash=False)
    (manifest_path.parent / "model.onnx").unlink()

    service = ModelService(
        make_settings(model_manifest_path=manifest_path, model_path=other / "model.onnx")
    )

    assert service.load().status is ModelStatus.READY
    service.shutdown()


def test_logits_model_is_accepted(manifest_factory: Callable[..., Path]) -> None:
    service = ModelService(make_settings(model_manifest_path=manifest_factory(with_softmax=False)))
    state = service.load()

    assert state.status is ModelStatus.READY
    scores = service.infer_sync(np.zeros((1, 32, 32, 1), dtype=np.float32))
    # Des logits bruts ne somment pas à 1.
    assert abs(float(scores.sum()) - 1.0) > 1e-6
    service.shutdown()


def test_model_is_loaded_once_not_per_inference(
    manifest_factory: Callable[..., Path],
) -> None:
    service = ModelService(make_settings(model_manifest_path=manifest_factory()))
    service.load()
    adapter = service.adapter

    for _ in range(5):
        service.infer_sync(np.zeros((1, 32, 32, 1), dtype=np.float32))

    assert service.adapter is adapter
    service.shutdown()


def test_non_thread_safe_adapter_calls_are_serialised() -> None:
    """Un runtime non réentrant ne doit jamais être appelé en parallèle."""
    service = ModelService(make_settings())
    concurrent = 0
    peak = 0
    guard = threading.Lock()

    class TrackingAdapter(FakeAdapter):
        def infer(self, tensor: np.ndarray) -> np.ndarray:
            nonlocal concurrent, peak
            with guard:
                concurrent += 1
                peak = max(peak, concurrent)
            try:
                # Fenêtre suffisante pour qu'un chevauchement soit observable.
                for _ in range(20_000):
                    pass
                return super().infer(tensor)
            finally:
                with guard:
                    concurrent -= 1

    service._adapter = TrackingAdapter(np.array([0.7, 0.1, 0.1, 0.1]), (1, 32, 32, 1))

    threads = [
        threading.Thread(target=service.infer_sync, args=(np.zeros((1, 32, 32, 1), np.float32),))
        for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert peak == 1


def test_shutdown_releases_the_model(manifest_factory: Callable[..., Path]) -> None:
    service = ModelService(make_settings(model_manifest_path=manifest_factory()))
    service.load()
    service.shutdown()

    assert service.is_ready is False
    with pytest.raises(ModelUnavailableError):
        _ = service.adapter
