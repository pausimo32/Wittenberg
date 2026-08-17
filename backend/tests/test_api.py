"""Contrat HTTP de bout en bout, avec un vrai modèle ONNX chargé."""

from __future__ import annotations

import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.conftest import FakeAdapter
from tests.support import images

PNG = images.encode(images.chladni_like(256), "PNG")
JPEG = images.encode(images.chladni_like(256), "JPEG")


def post(
    client: TestClient,
    payload: bytes,
    filename: str = "motif.png",
    content_type: str = "image/png",
    field: str = "image",
):  # type: ignore[no-untyped-def]
    return client.post("/api/v1/predict", files={field: (filename, payload, content_type)})


# ── Santé et disponibilité ───────────────────────────────────────────────


def test_health_is_ok_without_model(empty_client: TestClient) -> None:
    response = empty_client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_is_503_without_model(empty_client: TestClient) -> None:
    response = empty_client.get("/ready")
    body = response.json()

    assert response.status_code == 503
    assert body["ready"] is False
    assert body["status"] == "not_configured"
    assert body["missing"]


def test_ready_is_200_with_model(ready_client: TestClient) -> None:
    response = ready_client.get("/ready")
    body = response.json()

    assert response.status_code == 200
    assert body["ready"] is True
    assert body["model"]["version"] == "test-1.0.0"
    assert body["warmup_ms"] is not None


def test_model_info_without_model_explains_the_gap(empty_client: TestClient) -> None:
    """200 avec `available: false` : le frontend peut afficher un état précis."""
    response = empty_client.get("/api/v1/model-info")
    body = response.json()

    assert response.status_code == 200
    assert body["available"] is False
    assert body["classes"] is None
    assert body["requirements"]["max_upload_mb"] == 10.0


def test_model_info_exposes_classes_in_manifest_order(ready_client: TestClient) -> None:
    body = ready_client.get("/api/v1/model-info").json()

    assert body["available"] is True
    assert body["num_classes"] == 4
    assert [c["index"] for c in body["classes"]] == [0, 1, 2, 3]
    assert [c["frequency_hz"] for c in body["classes"]] == [111.0, 222.0, 333.0, 444.0]
    assert body["preprocessing_version"] == "prep-test-1"
    assert body["input"]["height"] == 32


# ── Prédiction ───────────────────────────────────────────────────────────


def test_predict_returns_the_full_contract(ready_client: TestClient) -> None:
    response = post(ready_client, PNG)
    body = response.json()

    assert response.status_code == 200
    assert set(body) >= {
        "request_id",
        "prediction",
        "top_k",
        "quality",
        "model",
        "preprocessing",
        "processing_ms",
    }
    prediction = body["prediction"]
    assert 0.0 <= prediction["confidence"] <= 1.0
    assert isinstance(prediction["reliable"], bool)
    assert prediction["frequency_hz"] in {111.0, 222.0, 333.0, 444.0}
    assert prediction["label"] == f"{prediction['frequency_hz']:.0f} Hz"
    assert body["model"]["version"] == "test-1.0.0"
    assert body["preprocessing"]["version"] == "prep-test-1"
    assert body["processing_ms"] > 0
    assert 0.0 <= body["quality"]["score"] <= 1.0


def test_top_k_is_sorted_and_consistent(ready_client: TestClient) -> None:
    body = post(ready_client, PNG).json()
    probabilities = [entry["probability"] for entry in body["top_k"]]

    assert len(body["top_k"]) == 3
    assert probabilities == sorted(probabilities, reverse=True)
    assert body["top_k"][0]["label"] == body["prediction"]["label"]
    assert body["top_k"][0]["probability"] == pytest.approx(
        body["prediction"]["confidence"], abs=1e-6
    )


def test_probabilities_sum_to_one(ready_client: TestClient) -> None:
    ready_client.app.state.model_service._adapter = FakeAdapter(  # type: ignore[attr-defined]
        np.array([0.5, 0.25, 0.15, 0.10]), (1, 32, 32, 1)
    )
    body = post(ready_client, PNG).json()

    assert sum(entry["probability"] for entry in body["top_k"]) == pytest.approx(0.9)
    assert body["prediction"]["confidence"] == pytest.approx(0.5)


def test_jpeg_is_accepted(ready_client: TestClient) -> None:
    assert post(ready_client, JPEG, "photo.jpg", "image/jpeg").status_code == 200


def test_webp_is_accepted(ready_client: TestClient) -> None:
    payload = images.encode(images.chladni_like(256), "WEBP")

    assert post(ready_client, payload, "photo.webp", "image/webp").status_code == 200


def test_camera_capture_field_name_alias(ready_client: TestClient) -> None:
    """Le champ « file » est accepté comme alias de « image »."""
    assert post(ready_client, JPEG, "capture.jpg", "image/jpeg", field="file").status_code == 200


def test_exif_rotated_photo_is_accepted(ready_client: TestClient) -> None:
    payload = images.jpeg_with_exif_orientation(orientation=6, size=256)

    assert post(ready_client, payload, "IMG_0042.jpg", "image/jpeg").status_code == 200


def test_low_confidence_returns_200_with_reliable_false(ready_client: TestClient) -> None:
    """Une image valide mal classée reste un succès HTTP, jamais une erreur 4xx."""
    ready_client.app.state.model_service._adapter = FakeAdapter(  # type: ignore[attr-defined]
        np.array([0.30, 0.28, 0.22, 0.20]), (1, 32, 32, 1)
    )
    response = post(ready_client, PNG)
    body = response.json()

    assert response.status_code == 200
    assert body["prediction"]["reliable"] is False
    assert body["prediction"]["reason"] == "low_confidence"


def test_quality_warnings_are_returned(ready_client: TestClient) -> None:
    payload = images.encode(images.flat_image(256, value=8), "PNG")
    body = post(ready_client, payload).json()
    codes = {warning["code"] for warning in body["quality"]["warnings"]}

    assert "no_pattern_detected" in codes
    for warning in body["quality"]["warnings"]:
        assert warning["severity"] in {"info", "warning"}
        assert warning["message"]


def test_debug_preview_is_absent_by_default(ready_client: TestClient) -> None:
    assert post(ready_client, PNG).json()["debug_preview_png_base64"] is None


def test_debug_preview_can_be_enabled(
    manifest_factory: Callable[..., Path], client_factory: Callable[..., TestClient]
) -> None:
    manifest = manifest_factory()
    with client_factory(model_manifest_path=manifest, enable_debug_preview=True) as client:
        assert post(client, PNG).json()["debug_preview_png_base64"]


# ── Erreurs ──────────────────────────────────────────────────────────────


def test_predict_without_model_is_503(empty_client: TestClient) -> None:
    response = post(empty_client, PNG)
    body = response.json()

    assert response.status_code == 503
    assert body["error"]["code"] == "model_unavailable"
    # Aucune prédiction simulée ne doit apparaître.
    assert "prediction" not in body


def test_missing_file_is_422(ready_client: TestClient) -> None:
    response = ready_client.post("/api/v1/predict", data={"autre": "champ"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] in {"missing_file", "invalid_request"}


def test_empty_file_is_400(ready_client: TestClient) -> None:
    response = post(ready_client, b"", "vide.png")

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "empty_file"


def test_text_file_disguised_as_png_is_415(ready_client: TestClient) -> None:
    """Extension et Content-Type mensongers : le contenu tranche."""
    response = post(ready_client, b"absolument pas une image" * 10, "faux.png", "image/png")

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_format"


def test_corrupt_image_is_400(ready_client: TestClient) -> None:
    response = post(ready_client, PNG[: len(PNG) // 3], "tronque.png")

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "corrupt_image"


def test_oversized_upload_is_413(
    manifest_factory: Callable[..., Path], client_factory: Callable[..., TestClient]
) -> None:
    manifest = manifest_factory()
    with client_factory(model_manifest_path=manifest, max_upload_mb=0.05) as client:
        payload = images.encode(images.noise_image(512), "PNG")
        response = post(client, payload)

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"


def test_heic_upload_is_415_with_guidance(ready_client: TestClient) -> None:
    payload = b"\x00\x00\x00\x20ftypheic" + b"\x00" * 128
    response = post(ready_client, payload, "IMG_1234.HEIC", "image/heic")
    body = response.json()

    assert response.status_code == 415
    assert "JPEG" in body["error"]["message"]


def test_errors_never_leak_internals(ready_client: TestClient) -> None:
    response = post(ready_client, b"pas une image du tout" * 8, "x.png")
    text = response.text

    assert "Traceback" not in text
    assert "wittenberg" not in text.lower()
    assert set(response.json()) == {"request_id", "error"}


def test_unknown_route_is_404(ready_client: TestClient) -> None:
    assert ready_client.get("/api/v1/inexistant").status_code == 404


# ── Transversal ──────────────────────────────────────────────────────────


def test_request_id_is_returned_and_unique(ready_client: TestClient) -> None:
    first = post(ready_client, PNG)
    second = post(ready_client, PNG)

    assert first.headers["X-Request-ID"] == first.json()["request_id"]
    assert first.json()["request_id"] != second.json()["request_id"]


def test_client_request_id_is_propagated(ready_client: TestClient) -> None:
    response = ready_client.get("/health", headers={"X-Request-ID": "trace-abc-123"})

    assert response.headers["X-Request-ID"] == "trace-abc-123"


def test_hostile_request_id_is_replaced(ready_client: TestClient) -> None:
    response = ready_client.get("/health", headers={"X-Request-ID": "x" * 500})

    assert response.headers["X-Request-ID"] != "x" * 500


def test_cors_allows_configured_origin(ready_client: TestClient) -> None:
    response = ready_client.options(
        "/api/v1/predict",
        headers={
            "Origin": "http://localhost:5500",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5500"


def test_cors_rejects_unknown_origin(ready_client: TestClient) -> None:
    response = ready_client.options(
        "/api/v1/predict",
        headers={
            "Origin": "https://site-malveillant.example",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.headers.get("access-control-allow-origin") is None


def test_wildcard_origin_is_never_configured(
    manifest_factory: Callable[..., Path], client_factory: Callable[..., TestClient]
) -> None:
    manifest = manifest_factory()
    with client_factory(model_manifest_path=manifest, allowed_origins="*,http://a.test") as client:
        response = client.options(
            "/api/v1/predict",
            headers={"Origin": "https://autre.test", "Access-Control-Request-Method": "POST"},
        )

    assert response.headers.get("access-control-allow-origin") is None


def test_temporary_files_are_cleaned_up(
    manifest_factory: Callable[..., Path],
    client_factory: Callable[..., TestClient],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Un téléversement déversé sur disque ne doit laisser aucun résidu."""
    spool = tmp_path / "spool"
    spool.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(spool))

    # Au-delà du seuil de Starlette, le corps est écrit dans un fichier temporaire.
    big = images.encode(images.noise_image(1400), "PNG")
    assert len(big) > 1024 * 1024

    manifest = manifest_factory()
    with client_factory(model_manifest_path=manifest, max_upload_mb=10) as client:
        assert post(client, big).status_code == 200

    assert list(spool.iterdir()) == []


def test_image_is_never_persisted(
    manifest_factory: Callable[..., Path],
    client_factory: Callable[..., TestClient],
    tmp_path: Path,
) -> None:
    """Aucune photographie ne doit apparaître sur le disque du serveur."""
    workdir = tmp_path / "travail"
    workdir.mkdir()
    manifest = manifest_factory(directory=workdir)
    before = set(workdir.rglob("*"))

    with client_factory(model_manifest_path=manifest) as client:
        assert post(client, PNG, "photo-privee.png").status_code == 200

    assert set(workdir.rglob("*")) == before


def test_user_filename_is_not_used_as_a_path(ready_client: TestClient) -> None:
    """Un nom de fichier hostile ne doit produire aucune écriture ni erreur 500."""
    response = post(ready_client, PNG, "../../../../etc/passwd.png")

    assert response.status_code == 200


def test_concurrent_requests_all_succeed(ready_client: TestClient) -> None:
    results: list[int] = []
    lock = threading.Lock()

    def call() -> None:
        status = post(ready_client, PNG).status_code
        with lock:
            results.append(status)

    threads = [threading.Thread(target=call) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == [200] * 8


def test_repeated_requests_are_deterministic(ready_client: TestClient) -> None:
    first = post(ready_client, PNG).json()
    second = post(ready_client, PNG).json()

    assert first["prediction"]["label"] == second["prediction"]["label"]
    assert first["prediction"]["confidence"] == pytest.approx(second["prediction"]["confidence"])


def test_openapi_is_served(ready_client: TestClient) -> None:
    schema = ready_client.get("/openapi.json").json()

    assert "/api/v1/predict" in schema["paths"]
    assert "/ready" in schema["paths"]
