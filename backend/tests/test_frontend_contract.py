"""Verrouille le contrat consommé par le frontend.

``js/predictionService.js`` lit des noms de champs précis. Ces tests échouent
si le backend les renomme, ce qui casserait l'analyseur en silence : le
navigateur afficherait « undefined » plutôt qu'une erreur.

Les chemins lus par le frontend sont également extraits du fichier JavaScript
lui-même, afin que les deux côtés ne puissent pas diverger sans alerte.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from tests.support import images

FRONTEND_ROOT = Path(__file__).resolve().parents[2]
PREDICTION_SERVICE_JS = FRONTEND_ROOT / "js" / "predictionService.js"
CONFIG_JS = FRONTEND_ROOT / "js" / "config.js"

PNG = images.encode(images.chladni_like(256), "PNG")


def test_predict_response_matches_frontend_expectations(ready_client: TestClient) -> None:
    """Chaque champ lu par `predictionService.js` doit exister et être typé."""
    body = ready_client.post(
        "/api/v1/predict", files={"image": ("motif.png", PNG, "image/png")}
    ).json()

    assert isinstance(body["request_id"], str)
    assert isinstance(body["processing_ms"], int | float)

    assert isinstance(body["model"]["name"], str)
    assert isinstance(body["model"]["version"], str)
    assert isinstance(body["preprocessing"]["version"], str)

    prediction = body["prediction"]
    assert isinstance(prediction["label"], str)
    assert isinstance(prediction["frequency_hz"], int | float)
    assert isinstance(prediction["class_index"], int)
    assert isinstance(prediction["confidence"], int | float)
    assert isinstance(prediction["reliable"], bool)
    assert prediction["reason"] is None or isinstance(prediction["reason"], str)
    assert isinstance(prediction["margin"], int | float)

    assert isinstance(body["top_k"], list) and body["top_k"]
    for entry in body["top_k"]:
        assert isinstance(entry["label"], str)
        assert isinstance(entry["frequency_hz"], int | float)
        assert isinstance(entry["class_index"], int)
        assert isinstance(entry["probability"], int | float)

    quality = body["quality"]
    assert isinstance(quality["score"], int | float)
    assert isinstance(quality["warnings"], list)


def test_model_info_matches_frontend_expectations(ready_client: TestClient) -> None:
    body = ready_client.get("/api/v1/model-info").json()

    assert isinstance(body["available"], bool)
    assert isinstance(body["status"], str)
    assert isinstance(body["model"]["version"], str)


def test_error_shape_matches_frontend_expectations(ready_client: TestClient) -> None:
    """Le frontend lit `error.code` et `error.message` sur tout échec."""
    body = ready_client.post(
        "/api/v1/predict", files={"image": ("faux.png", b"pas une image" * 8, "image/png")}
    ).json()

    assert isinstance(body["error"]["code"], str)
    assert isinstance(body["error"]["message"], str)
    assert isinstance(body["request_id"], str)


def test_frontend_sends_the_field_name_the_api_expects() -> None:
    source = PREDICTION_SERVICE_JS.read_text(encoding="utf-8")
    match = re.search(r"formData\.append\(\s*'([^']+)'", source)

    assert match is not None, "predictionService.js doit téléverser un champ nommé"
    assert match.group(1) == "image"


def test_frontend_targets_the_declared_routes() -> None:
    source = PREDICTION_SERVICE_JS.read_text(encoding="utf-8")

    assert "apiUrl('/predict')" in source
    assert "apiUrl('/model-info')" in source
    assert re.search(r"API_PREFIX:\s*'/api/v1'", CONFIG_JS.read_text(encoding="utf-8"))


def test_frontend_reads_only_fields_the_api_returns(ready_client: TestClient) -> None:
    """Aucun champ lu côté navigateur ne doit manquer dans la réponse réelle."""
    source = PREDICTION_SERVICE_JS.read_text(encoding="utf-8")
    body = ready_client.post(
        "/api/v1/predict", files={"image": ("motif.png", PNG, "image/png")}
    ).json()

    def has(path: str) -> bool:
        current: Any = body
        for part in path.split("."):
            if isinstance(current, list):
                current = current[0]
            if not isinstance(current, dict) or part not in current:
                return False
            current = current[part]
        return True

    # Accès de la forme body.x.y présents dans le service JavaScript. Les
    # chemins sous `error` appartiennent à la réponse d'échec, vérifiée par
    # `test_error_shape_matches_frontend_expectations`.
    paths = {p for p in re.findall(r"body\.((?:\w+\.)*\w+)", source) if not p.startswith("error")}

    assert paths, "aucun accès aux champs détecté dans predictionService.js"
    for path in sorted(paths):
        assert has(path), f"le backend ne renvoie pas « {path} », lu par le frontend"


def test_frontend_contains_no_simulated_inference() -> None:
    """Le service du navigateur ne doit contenir aucune inférence simulée."""
    source = PREDICTION_SERVICE_JS.read_text(encoding="utf-8").lower()

    for forbidden in ("_mockinference", "math.random", "seededrandom", "mock_latency"):
        assert forbidden not in source, f"code simulé résiduel : {forbidden}"
