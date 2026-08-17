"""Service de modèle : chargement unique, validation, inférence.

Garanties :

* l'artefact est chargé une seule fois, au démarrage du processus ;
* une inférence de préchauffage valide la cohérence manifeste ↔ modèle ;
* aucun rechargement n'a lieu par requête ;
* l'exécution se fait sur CPU ;
* les runtimes non réentrants sont protégés par un verrou ;
* l'absence de modèle produit une indisponibilité explicite, jamais une
  prédiction simulée.

Aucun artefact provenant d'une requête HTTP n'est jamais désérialisé : seul le
fichier de confiance désigné par la configuration du serveur est chargé.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np

from wittenberg.config import Settings
from wittenberg.imaging.pipeline import PreprocessingUnavailable, check_pipeline_requirements
from wittenberg.model.adapters.base import AdapterError, ModelAdapter
from wittenberg.model.manifest import (
    ManifestError,
    ModelManifest,
    compute_sha256,
    load_manifest,
)
from wittenberg.model.registry import build_adapter

logger = logging.getLogger(__name__)


class ModelStatus(StrEnum):
    #: ni modèle ni manifeste configurés
    NOT_CONFIGURED = "not_configured"
    #: configuration présente mais chargement en échec
    ERROR = "error"
    #: modèle chargé, validé et préchauffé
    READY = "ready"


class ModelUnavailableError(Exception):
    """Le modèle n'est pas disponible ; l'API doit répondre 503."""


class InferenceTimeoutError(Exception):
    """L'inférence a dépassé le délai autorisé."""


@dataclass(frozen=True)
class ModelState:
    """Instantané de l'état du service, exposé par ``/ready``."""

    status: ModelStatus
    reason: str | None = None
    #: artefacts manquants ou incohérents, décrits pour l'exploitant
    missing: list[str] | None = None


class ModelService:
    """Détient le modèle chargé et exécute les inférences."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._adapter: ModelAdapter | None = None
        self._manifest: ModelManifest | None = None
        self._state = ModelState(ModelStatus.NOT_CONFIGURED, reason="Service non initialisé.")
        self._lock = threading.Lock()
        self._warmup_ms: float | None = None
        self._artifact_sha256: str | None = None

    # ── État ─────────────────────────────────────────────────────────────

    @property
    def state(self) -> ModelState:
        return self._state

    @property
    def is_ready(self) -> bool:
        return self._state.status is ModelStatus.READY

    @property
    def manifest(self) -> ModelManifest:
        if self._manifest is None:
            raise ModelUnavailableError(self._state.reason or "Modèle non disponible.")
        return self._manifest

    @property
    def adapter(self) -> ModelAdapter:
        if self._adapter is None:
            raise ModelUnavailableError(self._state.reason or "Modèle non disponible.")
        return self._adapter

    @property
    def runtime_version(self) -> str | None:
        return self._adapter.runtime_version if self._adapter else None

    @property
    def warmup_ms(self) -> float | None:
        return self._warmup_ms

    @property
    def artifact_sha256(self) -> str | None:
        return self._artifact_sha256

    # ── Cycle de vie ─────────────────────────────────────────────────────

    def load(self) -> ModelState:
        """Charge manifeste puis modèle, valide et préchauffe.

        N'émet aucune exception : l'état résultant est renvoyé et consultable
        via ``/ready``. Le démarrage strict est appliqué par l'appelant.
        """
        settings = self._settings
        missing: list[str] = []
        if settings.model_manifest_path is None:
            missing.append("WITTENBERG_MODEL_MANIFEST_PATH n'est pas définie")
        if settings.model_path is None:
            missing.append(
                "WITTENBERG_MODEL_PATH n'est pas définie "
                "(sinon l'artefact du manifeste est utilisé)"
            )
        if settings.model_manifest_path is None:
            self._state = ModelState(
                ModelStatus.NOT_CONFIGURED,
                reason="Aucun modèle n'est configuré sur ce serveur.",
                missing=missing,
            )
            logger.warning("model_not_configured", extra={"missing": missing})
            return self._state

        try:
            manifest = load_manifest(settings.model_manifest_path)
            artifact = self._resolve_artifact(manifest)
            self._verify_hash(manifest, artifact)
            check_pipeline_requirements(manifest.preprocessing)
            adapter = build_adapter(
                manifest, artifact, onnx_intra_op_threads=settings.onnx_intra_op_threads
            )
        except (ManifestError, AdapterError, PreprocessingUnavailable, OSError) as exc:
            self._state = ModelState(ModelStatus.ERROR, reason=str(exc))
            logger.error("model_load_failed", extra={"error": str(exc)})
            return self._state

        try:
            self._validate_compatibility(manifest, adapter)
            if settings.warmup_on_startup:
                self._warmup(manifest, adapter)
        except (AdapterError, ValueError) as exc:
            adapter.close()
            self._state = ModelState(ModelStatus.ERROR, reason=str(exc))
            logger.error("model_validation_failed", extra={"error": str(exc)})
            return self._state

        self._manifest = manifest
        self._adapter = adapter
        self._state = ModelState(ModelStatus.READY)
        logger.info(
            "model_ready",
            extra={
                "model_name": manifest.model.name,
                "model_version": manifest.model.version,
                "num_classes": manifest.num_classes,
                "runtime": adapter.runtime_version,
                "warmup_ms": self._warmup_ms,
                "thread_safe": adapter.thread_safe,
            },
        )
        return self._state

    def shutdown(self) -> None:
        """Libère le runtime."""
        if self._adapter is not None:
            self._adapter.close()
            self._adapter = None
        self._manifest = None
        self._state = ModelState(ModelStatus.NOT_CONFIGURED, reason="Service arrêté.")

    # ── Inférence ────────────────────────────────────────────────────────

    def infer_sync(self, tensor: np.ndarray) -> np.ndarray:
        """Inférence bloquante, utilisable hors API (scripts, évaluation)."""
        adapter = self.adapter
        if adapter.thread_safe:
            return adapter.infer(tensor)
        with self._lock:
            return adapter.infer(tensor)

    async def infer(self, tensor: np.ndarray) -> np.ndarray:
        """Inférence asynchrone, exécutée hors de la boucle d'événements.

        Le délai d'attente protège l'API. Il n'interrompt pas le calcul en
        cours dans le fil d'exécution : sur un runtime non réentrant, la
        requête suivante attend simplement la libération du verrou.
        """
        if not self.is_ready:
            raise ModelUnavailableError(self._state.reason or "Modèle non disponible.")
        timeout = self._settings.inference_timeout_ms / 1000.0
        try:
            return await asyncio.wait_for(asyncio.to_thread(self.infer_sync, tensor), timeout)
        except TimeoutError as exc:
            raise InferenceTimeoutError(
                f"l'inférence a dépassé {self._settings.inference_timeout_ms} ms"
            ) from exc

    # ── Détails internes ─────────────────────────────────────────────────

    def _resolve_artifact(self, manifest: ModelManifest) -> Path:
        """Le chemin explicite prime ; sinon celui déclaré par le manifeste."""
        artifact = self._settings.model_path or manifest.artifact_path
        if not artifact.exists():
            raise ManifestError(
                f"artefact du modèle introuvable : {artifact}. "
                "Placez le fichier au chemin indiqué ou corrigez "
                "WITTENBERG_MODEL_PATH."
            )
        return artifact

    def _verify_hash(self, manifest: ModelManifest, artifact: Path) -> None:
        expected = manifest.model.sha256
        if not expected:
            logger.warning(
                "model_hash_absent",
                extra={"hint": "ajoutez model.sha256 au manifeste pour garantir l'intégrité"},
            )
            return
        digest = compute_sha256(artifact)
        self._artifact_sha256 = digest
        if not self._settings.verify_model_hash:
            logger.warning("model_hash_check_disabled")
            return
        if digest.lower() != expected.lower():
            raise ManifestError(
                "l'empreinte SHA-256 de l'artefact ne correspond pas au manifeste "
                f"(attendu {expected[:12]}…, obtenu {digest[:12]}…)"
            )

    def _validate_compatibility(self, manifest: ModelManifest, adapter: ModelAdapter) -> None:
        """Vérifie que le modèle chargé correspond bien au manifeste."""
        declared = adapter.output_size
        if declared is not None and declared != manifest.num_classes:
            raise ValueError(
                f"le modèle produit {declared} sorties alors que le manifeste "
                f"déclare {manifest.num_classes} classes"
            )

        expected_shape = manifest.input.tensor_shape
        actual = adapter.input_shape
        if len(actual) != len(expected_shape):
            raise ValueError(
                f"le modèle attend un tenseur de rang {len(actual)} "
                f"alors que le manifeste décrit un rang {len(expected_shape)}"
            )
        # La dimension de lot est ignorée : elle est souvent dynamique.
        for position, (got, want) in enumerate(zip(actual, expected_shape, strict=True)):
            if position == 0 or got is None:
                continue
            if got != want:
                raise ValueError(
                    f"géométrie d'entrée incompatible : le modèle attend {actual}, "
                    f"le manifeste décrit {expected_shape}"
                )

    def _warmup(self, manifest: ModelManifest, adapter: ModelAdapter) -> None:
        """Inférence de préchauffage sur un tenseur neutre.

        Contrôle décisif : c'est le seul moyen de vérifier le nombre réel de
        sorties lorsque l'artefact déclare une dimension dynamique.
        """
        spec = manifest.input
        tensor = np.zeros(spec.tensor_shape, dtype=np.dtype(spec.dtype))
        started = time.perf_counter()
        scores = adapter.infer(tensor)
        self._warmup_ms = round((time.perf_counter() - started) * 1000, 3)
        if scores.size != manifest.num_classes:
            raise ValueError(
                f"le préchauffage a produit {scores.size} scores alors que le "
                f"manifeste déclare {manifest.num_classes} classes"
            )
