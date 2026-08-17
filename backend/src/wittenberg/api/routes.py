"""Routes HTTP du service."""

from __future__ import annotations

import asyncio
import logging
import time

from fastapi import APIRouter, File, Request, Response, UploadFile, status

from wittenberg import __version__
from wittenberg.api.errors import ApiError
from wittenberg.api.schemas import (
    ClassInfo,
    HealthResponse,
    InputRequirements,
    ModelInfoResponse,
    ModelRef,
    Prediction,
    PredictionResponse,
    PreprocessingRef,
    Quality,
    QualityWarningOut,
    ReadyResponse,
    TopKEntry,
)
from wittenberg.config import Settings
from wittenberg.imaging.pipeline import PreprocessResult, preprocess_bytes
from wittenberg.imaging.validation import NATIVELY_SUPPORTED, ImageValidationError
from wittenberg.model.decision import decide, to_probabilities
from wittenberg.model.service import ModelService, ModelStatus

logger = logging.getLogger(__name__)

health_router = APIRouter(tags=["health"])
api_router = APIRouter(tags=["inference"])


def _settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def _service(request: Request) -> ModelService:
    service: ModelService = request.app.state.model_service
    return service


def _requirements(settings: Settings) -> InputRequirements:
    return InputRequirements(
        max_upload_mb=settings.max_upload_mb,
        max_image_pixels=settings.max_image_pixels,
        max_image_dimension=settings.max_image_dimension,
        min_image_dimension=settings.min_image_dimension,
        accepted_formats=sorted(fmt.value for fmt in NATIVELY_SUPPORTED),
    )


# ── Santé et disponibilité ───────────────────────────────────────────────


@health_router.get("/health", response_model=HealthResponse, summary="Processus actif")
async def health() -> HealthResponse:
    """Vivacité du processus. N'indique rien sur le modèle."""
    return HealthResponse(status="ok", service="wittenberg-backend", version=__version__)


@health_router.get("/ready", response_model=ReadyResponse, summary="Modèle prêt")
async def ready(request: Request, response: Response) -> ReadyResponse:
    """Disponibilité réelle du service d'inférence.

    Renvoie 503 tant qu'aucun modèle valide n'est chargé.
    """
    service = _service(request)
    state = service.state
    if not service.is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadyResponse(
            ready=False,
            status=state.status.value,
            reason=state.reason,
            missing=state.missing,
        )
    manifest = service.manifest
    return ReadyResponse(
        ready=True,
        status=ModelStatus.READY.value,
        model=ModelRef(name=manifest.model.name, version=manifest.model.version),
        warmup_ms=service.warmup_ms,
    )


# ── Informations sur le modèle ───────────────────────────────────────────


@api_router.get("/model-info", response_model=ModelInfoResponse, summary="Modèle et classes")
async def model_info(request: Request) -> ModelInfoResponse:
    """Version, classes disponibles et contraintes publiques d'entrée.

    Répond 200 même sans modèle : le frontend peut ainsi afficher un état
    « indisponible » précis plutôt qu'une erreur générique.
    """
    settings = _settings(request)
    service = _service(request)
    requirements = _requirements(settings)

    if not service.is_ready:
        state = service.state
        return ModelInfoResponse(
            available=False,
            status=state.status.value,
            reason=state.reason,
            missing=state.missing,
            requirements=requirements,
        )

    manifest = service.manifest
    spec = manifest.input
    return ModelInfoResponse(
        available=True,
        status=ModelStatus.READY.value,
        model=ModelRef(name=manifest.model.name, version=manifest.model.version),
        framework=manifest.model.framework.value,
        runtime=service.runtime_version,
        preprocessing_version=manifest.preprocessing.version,
        input={
            "height": spec.height,
            "width": spec.width,
            "channels": spec.channels,
            "layout": spec.layout.value,
            "dtype": spec.dtype,
            "value_range": list(spec.value_range),
            "normalization": spec.normalization.mode.value,
        },
        num_classes=manifest.num_classes,
        classes=[
            ClassInfo(index=c.index, label=c.label, frequency_hz=c.frequency_hz)
            for c in manifest.ordered_classes
        ],
        min_confidence=manifest.decision.min_confidence,
        min_margin=manifest.decision.min_margin,
        top_k=manifest.decision.top_k,
        dataset=manifest.dataset.model_dump(exclude_none=True) or None,
        requirements=requirements,
    )


# ── Prédiction ───────────────────────────────────────────────────────────


@api_router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Estimer la fréquence d'un motif de Chladni",
)
async def predict(
    request: Request,
    image: UploadFile | None = File(default=None),
    file: UploadFile | None = File(default=None),
) -> PredictionResponse:
    """Analyse une image unique envoyée en ``multipart/form-data``.

    Le champ attendu est ``image`` ; ``file`` est accepté comme alias.
    L'image n'est jamais écrite durablement sur le disque et le nom de fichier
    fourni par le client n'est jamais utilisé comme chemin.
    """
    started = time.perf_counter()
    settings = _settings(request)
    service = _service(request)
    request_id: str = request.state.request_id

    upload = image or file
    if upload is None:
        raise ApiError(
            "missing_file",
            "Aucun fichier reçu. Envoyez une image dans le champ multipart « image ».",
            status.HTTP_422_UNPROCESSABLE_CONTENT,
        )

    # Le modèle est vérifié avant de lire le corps : inutile de consommer
    # de la mémoire si aucune inférence n'est possible.
    if not service.is_ready:
        await _close(upload)
        raise ApiError(
            "model_unavailable",
            "Le modèle d'analyse n'est pas disponible sur ce serveur. "
            "Aucune estimation ne peut être produite pour le moment.",
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    try:
        data = await _read_limited(upload, settings.max_upload_bytes)
    finally:
        # Starlette peut avoir déversé le téléversement dans un fichier
        # temporaire : il est refermé et supprimé dans tous les cas.
        await _close(upload)

    manifest = service.manifest

    def _run() -> PreprocessResult:
        result, _decoded = preprocess_bytes(
            data,
            manifest,
            max_bytes=settings.max_upload_bytes,
            max_pixels=settings.max_image_pixels,
            max_dimension=settings.max_image_dimension,
            min_dimension=settings.min_image_dimension,
            debug_preview=settings.enable_debug_preview,
        )
        return result

    prepared = await asyncio.to_thread(_run)

    scores = await service.infer(prepared.tensor)
    probabilities = to_probabilities(scores, manifest.output.activation)
    decision = decide(probabilities, manifest, top_k=settings.top_k)

    processing_ms = round((time.perf_counter() - started) * 1000, 3)

    logger.info(
        "prediction",
        extra={
            "request_id": request_id,
            "model_version": manifest.model.version,
            "preprocessing_version": manifest.preprocessing.version,
            "predicted_index": decision.best.index,
            "confidence": round(decision.best.probability, 4),
            "reliable": decision.reliable,
            "quality_score": prepared.quality.score,
            "quality_warnings": [w.code for w in prepared.quality.warnings],
            "plate_method": prepared.plate.method,
            "bytes_received": len(data),
            "processing_ms": processing_ms,
        },
    )

    return PredictionResponse(
        request_id=request_id,
        prediction=Prediction(
            label=decision.best.label,
            frequency_hz=decision.best.frequency_hz,
            class_index=decision.best.index,
            confidence=round(decision.best.probability, 6),
            reliable=decision.reliable,
            reason=decision.reason,
            margin=decision.margin,
        ),
        top_k=[
            TopKEntry(
                label=entry.label,
                frequency_hz=entry.frequency_hz,
                class_index=entry.index,
                probability=round(entry.probability, 6),
            )
            for entry in decision.top_k
        ],
        quality=Quality(
            score=prepared.quality.score,
            warnings=[
                QualityWarningOut(code=w.code, message=w.message, severity=w.severity)
                for w in prepared.quality.warnings
            ],
            metrics=prepared.quality.metrics,
            plate_detected=prepared.plate.detected,
            plate_method=prepared.plate.method,
        ),
        model=ModelRef(name=manifest.model.name, version=manifest.model.version),
        preprocessing=PreprocessingRef(
            version=manifest.preprocessing.version, steps=prepared.steps
        ),
        processing_ms=processing_ms,
        debug_preview_png_base64=prepared.debug_preview,
    )


async def _read_limited(upload: UploadFile, max_bytes: int) -> bytes:
    """Lit le téléversement en refusant tout dépassement de taille.

    La lecture s'arrête dès qu'un octet de trop est rencontré : un fichier
    volumineux n'est jamais entièrement chargé en mémoire.
    """
    chunks: list[bytes] = []
    total = 0
    chunk_size = 64 * 1024
    while True:
        chunk = await upload.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            limit_mb = max_bytes / (1024 * 1024)
            raise ImageValidationError(
                "file_too_large",
                f"L'image dépasse la taille maximale autorisée ({limit_mb:.0f} Mo).",
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            )
        chunks.append(chunk)
    if total == 0:
        raise ImageValidationError("empty_file", "Le fichier reçu est vide.")
    return b"".join(chunks)


async def _close(upload: UploadFile | None) -> None:
    if upload is None:
        return
    try:
        await upload.close()
    except Exception:  # pragma: no cover - fermeture au mieux
        logger.debug("upload_close_failed")
