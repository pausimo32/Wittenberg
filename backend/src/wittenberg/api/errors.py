"""Gestion des erreurs : messages sûrs, codes stables, aucune trace exposée."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from wittenberg.api.schemas import ErrorBody, ErrorResponse
from wittenberg.imaging.pipeline import PreprocessingUnavailable
from wittenberg.imaging.validation import ImageValidationError
from wittenberg.model.service import InferenceTimeoutError, ModelUnavailableError

logger = logging.getLogger(__name__)


class ApiError(Exception):
    """Erreur applicative dotée d'un code stable et d'un message présentable."""

    def __init__(self, code: str, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def error_response(request: Request, code: str, message: str, status_code: int) -> JSONResponse:
    """Construit une réponse d'erreur normalisée."""
    request_id = getattr(request.state, "request_id", "unknown")
    payload = ErrorResponse(request_id=request_id, error=ErrorBody(code=code, message=message))
    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(),
        headers={"X-Request-ID": request_id},
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Installe les gestionnaires d'exceptions de l'application."""

    @app.exception_handler(ImageValidationError)
    async def _image_validation(request: Request, exc: ImageValidationError) -> JSONResponse:
        return error_response(request, exc.code, exc.message, exc.status_code)

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return error_response(request, exc.code, exc.message, exc.status_code)

    @app.exception_handler(ModelUnavailableError)
    async def _model_unavailable(request: Request, exc: ModelUnavailableError) -> JSONResponse:
        return error_response(
            request,
            "model_unavailable",
            "Le modèle d'analyse n'est pas disponible sur ce serveur. "
            "Aucune estimation ne peut être produite pour le moment.",
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    @app.exception_handler(InferenceTimeoutError)
    async def _timeout(request: Request, exc: InferenceTimeoutError) -> JSONResponse:
        return error_response(
            request,
            "inference_timeout",
            "L'analyse a dépassé le délai autorisé. Réessayez avec une image plus petite.",
            status.HTTP_504_GATEWAY_TIMEOUT,
        )

    @app.exception_handler(PreprocessingUnavailable)
    async def _preprocessing(request: Request, exc: PreprocessingUnavailable) -> JSONResponse:
        logger.error("preprocessing_unavailable", extra={"error": str(exc)})
        return error_response(
            request,
            "preprocessing_unavailable",
            "Le prétraitement requis par le modèle n'est pas disponible sur ce serveur.",
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    @app.exception_handler(RequestValidationError)
    async def _request_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        return error_response(
            request,
            "invalid_request",
            "La requête est invalide. Envoyez une image dans un champ "
            "multipart/form-data nommé « image ».",
            status.HTTP_422_UNPROCESSABLE_CONTENT,
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # La trace part dans les journaux du serveur, jamais dans la réponse.
        logger.exception(
            "unhandled_error",
            extra={"path": request.url.path, "error_type": type(exc).__name__},
        )
        return error_response(
            request,
            "internal_error",
            "Une erreur interne est survenue. Réessayez ultérieurement.",
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
