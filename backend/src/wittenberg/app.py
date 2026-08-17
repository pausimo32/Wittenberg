"""Fabrique de l'application FastAPI."""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from wittenberg import __version__
from wittenberg.api.errors import register_exception_handlers
from wittenberg.api.routes import api_router, health_router
from wittenberg.config import Settings, get_settings
from wittenberg.logging_config import configure_logging
from wittenberg.model.service import ModelService, ModelStatus

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"

DESCRIPTION = """
Service d'inférence de Wittenberg : estimation de la fréquence sonore à
l'origine d'un motif de Chladni photographié sur une plaque vibrante.

Les classes de fréquences proviennent exclusivement du manifeste du modèle
chargé sur le serveur. Aucune valeur n'est codée en dur dans le service.
""".strip()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Construit l'application, charge le modèle au démarrage."""
    resolved = settings or get_settings()
    configure_logging(resolved.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        service = ModelService(resolved)
        state = service.load()
        app.state.model_service = service
        if state.status is not ModelStatus.READY:
            if resolved.strict_startup:
                raise RuntimeError(f"démarrage strict : modèle indisponible — {state.reason}")
            logger.warning(
                "starting_without_model",
                extra={"status": state.status.value, "reason": state.reason},
            )
        try:
            yield
        finally:
            service.shutdown()

    app = FastAPI(
        title="Wittenberg — API d'inférence",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )
    app.state.settings = resolved

    origins = resolved.cors_origins
    if not origins:
        logger.warning("cors_no_origins", extra={"hint": "définissez WITTENBERG_ALLOWED_ORIGINS"})
    app.add_middleware(
        CORSMiddleware,
        # Liste explicite : jamais de joker, y compris en développement.
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", REQUEST_ID_HEADER],
        expose_headers=[REQUEST_ID_HEADER],
        max_age=600,
    )

    @app.middleware("http")
    async def request_id_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Attribue un identifiant à chaque requête, propagé dans les réponses."""
        incoming = request.headers.get(REQUEST_ID_HEADER, "")
        # Un identifiant fourni par le client n'est repris que s'il est inoffensif.
        request_id = (
            incoming
            if incoming and len(incoming) <= 64 and incoming.isascii() and incoming.isprintable()
            else str(uuid.uuid4())
        )
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response

    register_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(api_router, prefix=resolved.api_prefix)
    return app


app = create_app()
