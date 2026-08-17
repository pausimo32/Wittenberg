# Wittenberg — image de service d'inférence (CPU uniquement)
#
# Python 3.12 : compatible avec ONNX Runtime, OpenCV et — si vous servez un
# artefact Keras — TensorFlow, qui ne publie pas de roue pour Python 3.14.
#
# L'artefact du modèle n'est PAS copié dans l'image : il est monté en volume
# au démarrage (voir backend/models/README.md).

FROM python:3.12-slim-bookworm AS base

# Dépendances système d'OpenCV en version « headless ».
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Les dépendances sont installées avant le code : la couche reste en cache
# tant que requirements.txt ne change pas.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/model_manifest.example.json ./models/

# Exécution sans privilèges.
RUN useradd --create-home --uid 10001 wittenberg \
    && chown -R wittenberg:wittenberg /app
USER wittenberg

ENV WITTENBERG_MODEL_MANIFEST_PATH=/models/model_manifest.json \
    WITTENBERG_LOG_LEVEL=INFO \
    WITTENBERG_STRICT_STARTUP=false

EXPOSE 8000

# /health atteste la vivacité du processus. Pour que l'orchestrateur exige
# aussi un modèle chargé, remplacez /health par /ready.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail --silent http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "wittenberg.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
