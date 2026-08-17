"""Configuration du service, entièrement pilotée par variables d'environnement.

Toutes les variables portent le préfixe ``WITTENBERG_``. Voir ``.env.example``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Racine du dossier `backend/` : sert à résoudre les chemins relatifs.
BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Paramètres du service.

    Aucune valeur par défaut ne pointe vers un artefact réel : tant qu'aucun
    modèle n'est fourni, le service démarre mais se déclare non disponible.
    """

    model_config = SettingsConfigDict(
        env_prefix="WITTENBERG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # `model_*` est un espace de noms protégé par Pydantic v2 ; on le libère
        # pour pouvoir nommer les variables MODEL_PATH / MODEL_MANIFEST_PATH.
        protected_namespaces=(),
    )

    # ── Modèle ────────────────────────────────────────────────────────────
    model_path: Path | None = Field(
        default=None,
        description="Chemin de l'artefact du modèle (.onnx, .keras, .h5, .tflite, SavedModel).",
    )
    model_manifest_path: Path | None = Field(
        default=None,
        description="Chemin du manifeste JSON décrivant le modèle et ses classes.",
    )
    verify_model_hash: bool = Field(
        default=True,
        description="Vérifier le SHA-256 de l'artefact déclaré dans le manifeste.",
    )
    warmup_on_startup: bool = Field(
        default=True,
        description="Exécuter une inférence de préchauffage au démarrage.",
    )
    strict_startup: bool = Field(
        default=False,
        description=(
            "Si vrai, l'absence ou l'invalidité du modèle empêche le démarrage. "
            "Utile en production ; en développement le service reste joignable "
            "et répond 503 sur /ready."
        ),
    )

    # ── Limites d'entrée ──────────────────────────────────────────────────
    max_upload_mb: float = Field(default=10.0, gt=0, le=200)
    max_image_pixels: int = Field(default=40_000_000, gt=0)
    max_image_dimension: int = Field(default=12_000, gt=0)
    min_image_dimension: int = Field(default=32, gt=0)

    # ── Inférence ─────────────────────────────────────────────────────────
    inference_timeout_ms: int = Field(default=15_000, gt=0)
    top_k: int = Field(default=3, ge=1, le=20)
    onnx_intra_op_threads: int = Field(default=1, ge=0)

    # ── HTTP ──────────────────────────────────────────────────────────────
    allowed_origins: str = Field(
        default="http://localhost:5500,http://127.0.0.1:5500,http://localhost:8080,http://127.0.0.1:8080",
        description="Liste explicite d'origines CORS séparées par des virgules. Jamais '*'.",
    )
    api_prefix: str = Field(default="/api/v1")

    # ── Observabilité / debug ─────────────────────────────────────────────
    log_level: str = Field(default="INFO")
    enable_debug_preview: bool = Field(
        default=False,
        description=(
            "Expose l'image prétraitée en base64 dans la réponse. "
            "À réserver au développement : la réponse contient alors l'image."
        ),
    )

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, value: str) -> str:
        level = value.upper()
        allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
        if level not in allowed:
            raise ValueError(f"log_level doit être l'un de {sorted(allowed)}")
        return level

    @field_validator("model_path", "model_manifest_path")
    @classmethod
    def _resolve_path(cls, value: Path | None) -> Path | None:
        if value is None:
            return None
        expanded = Path(str(value)).expanduser()
        if not expanded.is_absolute():
            expanded = (BACKEND_ROOT / expanded).resolve()
        return expanded

    @property
    def cors_origins(self) -> list[str]:
        """Origines CORS sous forme de liste, sans entrée vide ni joker."""
        origins = [o.strip() for o in self.allowed_origins.split(",")]
        return [o for o in origins if o and o != "*"]

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Instance unique des paramètres (mémoïsée)."""
    return Settings()
