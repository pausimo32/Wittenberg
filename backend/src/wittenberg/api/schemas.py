"""Contrats de réponse de l'API.

Vocabulaire imposé : ``confidence`` et ``probability`` qualifient une sortie
individuelle. Aucun champ ne doit être présenté comme une « précision » ou un
« taux de réussite » — ces notions relèvent de l'évaluation globale du modèle.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Schema(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


class Prediction(Schema):
    """Classe retenue pour l'image analysée."""

    label: str = Field(examples=["440 Hz"])
    frequency_hz: float = Field(examples=[440.0])
    class_index: int
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Probabilité attribuée à CETTE prédiction. Ce n'est pas une précision.",
    )
    reliable: bool = Field(
        description="Faux si la confiance ou la marge est sous le seuil du manifeste."
    )
    reason: str | None = Field(
        default=None,
        description="Code expliquant le manque de fiabilité : low_confidence, ambiguous.",
    )
    margin: float = Field(description="Écart entre la première et la deuxième probabilité.")


class TopKEntry(Schema):
    """Hypothèse alternative."""

    label: str
    frequency_hz: float
    class_index: int
    probability: float = Field(ge=0.0, le=1.0)


class QualityWarningOut(Schema):
    """Indication d'amélioration de la prise de vue."""

    code: str = Field(examples=["blurry"])
    message: str
    severity: str = Field(examples=["warning", "info"])


class Quality(Schema):
    """Qualité perçue de l'image. Indicatif, jamais une certitude scientifique."""

    score: float = Field(ge=0.0, le=1.0)
    warnings: list[QualityWarningOut] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    plate_detected: bool
    plate_method: str = Field(examples=["homography", "center_crop"])


class ModelRef(Schema):
    name: str
    version: str


class PreprocessingRef(Schema):
    version: str
    steps: list[str] = Field(default_factory=list)


class PredictionResponse(Schema):
    """Réponse de ``POST /api/v1/predict``."""

    request_id: str
    prediction: Prediction
    top_k: list[TopKEntry]
    quality: Quality
    model: ModelRef
    preprocessing: PreprocessingRef
    processing_ms: float
    debug_preview_png_base64: str | None = Field(
        default=None,
        description="Image prétraitée, uniquement si WITTENBERG_ENABLE_DEBUG_PREVIEW=true.",
    )


class ClassInfo(Schema):
    index: int
    label: str
    frequency_hz: float


class InputRequirements(Schema):
    """Contraintes publiques d'entrée, destinées au frontend."""

    max_upload_mb: float
    max_image_pixels: int
    max_image_dimension: int
    min_image_dimension: int
    accepted_formats: list[str]
    field_name: str = Field(
        default="image", description="Nom du champ multipart attendu par /predict."
    )


class ModelInfoResponse(Schema):
    """Réponse de ``GET /api/v1/model-info``."""

    available: bool
    status: str
    reason: str | None = None
    missing: list[str] | None = None
    model: ModelRef | None = None
    framework: str | None = None
    runtime: str | None = None
    preprocessing_version: str | None = None
    input: dict[str, object] | None = None
    num_classes: int | None = None
    classes: list[ClassInfo] | None = None
    min_confidence: float | None = None
    min_margin: float | None = None
    top_k: int | None = None
    dataset: dict[str, object] | None = None
    requirements: InputRequirements


class HealthResponse(Schema):
    status: str = Field(examples=["ok"])
    service: str
    version: str


class ReadyResponse(Schema):
    ready: bool
    status: str
    reason: str | None = None
    missing: list[str] | None = None
    model: ModelRef | None = None
    warmup_ms: float | None = None


class ErrorBody(Schema):
    code: str = Field(examples=["corrupt_image"])
    message: str


class ErrorResponse(Schema):
    """Forme unique de toutes les erreurs. Ne contient jamais de trace interne."""

    request_id: str
    error: ErrorBody
