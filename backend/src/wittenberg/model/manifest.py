"""Manifeste du modèle : source unique de vérité.

Le manifeste décrit l'artefact, la géométrie d'entrée attendue, la version du
prétraitement et — surtout — la correspondance exacte ``index → fréquence``.

L'ordre des classes n'est JAMAIS déduit d'un tri alphabétique de noms de
dossiers : il provient exclusivement de ce fichier, où chaque classe porte son
index explicite.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

MANIFEST_SCHEMA_VERSION = 1


class ManifestError(Exception):
    """Le manifeste est absent, illisible ou incohérent."""


class Framework(StrEnum):
    ONNX = "onnx"
    KERAS = "keras"
    TFLITE = "tflite"


class ArtifactFormat(StrEnum):
    ONNX = "onnx"
    KERAS_FILE = "keras"
    HDF5 = "h5"
    SAVED_MODEL = "saved_model"
    TFLITE = "tflite"


class Layout(StrEnum):
    NHWC = "NHWC"
    NCHW = "NCHW"


class NormalizationMode(StrEnum):
    #: pixels / 255 → [0, 1]
    SCALE_01 = "scale_01"
    #: (pixels / 127.5) - 1 → [-1, 1]
    SCALE_SIGNED = "scale_signed"
    #: (pixels/255 - mean) / std, avec mean/std par canal
    MEAN_STD = "mean_std"
    #: aucune mise à l'échelle, valeurs brutes [0, 255]
    RAW = "raw"


class ResizeMode(StrEnum):
    #: conserve le rapport d'aspect et complète par du remplissage
    PAD = "pad"
    #: déforme l'image pour atteindre exactement la cible
    STRETCH = "stretch"


class OutputActivation(StrEnum):
    #: le modèle renvoie déjà des probabilités normalisées
    SOFTMAX = "softmax"
    #: le modèle renvoie des logits ; le service applique softmax
    LOGITS = "logits"


class StrictModel(BaseModel):
    """Base interdisant les champs inconnus, pour attraper les fautes de frappe."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class ModelSection(StrictModel):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    framework: Framework
    format: ArtifactFormat
    artifact: str = Field(
        min_length=1,
        description="Nom de fichier de l'artefact, relatif au dossier du manifeste.",
    )
    sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-fA-F]{64}$",
        description="Empreinte SHA-256 de l'artefact. Vivement recommandée.",
    )
    trained_at: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _framework_matches_format(self) -> Self:
        compatible: dict[Framework, set[ArtifactFormat]] = {
            Framework.ONNX: {ArtifactFormat.ONNX},
            Framework.KERAS: {
                ArtifactFormat.KERAS_FILE,
                ArtifactFormat.HDF5,
                ArtifactFormat.SAVED_MODEL,
            },
            Framework.TFLITE: {ArtifactFormat.TFLITE},
        }
        if self.format not in compatible[self.framework]:
            raise ValueError(
                f"format '{self.format.value}' incompatible avec framework '{self.framework.value}'"
            )
        return self


class NormalizationSection(StrictModel):
    mode: NormalizationMode = NormalizationMode.SCALE_01
    mean: list[float] | None = None
    std: list[float] | None = None

    @model_validator(mode="after")
    def _mean_std_required(self) -> Self:
        if self.mode is NormalizationMode.MEAN_STD:
            if not self.mean or not self.std:
                raise ValueError("mode 'mean_std' exige les tableaux 'mean' et 'std'")
            if len(self.mean) != len(self.std):
                raise ValueError("'mean' et 'std' doivent avoir la même longueur")
            if any(s == 0 for s in self.std):
                raise ValueError("'std' ne peut pas contenir de zéro")
        return self


class InputSection(StrictModel):
    height: int = Field(gt=0, le=4096)
    width: int = Field(gt=0, le=4096)
    channels: Literal[1, 3] = 1
    layout: Layout = Layout.NHWC
    dtype: Literal["float32", "float16", "uint8"] = "float32"
    value_range: tuple[float, float] = (0.0, 1.0)
    normalization: NormalizationSection = Field(default_factory=NormalizationSection)

    @model_validator(mode="after")
    def _check(self) -> Self:
        low, high = self.value_range
        if low >= high:
            raise ValueError("value_range doit être croissant, ex. [0.0, 1.0]")
        norm = self.normalization
        if (
            norm.mode is NormalizationMode.MEAN_STD
            and norm.mean is not None
            and len(norm.mean) not in (1, self.channels)
        ):
            raise ValueError(
                f"'mean'/'std' doivent contenir 1 ou {self.channels} valeurs "
                f"(channels={self.channels})"
            )
        return self

    @property
    def tensor_shape(self) -> tuple[int, int, int, int]:
        """Forme attendue par le modèle, lot de taille 1 inclus."""
        if self.layout is Layout.NHWC:
            return (1, self.height, self.width, self.channels)
        return (1, self.channels, self.height, self.width)


class EdgeDetectionSection(StrictModel):
    enabled: bool = False
    method: Literal["canny"] = "canny"
    low_threshold: int = Field(default=100, ge=0, le=255)
    high_threshold: int = Field(default=200, ge=0, le=255)
    aperture_size: Literal[3, 5, 7] = 3

    @model_validator(mode="after")
    def _thresholds_ordered(self) -> Self:
        if self.enabled and self.low_threshold >= self.high_threshold:
            raise ValueError("low_threshold doit être inférieur à high_threshold")
        return self


class PlateDetectionSection(StrictModel):
    enabled: bool = True
    #: proportion du côté conservée par le recadrage central de repli
    fallback_center_crop: float = Field(default=0.9, gt=0.1, le=1.0)
    #: surface minimale du quadrilatère détecté, en fraction de l'image
    min_area_ratio: float = Field(default=0.15, gt=0.0, le=1.0)
    #: écart maximal toléré au rectangle parfait (0 = rectangle exact)
    max_skew: float = Field(default=0.35, gt=0.0, le=1.0)


class ContrastSection(StrictModel):
    mode: Literal["none", "minmax", "clahe"] = "minmax"
    clip_limit: float = Field(default=2.0, gt=0)
    tile_grid: int = Field(default=8, ge=1, le=64)


class DenoiseSection(StrictModel):
    mode: Literal["none", "median", "gaussian"] = "none"
    kernel_size: int = Field(default=3, ge=3, le=15)

    @model_validator(mode="after")
    def _odd_kernel(self) -> Self:
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size doit être impair")
        return self


class PreprocessingSection(StrictModel):
    version: str = Field(min_length=1)
    grayscale: bool = True
    plate_detection: PlateDetectionSection = Field(default_factory=PlateDetectionSection)
    contrast: ContrastSection = Field(default_factory=ContrastSection)
    denoise: DenoiseSection = Field(default_factory=DenoiseSection)
    edge_detection: EdgeDetectionSection = Field(default_factory=EdgeDetectionSection)
    resize_mode: ResizeMode = ResizeMode.PAD
    pad_value: int = Field(default=0, ge=0, le=255)


class ClassEntry(StrictModel):
    index: int = Field(ge=0)
    label: str = Field(min_length=1)
    frequency_hz: float = Field(gt=0)


class OutputSection(StrictModel):
    activation: OutputActivation = OutputActivation.SOFTMAX


class DecisionSection(StrictModel):
    min_confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    #: écart minimal entre la 1re et la 2e probabilité pour juger fiable
    min_margin: float = Field(default=0.0, ge=0.0, le=1.0)
    top_k: int = Field(default=3, ge=1, le=50)


class DatasetSection(StrictModel):
    id: str | None = None
    date: str | None = None
    images_per_class: int | None = Field(default=None, ge=0)
    notes: str | None = None


class ModelManifest(StrictModel):
    """Manifeste versionné complet."""

    manifest_version: int = Field(default=MANIFEST_SCHEMA_VERSION, ge=1)
    model: ModelSection
    input: InputSection
    preprocessing: PreprocessingSection
    output: OutputSection = Field(default_factory=OutputSection)
    classes: Annotated[list[ClassEntry], Field(min_length=2)]
    decision: DecisionSection = Field(default_factory=DecisionSection)
    dataset: DatasetSection = Field(default_factory=DatasetSection)

    #: chemin du fichier chargé — renseigné par `load_manifest`, hors schéma JSON
    source_path: Path | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _validate_classes(self) -> Self:
        indices = [c.index for c in self.classes]
        expected = list(range(len(self.classes)))
        if sorted(indices) != expected:
            missing = sorted(set(expected) - set(indices))
            duplicates = sorted({i for i in indices if indices.count(i) > 1})
            raise ValueError(
                "les index de classes doivent être uniques et couvrir "
                f"0..{len(self.classes) - 1} — manquants : {missing}, "
                f"dupliqués : {duplicates}"
            )
        labels = [c.label for c in self.classes]
        if len(set(labels)) != len(labels):
            raise ValueError("les libellés de classes doivent être uniques")
        frequencies = [c.frequency_hz for c in self.classes]
        if len(set(frequencies)) != len(frequencies):
            raise ValueError("les fréquences des classes doivent être uniques")
        if self.decision.top_k > len(self.classes):
            raise ValueError("decision.top_k ne peut pas dépasser le nombre de classes")
        if self.manifest_version > MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"manifest_version {self.manifest_version} non pris en charge "
                f"(maximum : {MANIFEST_SCHEMA_VERSION})"
            )
        return self

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def ordered_classes(self) -> list[ClassEntry]:
        """Classes triées par index — l'ordre exact des sorties du modèle."""
        return sorted(self.classes, key=lambda c: c.index)

    @property
    def artifact_path(self) -> Path:
        """Chemin absolu de l'artefact, résolu depuis le dossier du manifeste."""
        artifact = Path(self.model.artifact)
        if artifact.is_absolute():
            return artifact
        base = self.source_path.parent if self.source_path else Path.cwd()
        return (base / artifact).resolve()

    def class_at(self, index: int) -> ClassEntry:
        return self.ordered_classes[index]


def compute_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 d'un fichier, ou du contenu concaténé d'un SavedModel."""
    digest = hashlib.sha256()
    if path.is_dir():
        for child in sorted(p for p in path.rglob("*") if p.is_file()):
            digest.update(str(child.relative_to(path)).replace("\\", "/").encode())
            with child.open("rb") as handle:
                while block := handle.read(chunk_size):
                    digest.update(block)
    else:
        with path.open("rb") as handle:
            while block := handle.read(chunk_size):
                digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> ModelManifest:
    """Charge et valide un manifeste.

    :raises ManifestError: fichier absent, JSON invalide ou schéma non respecté.
    """
    if not path.exists():
        raise ManifestError(f"manifeste introuvable : {path}")
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"manifeste JSON invalide ({path.name}) : {exc}") from exc
    if not isinstance(raw, dict):
        raise ManifestError("le manifeste doit contenir un objet JSON à la racine")
    try:
        manifest = ModelManifest.model_validate(raw)
    except Exception as exc:  # pydantic.ValidationError et dérivés
        raise ManifestError(f"manifeste invalide ({path.name}) : {exc}") from exc
    manifest.source_path = path.resolve()
    return manifest
