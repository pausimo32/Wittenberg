"""Sélection de l'adaptateur d'inférence d'après le manifeste."""

from __future__ import annotations

from pathlib import Path

from wittenberg.model.adapters.base import AdapterError, ModelAdapter
from wittenberg.model.manifest import Framework, ModelManifest


def build_adapter(
    manifest: ModelManifest,
    artifact: Path,
    *,
    onnx_intra_op_threads: int = 1,
) -> ModelAdapter:
    """Instancie l'adaptateur correspondant au framework déclaré.

    Les runtimes sont importés paresseusement : le service peut démarrer sans
    TensorFlow installé tant que le manifeste ne le réclame pas.
    """
    framework = manifest.model.framework

    if framework is Framework.ONNX:
        from wittenberg.model.adapters.onnx_adapter import OnnxAdapter

        return OnnxAdapter(artifact, intra_op_threads=onnx_intra_op_threads)

    if framework is Framework.KERAS:
        from wittenberg.model.adapters.keras_adapter import KerasAdapter

        return KerasAdapter(artifact)

    if framework is Framework.TFLITE:
        from wittenberg.model.adapters.tflite_adapter import TFLiteAdapter

        return TFLiteAdapter(artifact)

    raise AdapterError(f"framework non pris en charge : {framework}")
