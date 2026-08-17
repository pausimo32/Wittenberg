"""Adaptateur ONNX Runtime — chemin de service recommandé.

Un modèle Keras entraîné pour ce projet peut être exporté en ONNX via
``tf2onnx``. ONNX Runtime possède des roues CPU pour Python 3.11 à 3.14, là où
TensorFlow s'arrête à 3.13.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from wittenberg.model.adapters.base import AdapterError, ModelAdapter, as_score_vector


class OnnxAdapter(ModelAdapter):
    """Charge un ``.onnx`` et l'exécute sur CPU."""

    #: `InferenceSession.run` est sûr entre fils d'exécution.
    thread_safe = True

    def __init__(self, artifact: Path, intra_op_threads: int = 1) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - dépend de l'installation
            raise AdapterError(
                "onnxruntime n'est pas installé ; exécutez "
                "`pip install onnxruntime` pour servir un modèle ONNX"
            ) from exc

        if not artifact.is_file():
            raise AdapterError(f"artefact ONNX introuvable : {artifact}")

        options = ort.SessionOptions()
        if intra_op_threads > 0:
            options.intra_op_num_threads = intra_op_threads
            options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            self._session = ort.InferenceSession(
                str(artifact), sess_options=options, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:
            raise AdapterError(f"chargement ONNX impossible : {exc}") from exc

        inputs = self._session.get_inputs()
        if len(inputs) != 1:
            raise AdapterError(
                f"le modèle déclare {len(inputs)} entrées ; une seule est prise en charge"
            )
        self._input_name = inputs[0].name
        self._input_shape = tuple(dim if isinstance(dim, int) else None for dim in inputs[0].shape)
        outputs = self._session.get_outputs()
        if not outputs:
            raise AdapterError("le modèle ne déclare aucune sortie")
        self._output_name = outputs[0].name
        raw_out = outputs[0].shape
        last = raw_out[-1] if raw_out else None
        self._output_size = last if isinstance(last, int) else None
        self._runtime_version = str(ort.__version__)

    @property
    def input_shape(self) -> tuple[int | None, ...]:
        return self._input_shape

    @property
    def output_size(self) -> int | None:
        return self._output_size

    @property
    def runtime_version(self) -> str:
        return f"onnxruntime {self._runtime_version}"

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        feed: dict[str, Any] = {self._input_name: tensor}
        try:
            outputs = self._session.run([self._output_name], feed)
        except Exception as exc:
            raise AdapterError(f"échec de l'inférence ONNX : {exc}") from exc
        return as_score_vector(outputs[0])
