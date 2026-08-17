"""Adaptateur Keras / TensorFlow.

Attention : TensorFlow ne publie pas de roue pour Python 3.14. Pour servir un
artefact ``.keras``, ``.h5`` ou SavedModel, utilisez Python 3.11–3.13, ou
exportez le modèle en ONNX (voir ``backend/models/README.md``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wittenberg.model.adapters.base import AdapterError, ModelAdapter, as_score_vector


class KerasAdapter(ModelAdapter):
    """Charge un modèle Keras et l'exécute sur CPU."""

    #: `Model.predict` n'offre aucune garantie de réentrance : on sérialise.
    thread_safe = False

    def __init__(self, artifact: Path) -> None:
        try:
            import tensorflow as tf
        except ImportError as exc:  # pragma: no cover - dépend de l'installation
            raise AdapterError(
                "tensorflow n'est pas installé (aucune roue pour Python 3.14). "
                "Utilisez Python 3.11–3.13, ou exportez le modèle en ONNX."
            ) from exc

        if not artifact.exists():
            raise AdapterError(f"artefact Keras introuvable : {artifact}")

        try:
            self._model = tf.keras.models.load_model(str(artifact), compile=False)
        except Exception as exc:
            raise AdapterError(f"chargement Keras impossible : {exc}") from exc

        self._input_shape = tuple(
            int(d) if d is not None else None for d in self._model.input_shape
        )
        out_shape = self._model.output_shape
        last = out_shape[-1] if isinstance(out_shape, tuple) else None
        self._output_size = int(last) if isinstance(last, int) else None
        self._runtime_version = str(tf.__version__)

    @property
    def input_shape(self) -> tuple[int | None, ...]:
        return self._input_shape

    @property
    def output_size(self) -> int | None:
        return self._output_size

    @property
    def runtime_version(self) -> str:
        return f"tensorflow {self._runtime_version}"

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        try:
            # `__call__` évite la surcharge de traçage de `predict` sur un lot de 1.
            outputs = self._model(tensor, training=False)
        except Exception as exc:
            raise AdapterError(f"échec de l'inférence Keras : {exc}") from exc
        return as_score_vector(np.asarray(outputs))
