"""Adaptateur TensorFlow Lite, via ``tflite_runtime`` ou ``tf.lite``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from wittenberg.model.adapters.base import AdapterError, ModelAdapter, as_score_vector


def _load_interpreter(artifact: Path) -> Any:
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore[import-not-found]

        return Interpreter(model_path=str(artifact))
    except ImportError:
        pass
    try:
        import tensorflow as tf

        return tf.lite.Interpreter(model_path=str(artifact))
    except ImportError as exc:  # pragma: no cover - dépend de l'installation
        raise AdapterError(
            "aucun runtime TFLite disponible ; installez `tflite-runtime` ou `tensorflow`"
        ) from exc


class TFLiteAdapter(ModelAdapter):
    """Charge un ``.tflite`` et l'exécute sur CPU."""

    #: L'interpréteur possède un état de tenseurs partagé : on sérialise.
    thread_safe = False

    def __init__(self, artifact: Path) -> None:
        if not artifact.is_file():
            raise AdapterError(f"artefact TFLite introuvable : {artifact}")
        try:
            self._interpreter = _load_interpreter(artifact)
            self._interpreter.allocate_tensors()
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError(f"chargement TFLite impossible : {exc}") from exc

        inputs = self._interpreter.get_input_details()
        outputs = self._interpreter.get_output_details()
        if len(inputs) != 1:
            raise AdapterError(
                f"le modèle déclare {len(inputs)} entrées ; une seule est prise en charge"
            )
        self._input_detail = inputs[0]
        self._output_detail = outputs[0]
        self._input_shape = tuple(int(d) for d in self._input_detail["shape"])
        out_shape = [int(d) for d in self._output_detail["shape"]]
        self._output_size = out_shape[-1] if out_shape else None

    @property
    def input_shape(self) -> tuple[int | None, ...]:
        return self._input_shape

    @property
    def output_size(self) -> int | None:
        return self._output_size

    @property
    def runtime_version(self) -> str:
        return "tflite"

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        try:
            self._interpreter.set_tensor(self._input_detail["index"], tensor)
            self._interpreter.invoke()
            raw = self._interpreter.get_tensor(self._output_detail["index"])
        except Exception as exc:
            raise AdapterError(f"échec de l'inférence TFLite : {exc}") from exc
        return as_score_vector(raw)
