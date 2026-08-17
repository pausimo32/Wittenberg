"""Fabrique de petits modèles ONNX pour les tests.

Ces modèles sont minuscules mais authentiques : ils exercent le vrai chemin de
chargement et d'inférence d'ONNX Runtime, sans peser sur la durée des tests.

Aucun de ces artefacts n'est utilisable en production : leurs poids sont
arbitraires et n'ont jamais vu la moindre image de Chladni.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def make_cnn(
    path: Path,
    *,
    num_classes: int = 4,
    height: int = 32,
    width: int = 32,
    channels: int = 1,
    layout: str = "NHWC",
    with_softmax: bool = True,
    filters: int = 4,
) -> Path:
    """Écrit un CNN minimal : Conv → GlobalAveragePool → Gemm [→ Softmax].

    Les poids sont déterministes, donc les prédictions le sont aussi.
    """
    input_shape = [1, height, width, channels] if layout == "NHWC" else [1, channels, height, width]

    nodes = []
    initializers = []

    conv_input = "input"
    if layout == "NHWC":
        nodes.append(helper.make_node("Transpose", ["input"], ["nchw"], perm=[0, 3, 1, 2]))
        conv_input = "nchw"

    rng = np.random.default_rng(20260815)
    kernel = rng.standard_normal((filters, channels, 3, 3)).astype(np.float32) * 0.3
    initializers.append(numpy_helper.from_array(kernel, "conv_w"))
    nodes.append(
        helper.make_node(
            "Conv", [conv_input, "conv_w"], ["conv"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]
        )
    )
    nodes.append(helper.make_node("Relu", ["conv"], ["relu"]))
    nodes.append(helper.make_node("GlobalAveragePool", ["relu"], ["pooled"]))
    nodes.append(helper.make_node("Flatten", ["pooled"], ["flat"], axis=1))

    dense = rng.standard_normal((filters, num_classes)).astype(np.float32) * 2.0
    bias = np.zeros((num_classes,), dtype=np.float32)
    initializers.append(numpy_helper.from_array(dense, "dense_w"))
    initializers.append(numpy_helper.from_array(bias, "dense_b"))
    nodes.append(helper.make_node("Gemm", ["flat", "dense_w", "dense_b"], ["logits"]))

    output_name = "logits"
    if with_softmax:
        nodes.append(helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1))
        output_name = "probabilities"

    graph = helper.make_graph(
        nodes,
        "wittenberg-test-cnn",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, num_classes])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=9)
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(model.SerializeToString())
    return path
