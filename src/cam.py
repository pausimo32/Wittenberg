"""Grad-CAM, in one place, because two places is how the TDR figure and the app diverge.

`docs/04-model-spec.md` §4.6 makes Grad-CAM part of an evaluation; `docs/05-webapp-spec.md`
§8 makes it stage 12 of the pipeline viewer — "the single most didactic image in the
project". Those are two callers, not two algorithms. `src/evaluate.py` renders it into a
matplotlib figure for `runs/<ts>/report/gradcam/`, `src/api.py` blends it over the ROI and
base64-encodes it; both get the map from here.

Extracted rather than copied for the same reason `CLAUDE.md` rule 1 forbids forking
`preprocess`: a heat map that disagrees between the annex and the screen is a heat map that
explains nothing, and nothing would report the disagreement.

No matplotlib, no I/O — the serving process must not import a plotting stack to answer a
prediction.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class CamError(RuntimeError):
    """The hooks produced nothing — `cam_layer` never ran, or the graph was detached."""


def gradcam(model: nn.Module, x: torch.Tensor, class_index: int) -> np.ndarray:
    """The Grad-CAM map for `class_index`, upsampled to `x`'s spatial size.

    `x` is one model input, `(1, 1, H, W)`. Returns `(H, W)` float32 in [0, 1] — 0 where the
    region did not move the answer, 1 at the maximum. A map that is uniformly zero (no
    positive contribution anywhere) is returned as zeros rather than normalised, since
    dividing by its maximum would turn numerical noise into a confident-looking heat map.

    Requires grad: this runs its own forward and backward through `model.cam_layer`, so the
    caller must not be inside `torch.no_grad()`/`inference_mode()`.
    """
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    layer = model.cam_layer
    handles = [
        layer.register_forward_hook(lambda _m, _i, o: activations.append(o)),
        layer.register_full_backward_hook(lambda _m, _gi, go: gradients.append(go[0])),
    ]
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        logits[0, class_index].backward()
    finally:
        for handle in handles:
            handle.remove()

    if not activations or not gradients:
        raise CamError(
            "Grad-CAM hooks produced nothing — docs/04 §4.6 and docs/05 §8 both require it"
        )

    weights = gradients[0].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activations[0]).sum(dim=1, keepdim=True))
    cam = torch.nn.functional.interpolate(
        cam, size=x.shape[-2:], mode="bilinear", align_corners=False
    )[0, 0].detach().cpu().numpy()

    peak = float(cam.max())
    return (cam / peak).astype(np.float32) if peak > 0.0 else cam.astype(np.float32)
