"""The two architectures of `docs/04-model-spec.md` §1. No training code, no I/O.

**`baseline_paper`** is a faithful replication of Kumar & Wani (2024) §3.3 — one conv layer,
one pool, flatten, linear. It exists so the TDR can compare against the published
architecture on our own plate. Do not improve it: its numbers are a citation, not a target.

**`wittenberg_v1`** is the model that ships.

Two things this module refuses to do, both of them on purpose.

**The head does not know how many classes there are.** `n_classes` comes from
`config/classes.json`, which `tasks.py modes` generates from the mode sweep — six with the
synthetic sessions, ten to eighteen with real ones. A constant here would be correct exactly
once, and wrong silently: the checkpoint stores a class *index*, so a head sized for the
wrong list answers with a different frequency at full confidence. `n_classes` is a required
argument, and `classes_from_config()` is the one place that reads the file.

**The model does not apply softmax.** §3 uses `nn.CrossEntropyLoss(label_smoothing=0.05)`,
which takes logits. A softmax here would be applied twice, and the loss would be quietly
wrong rather than loudly broken.

The input is **NCHW** `(N, 1, 128, 128)`, while `docs/03` produces NHWC `(128, 128, 1)`. The
permute happens in `src/dataset.py`, never in `src/preprocess.py` — the preprocessing
contract is framework-agnostic and has to stay that way.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

CLASSES_PATH = Path(__file__).resolve().parents[1] / "config" / "classes.json"

MODEL_NAMES = ("wittenberg_v1", "baseline_paper")


class ModelError(ValueError):
    """The requested model cannot be built as asked."""


def classes_from_config(path: Path | str | None = None) -> list[dict[str, Any]]:
    """The frozen class list. Index is position; `freq_hz` is what it means."""
    p = Path(CLASSES_PATH if path is None else path)
    if not p.exists():
        raise ModelError(f"{p} not found — run `python tasks.py modes` first")
    payload = json.loads(p.read_text("utf-8"))
    classes = payload.get("classes") or []
    if not classes:
        raise ModelError(f"{p} lists no classes")
    if [c["index"] for c in classes] != list(range(len(classes))):
        raise ModelError(f"{p}: class indices are not 0..{len(classes) - 1} in order")
    return classes


def build_model(name: str, n_classes: int) -> nn.Module:
    """One of `MODEL_NAMES`, sized for `n_classes`."""
    if n_classes < 2:
        raise ModelError(f"n_classes must be at least 2, got {n_classes}")
    if name == "wittenberg_v1":
        return WittenbergV1(n_classes)
    if name == "baseline_paper":
        return BaselinePaper(n_classes)
    raise ModelError(f"unknown model {name!r} — one of {', '.join(MODEL_NAMES)}")


class WittenbergV1(nn.Module):
    """§1 B, exactly.

    ```
    [Conv2d(3, padding=1) -> BatchNorm2d -> ReLU] x2 -> MaxPool2d(2)    #  32ch, 64x64
    [Conv2d(3, padding=1) -> BatchNorm2d -> ReLU] x2 -> MaxPool2d(2)    #  64ch, 32x32
    [Conv2d(3, padding=1) -> BatchNorm2d -> ReLU] x2 -> MaxPool2d(2)    # 128ch, 16x16
    AdaptiveAvgPool2d(1) -> flatten -> Dropout(0.4) -> Linear(128, n)
    ```

    **GAP, not Flatten** (§2): flattening 16x16x128 gives a 32 768-wide dense layer — more
    parameters than we have training images — and ties the prediction to absolute pixel
    positions. GAP makes the model read nodal-line density and structure, which is the
    physics. Ablation A9 isolates this choice.
    """

    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            *_block(1, 32),
            *_block(32, 32),
            nn.MaxPool2d(2),
            *_block(32, 64),
            *_block(64, 64),
            nn.MaxPool2d(2),
            *_block(64, 128),
            *_block(128, 128),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(self.pool(x), 1)
        return self.classifier(self.dropout(x))

    @property
    def cam_layer(self) -> nn.Module:
        """The last convolution — where Grad-CAM (§4.6) takes its activations."""
        return self.features[-4]


class BaselinePaper(nn.Module):
    """§1 A — Kumar & Wani (2024) §3.3, replicated, not improved.

    `Conv2d(1, 64, 3)` + ReLU, `MaxPool2d`, flatten, `Linear(n_classes)`. The flatten is the
    published design and is left alone deliberately: A7 compares it against `wittenberg_v1`,
    and a "fixed" baseline would make that comparison meaningless.
    """

    def __init__(self, n_classes: int, input_size: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3),
            nn.ReLU(),          # not inplace — see `_block`
            nn.MaxPool2d(2),
        )
        side = (input_size - 2) // 2
        self.classifier = nn.Linear(64 * side * side, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.flatten(self.features(x), 1))

    @property
    def cam_layer(self) -> nn.Module:
        return self.features[0]


def _block(in_ch: int, out_ch: int) -> list[nn.Module]:
    """Conv -> BatchNorm -> ReLU. BN before ReLU, standard ordering (§2).

    **The convolutions keep their bias**, even though the BatchNorm that follows makes it
    redundant. `bias=False` is the usual optimisation and it removes exactly 448 parameters
    (32+32+64+64+128+128), which is the difference between this trunk and the 287 328 §1
    states — and therefore between a TDR that quotes 288 876 for 12 classes and one that
    quotes 288 428. The spec writes `Conv2d(3, padding=1)` with PyTorch's default, so that
    is what this is. Dropping the bias is an architecture change and needs §1 to change first.
    """
    return [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        # **Not `inplace=True`.** Grad-CAM (§4.6) attaches a full backward hook to
        # `cam_layer`, which makes that layer's output a view; an in-place ReLU on a view
        # created inside a hook raises "the autograd logic to handle view+inplace would
        # override the custom backward". `baseline_paper` hits it directly — its ReLU
        # follows the hooked conv with nothing between — and this trunk is one refactor
        # away from hitting it too. The saving is a few MB at batch 32 and §4.6 is
        # non-negotiable, so the trade is not close.
        nn.ReLU(),
    ]


def parameter_count(model: nn.Module) -> dict[str, int]:
    """§1: quote this, do not quote a remembered figure — it depends on `n_classes`."""
    return {
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total": sum(p.numel() for p in model.parameters()),
        "buffers": sum(b.numel() for b in model.buffers()),
    }
