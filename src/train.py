"""Training — `docs/04-model-spec.md` §3, and nothing that is not in it.

One run writes one `runs/<timestamp>/` and updates `runs/latest.txt`. Every hyperparameter
below is quoted from §3; none of them is a knob to be turned by looking at a number.

**Determinism is a reported property, not a nicety.** `CLAUDE.md` rule 5 and §3: seed 42 for
`random`, `numpy` and `torch`, `torch.use_deterministic_algorithms(True)`, and
`CUBLAS_WORKSPACE_CONFIG=:4096:8` — which has to be set *before* CUDA initialises, so
`configure_determinism()` sets it in the environment and is called first. The seed goes into
`metrics.json`, because `docs/06` §4 promises re-training reproduces reported metrics.

**The split is not chosen here.** It is read from `metadata.csv`, where `tasks.py split` put
it, grouped by `session_id` (`docs/adr/0002`). Nothing in this module can produce a random
split even by accident, which is the point.

**Never report an accuracy measured on synthetic data.** `data/SYNTHETIC` marks a working
copy whose captures came from `tools/synth_capture.py`; when it is present, `metrics.json`
carries `"synthetic": true` and a refusal to be quoted. A CNN reaches ~100 % on a closed-form
mode shape, and that number measures how well it inverted the formula that drew the images.

**A known CUDA caveat, for the day this first runs on the RTX 5070.** `AdaptiveAvgPool2d` has
no deterministic CUDA backward in some PyTorch builds, and with
`use_deterministic_algorithms(True)` that raises rather than degrading silently. If it does,
the fix is `x.mean(dim=(2, 3))` — the same function, a deterministic kernel — not switching
determinism off. Nothing here has been able to test that path: F4a runs on CPU.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

import runs as runs_module
from dataset import class_weights, load_split, to_model_input
from environment import opencv_signature
from metadata import is_synthetic
from model import MODEL_NAMES, build_model, classes_from_config, parameter_count
from preprocess import load_config

ROOT = Path(__file__).resolve().parents[1]

#: CLAUDE.md rule 5. Reported in every metrics.json.
SEED = 42

#: docs/04 §3, verbatim. Not tunable by inspection — an ablation moves these, not a hunch.
HYPERPARAMETERS = {
    "optimizer": "AdamW",
    "lr": 1e-3,
    "lr_min": 1e-5,
    "scheduler": "CosineAnnealingLR",
    "batch_size": 32,
    "epochs": 80,
    "early_stopping_patience": 12,
    "early_stopping_metric": "val_loss",
    "label_smoothing": 0.05,
    "amp": False,
}


class TrainingError(RuntimeError):
    """The run cannot start, or cannot be trusted if it did."""


def configure_determinism(seed: int = SEED) -> dict[str, Any]:
    """Everything §3 asks for, in the order it has to happen.

    `CUBLAS_WORKSPACE_CONFIG` is read by cuBLAS when the CUDA context is created, so setting
    it after the first tensor lands on the GPU is setting it too late — silently.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    return {
        "seed": seed,
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }


def select_device() -> torch.device:
    """CUDA if it is there, CPU otherwise. CPU is supported, not degraded (ADR 0005)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    data_root: Path | str,
    model_name: str = "wittenberg_v1",
    runs_root: Path | str | None = None,
    epochs: int | None = None,
    seed: int = SEED,
    classes_path: Path | str | None = None,
) -> Path:
    """Train one model and write one run directory. Returns the run directory.

    `classes_path` defaults to `config/classes.json`, which is what production uses. It is a
    parameter because the class list belongs to a *dataset*: a test that builds its own
    dataset has to be able to build its own frozen list too, rather than depending on
    whatever happens to be checked in.
    """
    if model_name not in MODEL_NAMES:
        raise TrainingError(f"unknown model {model_name!r} — one of {', '.join(MODEL_NAMES)}")

    determinism = configure_determinism(seed)
    device = select_device()
    data_root = Path(data_root)

    config = load_config()
    classes = classes_from_config(classes_path)
    n_classes = len(classes)

    train_split = load_split(data_root, "train", classes)
    val_split = load_split(data_root, "val", classes)

    weights = class_weights(train_split, n_classes)
    model = build_model(model_name, n_classes).to(device)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=HYPERPARAMETERS["label_smoothing"],
        weight=None if weights is None else weights.to(device),
    )
    optimiser = torch.optim.AdamW(model.parameters(), lr=HYPERPARAMETERS["lr"])
    total_epochs = int(epochs or HYPERPARAMETERS["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=total_epochs, eta_min=HYPERPARAMETERS["lr_min"]
    )

    generator = torch.Generator().manual_seed(seed)
    history: list[dict[str, float]] = []
    best = {"val_loss": float("inf"), "epoch": -1, "state": None}
    started = time.perf_counter()

    for epoch in range(total_epochs):
        train_loss, train_acc = _train_one_epoch(
            model, train_split, criterion, optimiser, config, device, generator
        )
        val_loss, val_acc = evaluate_loss(model, val_split, criterion, config, device)
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "lr": float(optimiser.param_groups[0]["lr"]),
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        if val_loss < best["val_loss"]:
            # §3: restore best weights. Kept on the CPU so a long run does not hold a
            # second copy of the model in VRAM for eighty epochs.
            best = {
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }

        if epoch - int(best["epoch"]) >= HYPERPARAMETERS["early_stopping_patience"]:
            break

    if best["state"] is None:
        raise TrainingError("no epoch completed — nothing to write")
    model.load_state_dict(best["state"])
    elapsed = time.perf_counter() - started

    run_dir = runs_module.new_run_dir(runs_root)
    _write_history(run_dir, history)
    torch.save(
        {
            "model": model_name,
            "n_classes": n_classes,
            "classes": classes,
            "state_dict": model.state_dict(),
            "preprocess_sha256": runs_module.preprocess_hash(config),
            "preprocess_version": config.get("version"),
            # Both keys of the §4 guard travel inside the checkpoint as well as beside it:
            # `model.pt` ships with the repository (ADR 0005) and has to carry its own
            # provenance. The OpenCV version is not decoration — see docs/adr/0006.
            **{k: v for k, v in opencv_signature().items() if not k.startswith("opencv_dist")},
        },
        run_dir / "model.pt",
    )
    _write_config(run_dir, model_name, config, classes, determinism, total_epochs, weights)
    _write_metrics(
        run_dir,
        model_name,
        model,
        history,
        best,
        train_split,
        val_split,
        elapsed,
        device,
        data_root,
    )
    runs_module.write_latest(run_dir, runs_root)
    return run_dir


# --------------------------------------------------------------------------- #
# the loop
# --------------------------------------------------------------------------- #

def _train_one_epoch(model, split, criterion, optimiser, config, device, generator):
    """One pass over the resident tensor, indexed by a seeded random permutation.

    §3: the dataset lives in memory, so there is no `DataLoader` and no worker processes —
    both of which would be a source of nondeterminism this project has promised not to have.
    """
    model.train()
    order = torch.randperm(split.size, generator=generator)
    batch_size = int(HYPERPARAMETERS["batch_size"])
    total_loss, correct = 0.0, 0

    for start in range(0, len(order), batch_size):
        index = order[start: start + batch_size]
        x = to_model_input(split.images[index], config, device)
        y = split.labels[index].to(device)

        optimiser.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimiser.step()

        total_loss += float(loss.detach()) * len(index)
        correct += int((logits.detach().argmax(1) == y).sum())

    return total_loss / split.size, correct / split.size


@torch.no_grad()
def evaluate_loss(model, split, criterion, config, device):
    model.eval()
    batch_size = int(HYPERPARAMETERS["batch_size"])
    total_loss, correct = 0.0, 0
    for start in range(0, split.size, batch_size):
        x = to_model_input(split.images[start: start + batch_size], config, device)
        y = split.labels[start: start + batch_size].to(device)
        logits = model(x)
        total_loss += float(criterion(logits, y)) * len(y)
        correct += int((logits.argmax(1) == y).sum())
    return total_loss / split.size, correct / split.size


@torch.no_grad()
def predict(model, split, config, device) -> tuple[torch.Tensor, torch.Tensor]:
    """`(probabilities, predictions)` for a whole split. Softmax lives here, not in the model."""
    model.eval()
    batch_size = int(HYPERPARAMETERS["batch_size"])
    chunks = []
    for start in range(0, split.size, batch_size):
        x = to_model_input(split.images[start: start + batch_size], config, device)
        chunks.append(torch.softmax(model(x), dim=1).cpu())
    probabilities = torch.cat(chunks)
    return probabilities, probabilities.argmax(1)


# --------------------------------------------------------------------------- #
# artefacts
# --------------------------------------------------------------------------- #

def _write_history(run_dir: Path, history: list[dict[str, float]]) -> Path:
    path = run_dir / "history.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return path


def _write_config(run_dir, model_name, config, classes, determinism, epochs, weights) -> Path:
    """`config.json` — everything needed to say what this run *was*.

    The preprocessing hash is the load-bearing field: `docs/03` §4 makes it what the API
    asserts at startup and what `tasks.py check-model` checks from the command line.
    """
    payload = {
        "model": model_name,
        "n_classes": len(classes),
        "classes": classes,
        "hyperparameters": {**HYPERPARAMETERS, "epochs": epochs},
        "class_weights": None if weights is None else [round(float(w), 6) for w in weights],
        "determinism": determinism,
        "preprocess_version": config.get("version"),
        "preprocess_sha256": runs_module.preprocess_hash(config),
        "preprocess_sha256_method": (
            "sha256 of json.dumps(resolved config/preprocess.json, sort_keys=True, "
            "separators=(',', ':')) — see src/runs.py"
        ),
        "preprocess_config": config,
        # Flat, at the top level, because `src/runs.py` gates on these two and a gate should
        # not have to reach into a nested "environment" bag to find its key.
        **opencv_signature(),
        "environment": _environment(),
    }
    path = run_dir / "config.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _write_metrics(
    run_dir, model_name, model, history, best, train_split, val_split, elapsed, device, data_root
) -> Path:
    # The session ids are the reliable signal: `data/SYNTHETIC` is a convenience marker
    # that can be deleted, but `src/metadata.py` refuses to mix provenance, so a single
    # SYNTH-* session in a split means the whole run is synthetic.
    synthetic = (
        (Path(data_root) / "SYNTHETIC").exists()
        or any(is_synthetic(s) for s in (*train_split.sessions, *val_split.sessions))
    )
    payload = {
        "run": run_dir.name,
        "model": model_name,
        "seed": SEED,
        "device": device.type,
        "synthetic": synthetic,
        "parameters": parameter_count(model),
        "epochs_run": len(history),
        "best_epoch": int(best["epoch"]),
        "best_val_loss": round(float(best["val_loss"]), 6),
        "best_val_accuracy": round(float(best["val_acc"]), 6),
        "final_train_loss": round(float(history[-1]["train_loss"]), 6),
        "train_seconds": round(elapsed, 1),
        "splits": {
            "train": _split_summary(train_split),
            "val": _split_summary(val_split),
        },
    }
    if synthetic:
        payload["do_not_report"] = (
            "This run used tools/synth_capture.py sessions. The images come from a "
            "closed-form mode shape, so any accuracy here measures how well the network "
            "inverted the formula that drew them, not anything about a plate. No number "
            "from this file may appear in the TDR (docs/00-master-plan.md F2.5)."
        )
    path = run_dir / "metrics.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _split_summary(split) -> dict[str, Any]:
    return {
        "images": split.size,
        "sessions": sorted(set(split.sessions)),
        "per_class": {str(k): v for k, v in sorted(split.class_counts.items())},
    }


def _environment() -> dict[str, Any]:
    """The rest of the environment — recorded for the TDR, not gated on.

    `docs/06` §4 promises a reader can reproduce the run; that needs the whole picture. Only
    the OpenCV pair is a gate, because it is the only one measured to change the pixels
    (`docs/adr/0006`).
    """
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__,
        **opencv_signature(),
    }
