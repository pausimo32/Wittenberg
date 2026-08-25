"""Evaluation — the six mandatory outputs of `docs/04-model-spec.md` §4, all six.

§4.6 is explicit that a run without Grad-CAM is not evaluated, so this module produces it or
raises; there is no partial success. Everything lands in `runs/<ts>/report/` and the headline
numbers are merged back into `runs/<ts>/metrics.json`, which is one of the two files
`CLAUDE.md` rule 4 commits.

**Nothing here opens a window.** `CLAUDE.md` rule 2 — the Agg backend is selected before
`pyplot` is imported, and no figure is ever shown; every figure is written and closed.

**The confidence interval is Wilson, and it is labelled with what it is an interval over.**
§4.1: under leave-one-session-out the independent unit is the *session*, not the frame. A
burst of 60 frames with the rig untouched is one observation (`docs/02` §5), so the
frame-level interval is reported — because it is asked for — and marked as the optimistic
one. Reporting only the frame-level number is how a 5-image test set comes to look precise.

**The frequency-distance analysis is the point of §4.5, not decoration.** A confusion between
740 and 760 Hz says the model has learned nodal geometry and is losing on a fine distinction;
a confusion between 740 and 3200 Hz says it has learned nothing of the sort. Those are
different results and the TDR has to be able to tell them apart.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # CLAUDE.md rule 2 — before pyplot, never after.

import csv

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

import runs as runs_module
from dataset import load_split, to_model_input
from model import build_model
from preprocess import load_config
from train import predict, select_device

#: §4.6: three correct and three incorrect per class. Fewer is not a smaller sample, it is
#: a different claim — "we looked at what it got right" instead of "we looked at both".
GRADCAM_PER_CLASS = 3

Z_95 = 1.959963984540054


class EvaluationError(RuntimeError):
    """The run cannot be evaluated as it stands."""


def evaluate_run(
    run_dir: Path | str, data_root: Path | str, split_name: str = "test"
) -> dict[str, Any]:
    """Load a run's checkpoint, score it on `split_name`, write every §4 output."""
    run_dir = Path(run_dir)
    data_root = Path(data_root)
    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    config = load_config()
    # The same gate `tasks.py check-model` runs. Evaluating a checkpoint against images
    # built by a different preprocessing produces a number that describes nothing.
    runs_module.check_preprocess_hash(run_dir, config)

    checkpoint = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    classes = checkpoint["classes"]
    frequencies = [int(c["freq_hz"]) for c in classes]

    device = select_device()
    model = build_model(checkpoint["model"], checkpoint["n_classes"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    split = load_split(data_root, split_name, classes)
    probabilities, predictions = predict(model, split, config, device)
    truth = split.labels

    summary = {
        "split": split_name,
        "sessions": sorted(set(split.sessions)),
        "images": split.size,
        "accuracy": _accuracy_block(truth, predictions, split.sessions),
        "per_class": _per_class(truth, predictions, frequencies, report_dir),
        "confusion": _confusion(truth, predictions, frequencies, report_dir),
        "frequency_distance": _frequency_distance(truth, predictions, frequencies, report_dir),
        "learning_curves": _learning_curves(run_dir, report_dir),
        "gradcam": _gradcam(model, split, probabilities, predictions, config, device,
                            frequencies, report_dir),
    }

    (report_dir / "evaluation.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _merge_into_metrics(run_dir, summary)
    return summary


# --------------------------------------------------------------------------- #
# §4.1 — accuracy and its interval
# --------------------------------------------------------------------------- #

def wilson_interval(successes: int, total: int, z: float = Z_95) -> tuple[float, float]:
    """95 % Wilson score interval. Correct at the small `n` this project actually has.

    §5 of `docs/02`: 4 correct out of 5 is 38–96 %. The normal approximation gives 45–115 %,
    which is not a probability, and that is the whole reason this is Wilson.
    """
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def _accuracy_block(truth, predictions, sessions) -> dict[str, Any]:
    correct = int((predictions == truth).sum())
    total = int(truth.numel())
    low, high = wilson_interval(correct, total)

    per_session = {}
    for session in sorted(set(sessions)):
        mask = np.array([s == session for s in sessions])
        hits = int((predictions.numpy()[mask] == truth.numpy()[mask]).sum())
        per_session[session] = {"images": int(mask.sum()), "accuracy": hits / int(mask.sum())}

    return {
        "correct": correct,
        "total": total,
        "frame_level": round(correct / total, 6),
        "frame_level_wilson_95": [round(low, 6), round(high, 6)],
        "frame_level_note": (
            "Optimistic. A burst with the rig untouched is one observation, not 60 "
            "(docs/02 §5), so this interval is narrower than the evidence warrants."
        ),
        "per_session": per_session,
        "independent_units": len(per_session),
        "session_level_note": (
            f"§4.1: the independent unit is the session. This split has "
            f"{len(per_session)} of them, so no session-level interval is quoted — "
            f"leave-one-session-out over all sessions is what produces one."
        ),
    }


# --------------------------------------------------------------------------- #
# §4.2 — per class
# --------------------------------------------------------------------------- #

def _per_class(truth, predictions, frequencies, report_dir: Path) -> dict[str, Any]:
    labels = list(range(len(frequencies)))
    names = [f"{f} Hz" for f in frequencies]
    report = classification_report(
        truth.numpy(), predictions.numpy(), labels=labels, target_names=names,
        output_dict=True, zero_division=0,
    )

    path = report_dir / "classification_report.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["class", "freq_hz", "precision", "recall", "f1", "support"])
        for index, name in enumerate(names):
            row = report[name]
            writer.writerow([
                index, frequencies[index],
                round(row["precision"], 6), round(row["recall"], 6),
                round(row["f1-score"], 6), int(row["support"]),
            ])

    recalls = {frequencies[i]: report[names[i]]["recall"] for i in labels}
    worst = min(recalls, key=recalls.get)
    return {
        "csv": path.name,
        "macro_f1": round(report["macro avg"]["f1-score"], 6),
        "worst_class_hz": worst,
        "worst_class_recall": round(recalls[worst], 6),
        "recall_by_class": {str(k): round(v, 6) for k, v in recalls.items()},
    }


# --------------------------------------------------------------------------- #
# §4.3 — confusion matrix, absolute and row-normalised, PNG and CSV
# --------------------------------------------------------------------------- #

def _confusion(truth, predictions, frequencies, report_dir: Path) -> dict[str, Any]:
    labels = list(range(len(frequencies)))
    absolute = confusion_matrix(truth.numpy(), predictions.numpy(), labels=labels)
    row_sums = absolute.sum(axis=1, keepdims=True)
    normalised = np.divide(absolute, row_sums, out=np.zeros_like(absolute, float), where=row_sums > 0)

    _matrix_csv(report_dir / "confusion_matrix.csv", absolute, frequencies, "{:d}")
    _matrix_csv(report_dir / "confusion_matrix_normalised.csv", normalised, frequencies, "{:.6f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, data, title, fmt in (
        (axes[0], absolute, "Confusion matrix (counts)", "{:d}"),
        (axes[1], normalised, "Row-normalised (recall per class)", "{:.2f}"),
    ):
        image = ax.imshow(data, cmap="Blues", vmin=0, vmax=data.max() if data.max() else 1)
        ax.set_title(title)
        ax.set_xlabel("predicted (Hz)")
        ax.set_ylabel("true (Hz)")
        ax.set_xticks(labels, frequencies, rotation=45, ha="right")
        ax.set_yticks(labels, frequencies)
        threshold = (data.max() if data.max() else 1) * 0.6
        for i in labels:
            for j in labels:
                ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if data[i, j] > threshold else "black")
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(report_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    return {
        "png": "confusion_matrix.png",
        "csv": "confusion_matrix.csv",
        "normalised_csv": "confusion_matrix_normalised.csv",
    }


def _matrix_csv(path: Path, data, frequencies, fmt: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["true\\predicted", *frequencies])
        for i, freq in enumerate(frequencies):
            writer.writerow([freq, *(fmt.format(v) for v in data[i])])


# --------------------------------------------------------------------------- #
# §4.4 — learning curves
# --------------------------------------------------------------------------- #

def _learning_curves(run_dir: Path, report_dir: Path) -> dict[str, Any]:
    path = run_dir / "history.csv"
    if not path.exists():
        raise EvaluationError(f"{path} not found — §4.4 needs the training history")
    with path.open(newline="", encoding="utf-8") as fh:
        history = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(fh)]

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in ((axes[0], "loss", "Loss"), (axes[1], "acc", "Accuracy")):
        ax.plot(epochs, [h[f"train_{key}"] for h in history], label="train")
        ax.plot(epochs, [h[f"val_{key}"] for h in history], label="val")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(report_dir / "learning_curves.png", dpi=150)
    plt.close(fig)
    return {"png": "learning_curves.png", "epochs": len(history)}


# --------------------------------------------------------------------------- #
# §4.5 — frequency-distance error analysis
# --------------------------------------------------------------------------- #

def _frequency_distance(truth, predictions, frequencies, report_dir: Path) -> dict[str, Any]:
    errors = [
        (frequencies[int(t)], frequencies[int(p)], abs(frequencies[int(t)] - frequencies[int(p)]))
        for t, p in zip(truth.tolist(), predictions.tolist())
        if int(t) != int(p)
    ]

    path = report_dir / "frequency_distance.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["true_hz", "predicted_hz", "abs_distance_hz", "count"])
        for (true_hz, pred_hz, distance), count in sorted(Counter(errors).items()):
            writer.writerow([true_hz, pred_hz, distance, count])

    distances = [d for _t, _p, d in errors]
    fig, ax = plt.subplots(figsize=(7, 4))
    if distances:
        ax.hist(distances, bins=min(20, max(3, len(set(distances)))), color="#3b6ea5")
    else:
        ax.text(0.5, 0.5, "no misclassifications", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_title("|f_true − f_pred| over misclassified frames  (§4.5)")
    ax.set_xlabel("Hz")
    ax.set_ylabel("frames")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(report_dir / "frequency_distance.png", dpi=150)
    plt.close(fig)

    neighbour_gap = min((b - a for a, b in pairwise(frequencies)), default=0)
    return {
        "csv": path.name,
        "png": "frequency_distance.png",
        "errors": len(errors),
        "median_distance_hz": float(np.median(distances)) if distances else None,
        "max_distance_hz": max(distances) if distances else None,
        "adjacent_class_share": (
            round(sum(1 for d in distances if d <= neighbour_gap) / len(distances), 6)
            if distances else None
        ),
        "note": (
            "The share of errors landing on an adjacent mode is what §4.5 asks about: "
            "confusions clustered among neighbours mean the model reads nodal geometry "
            "and loses on a fine distinction; confusions spread across the range mean it "
            "does not."
        ),
    }


# --------------------------------------------------------------------------- #
# §4.6 — Grad-CAM. Non-negotiable.
# --------------------------------------------------------------------------- #

def _gradcam(model, split, probabilities, predictions, config, device, frequencies, report_dir):
    """Three correct and three incorrect per class, overlaid on the model's own input.

    Without this there is no evidence the model looked at nodal lines rather than at a
    scratch on the plate or the rig's shadow, which is §4.6's entire argument.
    """
    out_dir = report_dir / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)
    truth = split.labels
    written: dict[str, dict[str, int]] = {}

    for class_index, freq in enumerate(frequencies):
        of_class = [i for i in range(split.size) if int(truth[i]) == class_index]
        correct = [i for i in of_class if int(predictions[i]) == class_index]
        incorrect = [i for i in of_class if int(predictions[i]) != class_index]
        # Most confident first for the correct ones, most confident *wrong* first for the
        # others: a confidently wrong frame is the one worth looking at.
        correct.sort(key=lambda i: -float(probabilities[i, class_index]))
        incorrect.sort(key=lambda i: -float(probabilities[i].max()))

        counts = {"correct": 0, "incorrect": 0}
        for kind, indices in (("correct", correct), ("incorrect", incorrect)):
            for rank, i in enumerate(indices[:GRADCAM_PER_CLASS]):
                _write_cam(
                    model, split, i, int(predictions[i]), config, device, frequencies,
                    out_dir / f"{freq}Hz_{kind}_{rank}_{split.image_ids[i]}.png",
                )
                counts[kind] += 1
        written[str(freq)] = counts

    shortfall = {
        freq: counts for freq, counts in written.items()
        if counts["correct"] < GRADCAM_PER_CLASS or counts["incorrect"] < GRADCAM_PER_CLASS
    }
    return {
        "directory": "gradcam",
        "per_class_target": GRADCAM_PER_CLASS,
        "written": written,
        "shortfall": shortfall,
        "shortfall_note": (
            "A class with fewer than three of a kind simply does not have three in this "
            "split — three incorrect samples cannot be produced for a class the model got "
            "entirely right. That is the split being small, not the evaluation skipping §4.6."
        ),
    }


def _write_cam(model, split, index, class_index, config, device, frequencies, path: Path) -> None:
    x = to_model_input(split.images[index: index + 1], config, device)
    x.requires_grad_(False)

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
        raise EvaluationError("Grad-CAM hooks produced nothing — §4.6 cannot be satisfied")

    weights = gradients[0].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activations[0]).sum(dim=1, keepdim=True))
    cam = torch.nn.functional.interpolate(
        cam, size=x.shape[-2:], mode="bilinear", align_corners=False
    )[0, 0].detach().cpu().numpy()
    if cam.max() > 0:
        cam = cam / cam.max()

    edges = split.images[index].numpy()
    true_hz = frequencies[int(split.labels[index])]
    predicted_hz = frequencies[class_index]

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.4))
    axes[0].imshow(edges, cmap="gray")
    axes[0].set_title("model input (step 7)", fontsize=8)
    axes[1].imshow(edges, cmap="gray")
    axes[1].imshow(cam, cmap="jet", alpha=0.45)
    axes[1].set_title("Grad-CAM", fontsize=8)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"{split.image_ids[index]} — true {true_hz} Hz, predicted {predicted_hz} Hz",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #

def _merge_into_metrics(run_dir: Path, summary: dict[str, Any]) -> None:
    """`metrics.json` is committed (`CLAUDE.md` rule 4); the headline numbers belong in it."""
    path = run_dir / "metrics.json"
    metrics = json.loads(path.read_text("utf-8")) if path.exists() else {}
    metrics["evaluation"] = {
        "split": summary["split"],
        "sessions": summary["sessions"],
        "images": summary["images"],
        "accuracy": summary["accuracy"]["frame_level"],
        "accuracy_wilson_95": summary["accuracy"]["frame_level_wilson_95"],
        "macro_f1": summary["per_class"]["macro_f1"],
        "worst_class_hz": summary["per_class"]["worst_class_hz"],
        "worst_class_recall": summary["per_class"]["worst_class_recall"],
        "errors": summary["frequency_distance"]["errors"],
        "adjacent_class_share": summary["frequency_distance"]["adjacent_class_share"],
    }
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
