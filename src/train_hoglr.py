"""`python tasks.py train-hoglr` — fit the ADR 0009 model on every accepted image, write a run.

The protocol's final step (p. 27): once the selection rule has chosen a candidate, refit it on
all development data. There is no held-out split here *by design* — the out-of-fold estimate
lives in `runs/protocol_v1/metrics.json` and is copied into this run's `metrics.json` with its
provenance, so the app shows a number that was measured, not the training-set fit.

The run passes the same startup gate as a CNN run: `config.json` carries the preprocessing
hash and the OpenCV signature (`src/runs.py`), and `model: "hog_lr"` tells the API which
loader to use.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import hoglr
import runs as runs_module
from environment import opencv_signature
from groups import preparation_groups
from metadata import is_accepted, read_metadata
from model import classes_from_config
from preprocess import load_config

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_METRICS = ROOT / "runs" / "protocol_v1" / "metrics.json"


class TrainHogLRError(RuntimeError):
    pass


def load_all_accepted(data_root: Path, classes: list[dict[str, Any]]):
    rows = [r for r in read_metadata(data_root / "metadata.csv") if is_accepted(r)]
    known = {int(c["freq_hz"]): c for c in classes}
    stray = sorted({int(r["freq_hz"]) for r in rows} - set(known))
    if stray:
        raise TrainHogLRError(f"frequencies {stray} are not in config/classes.json")
    present = sorted({int(r["freq_hz"]) for r in rows})
    used = [{"index": int(known[f]["index"]), "freq_hz": f} for f in present]
    position = {c["freq_hz"]: i for i, c in enumerate(used)}
    rows.sort(key=lambda r: r["image_id"])
    images, labels = [], []
    for r in rows:
        name = f"{r['session_id']}_{r['image_id'].rsplit('_', 1)[-1]}.png"
        path = data_root / "processed" / (r.get("split") or "unassigned") / str(r["freq_hz"]) / name
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise TrainHogLRError(f"{path} is missing or unreadable — run `python tasks.py ingest --all`")
        images.append(img)
        labels.append(position[int(r["freq_hz"])])
    return rows, np.stack(images), np.array(labels), used


def train_run(data_root: Path | str, runs_root: Path | str | None = None,
              C: float = hoglr.C_DEFAULT) -> Path:
    data_root = Path(data_root)
    t0 = time.perf_counter()
    rows, images, labels, used = load_all_accepted(data_root, classes_from_config())
    if len(used) < 2:
        raise TrainHogLRError("need at least two classes with accepted images")
    gmap = preparation_groups(rows, data_root)
    groups = np.array([gmap[r["image_id"]] for r in rows])
    model = hoglr.fit(images, labels, groups, used, C=C)
    train_acc = float((model.predict_proba(hoglr.hog_features(images)).argmax(1) == labels).mean())

    config = load_config()
    run_dir = runs_module.new_run_dir(runs_root)
    model.save(run_dir)
    (run_dir / "config.json").write_text(json.dumps({
        "model": hoglr.MODEL_NAME,
        "n_classes": len(used),
        "classes": used,
        "hyperparameters": {"C": C, "hog": hoglr.HOG_PARAMS, "sample_weight": "1/n_group, mean 1",
                            "group_unit": "preparation_block (ADR 0009)"},
        "preprocess_version": config.get("version"),
        "preprocess_sha256": runs_module.preprocess_hash(config),
        "preprocess_config": config,
        **opencv_signature(),
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "numpy": np.__version__, **model.meta},
        "seed": 42,
    }, indent=2) + "\n", encoding="utf-8")

    estimate = None
    if PROTOCOL_METRICS.exists():
        pm = json.loads(PROTOCOL_METRICS.read_text("utf-8"))
        sel = pm["results"]["selector"]["weighted"]
        estimate = {
            "source": "runs/protocol_v1/metrics.json",
            "prereg_sha256": pm.get("prereg_sha256"),
            "selected": pm.get("final", {}).get("chosen"),
            "macro_f1_group_weighted_oof": sel["macro_f1"],
            "per_class_recall": {k: v["recall"] for k, v in sel["per_class"].items()},
            "bootstrap_95": pm.get("bootstrap_groups", {}).get("selector"),
            "status": "provisional — not confirmed on a later session (ADR 0009)",
            "caveat": "2241 Hz is alone in its session; part of its score may be the session, not the pattern",
        }
    (run_dir / "metrics.json").write_text(json.dumps({
        "run": run_dir.name, "model": hoglr.MODEL_NAME, "seed": 42, "synthetic": False,
        "do_not_report": None,
        "n_images": int(len(labels)), "n_groups": int(len(set(groups))),
        "sessions": sorted({r["session_id"] for r in rows}),
        "training_set_accuracy": train_acc,
        "training_set_accuracy_note": "fit on the same images — NOT an estimate; see out_of_fold_estimate",
        "out_of_fold_estimate": estimate,
        "train_seconds": round(time.perf_counter() - t0, 2),
    }, indent=2) + "\n", encoding="utf-8")
    runs_module.write_latest(run_dir, runs_root)
    return run_dir
