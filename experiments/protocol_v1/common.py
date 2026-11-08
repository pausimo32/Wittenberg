"""Data, groups, verified splits and group-weighted metrics for the protocol campaign.

Images come from data/processed/ — the output of src/preprocess.py (step 7). Step 8 is
`preprocess._scale`, imported, never re-implemented (CLAUDE.md rule 1).
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CLASSES_HZ = [497, 863, 1218, 2241]
SPLIT_SEED = 42


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


# --- manifest + groups ----------------------------------------------------------------------

def load_manifest() -> list[dict]:
    """One row per accepted original, with its preparation block (src/groups.py) as `group_id`."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from groups import frame_number, preparation_groups
    rows = [r for r in csv.DictReader(open(ROOT / "data" / "metadata.csv", encoding="utf-8"))
            if r["quality"] == "ok" and int(r["freq_hz"]) in CLASSES_HZ]
    gmap = preparation_groups(rows, ROOT / "data")
    rows.sort(key=lambda r: (r["session_id"], int(r["freq_hz"]), frame_number(r["image_id"])))
    out, rank = [], {}
    for r in rows:
        key = (r["session_id"], int(r["freq_hz"]))
        rank[key] = rank.get(key, -1) + 1
        n = r["image_id"].rsplit("_", 1)[-1]
        hz = int(r["freq_hz"])
        out.append({
            "image_id": r["image_id"], "session_id": r["session_id"], "freq_hz": hz,
            "y": CLASSES_HZ.index(hz), "rank": rank[key], "group_id": gmap[r["image_id"]],
            "path": ROOT / "data" / "processed" / "unassigned" / str(hz) / f"{r['session_id']}_{n}.png",
        })
    return out


def load_images(manifest: list[dict]) -> np.ndarray:
    """(N,128,128) uint8, step 7 exactly as on disk."""
    imgs = []
    for r in manifest:
        a = cv2.imread(str(r["path"]), cv2.IMREAD_GRAYSCALE)
        if a is None or a.shape != (128, 128):
            raise RuntimeError(f"bad image {r['path']}")
        imgs.append(a)
    return np.stack(imgs)


# --- splits (protocol p.8) ------------------------------------------------------------------

def round_robin_folds(y, groups, n_splits):
    """Fixed, label/group-only assignment: within each session, blocks in capture order go
    round-robin to folds (block b -> fold b mod K). With 3 blocks of 2241 Hz this puts exactly
    one in each outer fold. Amendment A1 (amendments.md): StratifiedGroupKFold(seed=42) left a
    test fold without 2241 Hz, and the protocol forbids trying seeds."""
    groups = np.asarray(groups)
    fold_of = {}
    for session in sorted({g.rsplit("_b", 1)[0] for g in groups}):
        blocks = sorted({g for g in groups if g.startswith(session)})
        for b, g in enumerate(blocks):
            fold_of[g] = b % n_splits
    folds = []
    for k in range(n_splits):
        te = np.flatnonzero([fold_of[g] == k for g in groups])
        tr = np.flatnonzero([fold_of[g] != k for g in groups])
        folds.append((tr, te))
    return folds


def checked_splits(y, groups, n_splits):
    y, groups = np.asarray(y), np.asarray(groups)
    labels = set(np.unique(y))
    folds = round_robin_folds(y, groups, n_splits)
    seen = np.zeros(len(y), dtype=int)
    for tr, te in folds:
        assert not set(groups[tr]) & set(groups[te]), "group on both sides"
        assert set(y[tr]) == labels, "class missing in train"
        assert set(y[te]) == labels, "class missing in test"
        assert not set(tr) & set(te)
        seen[te] += 1
    assert np.all(seen == 1), "an original is not tested exactly once"
    return folds


def make_splits(y, groups) -> dict:
    """Outer 3-fold, inner 2-fold strictly inside each outer train; global indices."""
    y, groups = np.asarray(y), np.asarray(groups)
    outer = []
    for k, (tr, te) in enumerate(checked_splits(y, groups, 3)):
        inner = []
        for itr, iva in checked_splits(y[tr], groups[tr], 2):
            gtr, gva = tr[itr], tr[iva]  # local -> global
            assert set(gtr) | set(gva) == set(tr) and not set(gva) & set(te)
            inner.append({"train": gtr.tolist(), "val": gva.tolist()})
        outer.append({"fold": k, "train": tr.tolist(), "test": te.tolist(), "inner": inner})
    all_inner = [{"train": a.tolist(), "val": b.tolist()} for a, b in checked_splits(y, groups, 2)]
    return {"method": "round_robin_blocks_in_capture_order", "outer": outer, "final_inner": all_inner}


# --- metrics (protocol p.21) ----------------------------------------------------------------

def group_weights(groups) -> np.ndarray:
    groups = np.asarray(groups)
    _, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return 1.0 / counts[inv]


def metrics(y_true, y_pred, groups=None) -> dict:
    """Macro-F1 etc. from a (optionally group-weighted) confusion matrix. rows=true, cols=pred."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    w = group_weights(groups) if groups is not None else np.ones(len(y_true))
    C = len(CLASSES_HZ)
    M = confusion_matrix(y_true, y_pred, labels=list(range(C)), sample_weight=w)
    tp = np.diag(M); fn = M.sum(1) - tp; fp = M.sum(0) - tp
    support = M.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(support > 0, tp / support, np.nan)
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
    return {
        "macro_f1": float(np.mean(f1)),
        "balanced_accuracy": float(np.nanmean(rec)),
        "accuracy": float(tp.sum() / M.sum()),
        "per_class": {str(hz): {"precision": float(prec[i]), "recall": None if np.isnan(rec[i]) else float(rec[i]),
                                "f1": float(f1[i]), "support_effective": float(support[i])}
                      for i, hz in enumerate(CLASSES_HZ)},
        "confusion": M.tolist(),
    }


def bootstrap_groups(y, pred_by_model: dict, groups, reps=2000, seed=71) -> dict:
    """Resample whole groups; same replicates for every model (paired). Descriptive only."""
    y, groups = np.asarray(y), np.asarray(groups)
    ug = np.unique(groups)
    idx_by_g = {g: np.flatnonzero(groups == g) for g in ug}
    rng = np.random.default_rng(seed)
    res = {m: [] for m in pred_by_model}
    valid = 0
    for _ in range(reps):
        pick = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in pick])
        gid = np.concatenate([np.full(len(idx_by_g[g]), j) for j, g in enumerate(pick)])  # each appearance its own group
        if len(np.unique(y[idx])) < len(CLASSES_HZ):
            continue
        valid += 1
        for m, p in pred_by_model.items():
            res[m].append(metrics(y[idx], np.asarray(p)[idx], gid)["macro_f1"])
    out = {"replicates": reps, "valid": valid}
    for m, v in res.items():
        v = np.array(v)
        out[m] = {"median": float(np.median(v)), "p2.5": float(np.percentile(v, 2.5)),
                  "p97.5": float(np.percentile(v, 97.5))} if len(v) else None
    return out
