"""Nested grouped CV campaign for the 20261108 captures (protocol v1.0, pages 6-27).

    python experiments/protocol_v1/run_protocol.py prep     # audit, groups, splits, latency, caches
    python experiments/protocol_v1/run_protocol.py run      # nested CV, controls, metrics, final model

Everything is written to runs/protocol_v1/. Nothing is shown on screen (CLAUDE.md rule 2).
"""
from __future__ import annotations

import csv
import json
import math
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import models as M  # noqa: E402
from common import (CLASSES_HZ, HERE, ROOT, bootstrap_groups, load_images, load_manifest,  # noqa: E402
                    make_splits, metrics, sha256_file, sha256_obj)

OUT = ROOT / "runs" / "protocol_v1"
PREREG = HERE / "preregister.yaml"
GPU = "cuda" if torch.cuda.is_available() else "cpu"
# batch-4 C1 is faster on CPU (launch overhead dominates); W1 at batch 32 is ~14x faster on the GPU
DEVICE_FOR = {"C1": "cpu", "W1": GPU}
DEVICE = DEVICE_FOR
SEEDS = [42, 17, 29]
FAMILY_ORDER = ["B0", "B1", "C1", "W1", "F1", "F2"]


def determinism():
    import random
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def candidates() -> list[dict]:
    reg = yaml.safe_load(PREREG.read_text(encoding="utf-8"))["candidates"]
    return [{"id": k, **v, "aug": v.get("aug", "A0"), "eligible": v.get("eligible", True)} for k, v in reg.items()]


def write_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


# ============================================================================== prep

def prep():
    OUT.mkdir(parents=True, exist_ok=True)
    man = load_manifest()
    y = np.array([r["y"] for r in man]); groups = np.array([r["group_id"] for r in man])
    imgs = load_images(man)

    # manifest_grouped.csv
    with open(OUT / "manifest_grouped.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["image_id", "session_id", "freq_hz", "class_id", "rank", "group_id"])
        for r in man:
            w.writerow([r["image_id"], r["session_id"], r["freq_hz"], r["y"], r["rank"], r["group_id"]])

    # audit (protocol p.5): counts, class x group, exact + near duplicates
    hashes = Counter(sha256_file(r["path"]) for r in man)
    cg = defaultdict(Counter)
    for r in man:
        cg[r["group_id"]][r["freq_hz"]] += 1
    small = np.stack([cv2.resize(cv2.GaussianBlur(a, (0, 0), 2), (32, 32), interpolation=cv2.INTER_AREA)
                      for a in imgs]).reshape(len(man), -1).astype(np.float64)
    small -= small.mean(1, keepdims=True); small /= np.linalg.norm(small, axis=1, keepdims=True)
    S = small @ small.T
    sim = {}
    for c, hz in enumerate(CLASSES_HZ):
        idx = np.flatnonzero(y == c)
        same, diff, consec = [], [], []
        for a in idx:
            for b in idx:
                if a < b:
                    (same if groups[a] == groups[b] else diff).append(S[a, b])
                    if abs(man[a]["rank"] - man[b]["rank"]) == 1:
                        consec.append(S[a, b])
        sim[str(hz)] = {"within_block_mean": float(np.mean(same)), "between_block_mean": float(np.mean(diff)),
                        "consecutive_mean": float(np.mean(consec))}
    cross = float(np.mean([S[a, b] for a in range(len(man)) for b in range(a + 1, len(man)) if y[a] != y[b]]))
    G_c = {str(hz): int(len(set(groups[y == c]))) for c, hz in enumerate(CLASSES_HZ)}
    audit = {
        "n_originals": len(man), "counts_by_class": {str(hz): int((y == c).sum()) for c, hz in enumerate(CLASSES_HZ)},
        "groups_per_class": G_c, "G_min": min(G_c.values()), "n_groups": int(len(set(groups))),
        "class_by_session": {s: dict(Counter(str(r["freq_hz"]) for r in man if r["session_id"] == s))
                             for s in sorted({r["session_id"] for r in man})},
        "class_by_group": {g: dict(v) for g, v in sorted(cg.items())},
        "exact_duplicates": int(sum(v - 1 for v in hashes.values() if v > 1)),
        "edge_map_similarity_cosine": sim, "cross_class_similarity_mean": cross,
        "confounds": ["2241 Hz == session 20261108-04 (no other class in that session)"],
    }
    write_json("audit.json", audit)

    splits = make_splits(y, groups)
    table = []
    for o in splits["outer"]:
        for role in ("train", "test"):
            ix = np.array(o[role])
            table.append({"fold": o["fold"], "role": role, "groups": sorted(set(groups[ix])),
                          "per_class": {str(hz): int((y[ix] == c).sum()) for c, hz in enumerate(CLASSES_HZ)}})
    splits["table"] = table
    splits["sha256_manifest"] = sha256_file(OUT / "manifest_grouped.csv")
    write_json("splits.json", splits)

    write_json("latency.json", M.measure_latency())

    unit = M.to_unit(imgs)
    np.save(OUT / "cache_hog.npy", M.hog_features(imgs))
    ring = imgs.copy(); ring[:, 8:-8, 8:-8] = 0
    np.save(OUT / "cache_hog_background.npy", M.hog_features(ring))
    wid = {}
    for fam, name in (("F1", "mobilenet_v3_small"), ("F2", "efficientnet_b0")):
        E, wid[fam] = M.embeddings(name, unit)
        np.save(OUT / f"cache_emb_{fam}.npy", E)
    write_json("prep.json", {"prereg_sha256": sha256_file(PREREG), "splits_sha256": sha256_file(OUT / "splits.json"),
                             "weights": wid, "torch": torch.__version__, "device_for_run": DEVICE})
    print(json.dumps({k: audit[k] for k in ("n_originals", "groups_per_class", "G_min", "exact_duplicates")}))
    print("similarity", json.dumps(sim), "cross", round(cross, 3))
    print("splits", [(t["fold"], t["role"], t["per_class"]) for t in table])


# ============================================================================== run

class Data:
    def __init__(self):
        self.man = load_manifest()
        self.y = np.array([r["y"] for r in self.man])
        self.groups = np.array([r["group_id"] for r in self.man])
        self.ids = [r["image_id"] for r in self.man]
        imgs = load_images(self.man)
        X = torch.from_numpy(M.to_unit(imgs)).unsqueeze(1)
        self.X = {d: X.to(d) for d in set(DEVICE_FOR.values())}
        self.Y = {d: torch.from_numpy(self.y).to(d) for d in set(DEVICE_FOR.values())}
        self.feat = {"B0": np.load(OUT / "cache_hog.npy"), "B1": np.load(OUT / "cache_hog.npy"),
                     "F1": np.load(OUT / "cache_emb_F1.npy"), "F2": np.load(OUT / "cache_emb_F2.npy"),
                     "BG": np.load(OUT / "cache_hog_background.npy")}


def xy(D, cfg):
    d = DEVICE_FOR[cfg["family"]]
    return D.X[d], D.Y[d]


def is_neural(cfg):
    return cfg["family"] in ("C1", "W1")


def inner_eval(D: Data, cfg, inner_splits, log):
    """Score one candidate on the inner splits of an outer train. Returns inner macro-F1
    (group-weighted, over the pooled inner-val predictions) and per-seed best epochs."""
    pred = {}
    best_epochs = defaultdict(list)
    for k, s in enumerate(inner_splits):
        tr, va = np.array(s["train"]), np.array(s["val"])
        if is_neural(cfg):
            probs = []
            for seed in SEEDS:
                t0 = time.perf_counter()
                r = M.train_net(cfg, *xy(D, cfg), D.groups, tr, va=va, seed=seed, device=DEVICE_FOR[cfg['family']])
                probs.append(r["val_probs"]); best_epochs[seed].append(r["best_epoch"])
                log({"kind": "inner_fit", "candidate": cfg["id"], "inner_fold": k, "seed": seed,
                     "best_epoch": r["best_epoch"], "val_loss": r["val_loss"], "fit_s": time.perf_counter() - t0,
                     "n_train": len(tr), "n_val": len(va)})
            p = np.mean(probs, 0).argmax(1)
        else:
            p = M.fit_linear(cfg, D.feat[cfg["family"]], D.y, D.groups, tr, va)["pred"]
        for i, j in zip(va, p):
            pred[int(i)] = int(j)
    idx = np.array(sorted(pred))
    score = metrics(D.y[idx], [pred[i] for i in idx], D.groups[idx])["macro_f1"]
    epochs = {seed: int(math.ceil(np.median(v))) for seed, v in best_epochs.items()}
    return score, epochs


def refit_predict(D: Data, cfg, tr, te, epochs):
    if is_neural(cfg):
        probs = []
        for seed in SEEDS:
            t = M.train_net(cfg, *xy(D, cfg), D.groups, tr, epochs=epochs[seed], seed=seed, device=DEVICE_FOR[cfg['family']])
            probs.append(M.predict_net(t, xy(D, cfg)[0], te))
        s = np.mean(probs, 0)
        return {"pred": s.argmax(1), "scores": s, "score_type": "prob_mean_3seeds"}
    return M.fit_linear(cfg, D.feat[cfg["family"]], D.y, D.groups, tr, te)


def select(scores: dict, cands: list[dict], latency: dict, delta: float):
    """Pre-registered rule: tie set within delta of best -> lowest latency -> smaller artifact -> earlier entry."""
    elig = [c for c in cands if c["eligible"]]
    best = max(scores[c["id"]] for c in elig)
    tie = [c for c in elig if scores[c["id"]] >= best - delta]
    order = {c["id"]: i for i, c in enumerate(cands)}
    tie.sort(key=lambda c: (latency[c["family"]]["median_ms"], latency[c["family"]]["artifact_bytes"], order[c["id"]]))
    return tie[0], best, [c["id"] for c in tie]


def run():
    determinism()
    D = Data()
    cands = candidates()
    prereg = yaml.safe_load(PREREG.read_text(encoding="utf-8"))
    delta = prereg["evaluation"]["practical_delta"]
    splits = json.loads((OUT / "splits.json").read_text())
    latency = json.loads((OUT / "latency.json").read_text())
    logf = open(OUT / "selection_log.jsonl", "w", encoding="utf-8")

    def log(rec):
        logf.write(json.dumps(rec) + "\n"); logf.flush()

    N = len(D.y)
    oof_sel = np.full(N, -1); oof_fam = {f: np.full(N, -1) for f in FAMILY_ORDER}
    sel_cfg = [None] * N
    refit_cache = {}
    inner_table = {}
    for o in splits["outer"]:
        k = o["fold"]; tr, te = np.array(o["train"]), np.array(o["test"])
        scores, epochs = {}, {}
        for cfg in cands:
            t0 = time.perf_counter()
            scores[cfg["id"]], epochs[cfg["id"]] = inner_eval(D, cfg, o["inner"], log)
            log({"kind": "inner_score", "outer_fold": k, "candidate": cfg["id"], "inner_macro_f1": scores[cfg["id"]],
                 "refit_epochs": epochs[cfg["id"]], "s": time.perf_counter() - t0})
            print(f"fold {k} {cfg['id']:<20} inner F1 {scores[cfg['id']]:.3f} epochs {epochs[cfg['id']]}", flush=True)
        inner_table[k] = scores

        def get(cfg):
            key = (cfg["id"], k)
            if key not in refit_cache:
                refit_cache[key] = refit_predict(D, cfg, tr, te, epochs[cfg["id"]])
            return refit_cache[key]

        chosen, best, tie = select(scores, cands, latency, delta)
        log({"kind": "selection", "outer_fold": k, "chosen": chosen["id"], "best_inner": best, "tie_set": tie})
        r = get(chosen)
        oof_sel[te] = r["pred"]
        for i in te:
            sel_cfg[i] = chosen["id"]
        for fam in FAMILY_ORDER:  # per-family OOF: best config of the family by inner score
            fc = [c for c in cands if c["family"] == fam]
            fbest = max(fc, key=lambda c: (scores[c["id"]], -cands.index(c)))
            oof_fam[fam][te] = get(fbest)["pred"]
            log({"kind": "family_pick", "outer_fold": k, "family": fam, "candidate": fbest["id"]})
        print(f"fold {k}: chosen {chosen['id']} (tie set {tie})", flush=True)

    assert (oof_sel >= 0).all() and all((v >= 0).all() for v in oof_fam.values()), "incomplete OOF coverage"

    # controls, outside the ranking
    bg_cfg = {"id": "BG_hog_lr_C1", "family": "BG", "C": 1.0, "head": "logreg"}
    oof_bg = np.full(N, -1)
    for o in splits["outer"]:
        te = np.array(o["test"])
        oof_bg[te] = M.fit_linear(bg_cfg, D.feat["BG"], D.y, D.groups, np.array(o["train"]), te)["pred"]
    rs_cfg = {"id": "RS_hog_lr_C1", "family": "B1", "C": 1.0, "head": "logreg"}
    oof_rs = np.full(N, -1)
    for tr, te in StratifiedKFold(3, shuffle=True, random_state=42).split(D.y, D.y):
        oof_rs[te] = M.fit_linear(rs_cfg, D.feat["B1"], D.y, D.groups, tr, te)["pred"]

    # outputs
    with open(OUT / "oof_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "group_id", "outer_fold", "y_true_hz", "selector_hz", "selector_config"]
                   + [f"{fam}_hz" for fam in FAMILY_ORDER] + ["control_background_hz", "control_random_split_hz"])
        fold_of = {i: o["fold"] for o in splits["outer"] for i in o["test"]}
        for i in range(N):
            w.writerow([D.ids[i], D.groups[i], fold_of[i], CLASSES_HZ[D.y[i]], CLASSES_HZ[oof_sel[i]], sel_cfg[i]]
                       + [CLASSES_HZ[oof_fam[f][i]] for f in FAMILY_ORDER]
                       + [CLASSES_HZ[oof_bg[i]], CLASSES_HZ[oof_rs[i]]])

    res = {"selector": {"weighted": metrics(D.y, oof_sel, D.groups), "unweighted": metrics(D.y, oof_sel)}}
    for fam in FAMILY_ORDER:
        res[f"family_{fam}"] = {"weighted": metrics(D.y, oof_fam[fam], D.groups), "unweighted": metrics(D.y, oof_fam[fam])}
    res["control_background_only"] = {"weighted": metrics(D.y, oof_bg, D.groups), "unweighted": metrics(D.y, oof_bg)}
    res["control_random_split"] = {"unweighted": metrics(D.y, oof_rs), "note": "leakage diagnostic, not an estimate"}
    boot = bootstrap_groups(D.y, {"selector": oof_sel, **{f"family_{f}": oof_fam[f] for f in FAMILY_ORDER},
                                  "control_background_only": oof_bg},
                            D.groups, reps=prereg["evaluation"]["bootstrap_replicates"], seed=prereg["evaluation"]["bootstrap_seed"])
    per_group = {}
    for g in sorted(set(D.groups)):
        ix = np.flatnonzero(D.groups == g)
        per_group[g] = {"n": int(len(ix)), "selector_correct": int((oof_sel[ix] == D.y[ix]).sum()),
                        **{f"{f}_correct": int((oof_fam[f][ix] == D.y[ix]).sum()) for f in FAMILY_ORDER}}

    # final artefact: same selection rule on all development data, then refit on everything
    all_idx = np.arange(N)
    fscores, fepochs = {}, {}
    for cfg in cands:
        fscores[cfg["id"]], fepochs[cfg["id"]] = inner_eval(D, cfg, splits["final_inner"], log)
    fchosen, fbest, ftie = select(fscores, cands, latency, delta)
    log({"kind": "final_selection", "chosen": fchosen["id"], "best_inner": fbest, "tie_set": ftie, "scores": fscores})
    final = save_final(D, fchosen, fepochs[fchosen["id"]], all_idx)

    metrics_out = {
        "experiment_id": prereg["experiment_id"], "seed": 42, "train_seeds": SEEDS, "split_seed": 42,
        "prereg_sha256": sha256_file(PREREG), "splits_sha256": sha256_file(OUT / "splits.json"),
        "device": DEVICE, "torch": torch.__version__, "python": platform.python_version(),
        "n_originals": int(N), "n_groups": int(len(set(D.groups))), "group_unit": "preparation_block",
        "weight_formula": "w_i = 1 / n_originals_in_group",
        "results": res, "bootstrap_groups": boot, "per_group": per_group,
        "inner_scores_by_outer_fold": inner_table,
        "final": {"chosen": fchosen["id"], "inner_scores_all_dev": fscores, "tie_set": ftie, **final},
        "latency": latency,
    }
    write_json("metrics.json", metrics_out)
    write_json("config.json", {"prereg": prereg, "candidates": cands, "seeds": SEEDS, "device": DEVICE})
    plot_confusions(D, oof_sel, oof_bg)
    logf.close()
    print(json.dumps({k: {"macro_f1": v.get("weighted", v.get("unweighted"))["macro_f1"]} for k, v in res.items()}, indent=1))
    print("final:", fchosen["id"], "tie", ftie)


def save_final(D, cfg, epochs, idx):
    import pickle
    if is_neural(cfg):
        states = []
        for seed in SEEDS:
            t = M.train_net(cfg, *xy(D, cfg), D.groups, idx, epochs=epochs[seed], seed=seed, device=DEVICE_FOR[cfg['family']])
            states.append({k: v.cpu() for k, v in t["net"].state_dict().items()})
        path = OUT / "final_model.pt"
        torch.save({"config": cfg, "epochs": epochs, "seeds": SEEDS, "classes_hz": CLASSES_HZ, "states": states}, path)
    else:
        from sklearn.base import clone  # noqa: F401  (fit_linear refits from scratch)
        path = OUT / "final_model.pkl"
        X = D.feat[cfg["family"]]
        r = M.fit_linear(cfg, X, D.y, D.groups, idx, idx)
        pickle.dump({"config": cfg, "classes_hz": CLASSES_HZ, "feature": "cv2 HOG (128,128),(32,32),(16,16),(16,16),9 on preprocess v1.1.0 step 7",
                     "pipeline": r["model"], "train_pred_check": r["pred"].tolist()}, open(path, "wb"))
    return {"artifact": path.name, "artifact_sha256": sha256_file(path), "refit_epochs": epochs}


def plot_confusions(D, oof_sel, oof_bg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [f"{h} Hz" for h in CLASSES_HZ]
    views = [("confusion_counts.png", metrics(D.y, oof_sel)["confusion"], "Selector OOF — counts"),
             ("confusion_group_weighted.png", metrics(D.y, oof_sel, D.groups)["confusion"], "Selector OOF — group-weighted"),
             ("confusion_background_control.png", metrics(D.y, oof_bg)["confusion"], "Background-only control — counts")]
    for name, Mx, title in views:
        Mx = np.array(Mx)
        fig, ax = plt.subplots(figsize=(4.6, 4))
        ax.imshow(Mx, cmap="Blues")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{Mx[i, j]:.2g}", ha="center", va="center", fontsize=9)
        ax.set_xticks(range(4), labels, rotation=30); ax.set_yticks(range(4), labels)
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title, fontsize=10)
        fig.tight_layout(); fig.savefig(OUT / name, dpi=150); plt.close(fig)


if __name__ == "__main__":
    {"prep": prep, "run": run}[sys.argv[1]]()
