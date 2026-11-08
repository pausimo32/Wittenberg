"""Campaign v2: wittenberg_v1 alone, A2 (D4 + A1), batch 8. Measures its out-of-fold accuracy
on the v1 splits, its robustness, and writes the app run (seed 42, all 125 photos).

    python experiments/protocol_v2/run_v2.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "protocol_v1"))
sys.path.insert(0, str(ROOT / "src"))

import models as M  # noqa: E402
import run_protocol as R  # noqa: E402
from common import CLASSES_HZ, bootstrap_groups, metrics, sha256_file  # noqa: E402
import preprocess as pp  # noqa: E402
import runs as runs_module  # noqa: E402
from environment import opencv_signature  # noqa: E402

OUT = ROOT / "runs" / "protocol_v2"
PREREG = HERE / "preregister.yaml"
CFG = {"id": "W1_v2_A2_b8", "family": "W1", "lr": 1e-3, "aug": "A2", "batch": 8}
DEPLOY_SEED = 42
SEEDS = [42, 17, 29]
DEV = R.DEVICE_FOR["W1"]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def epochs_from_inner(X, Y, groups, inner, seed, cfg):
    best = [M.train_net(cfg, X, Y, groups, np.array(s["train"]), va=np.array(s["val"]),
                        seed=seed, device=DEV)["best_epoch"] for s in inner]
    return int(math.ceil(np.median(best))), best


def lighting_images(man, factor):
    """Raw photo x factor (RGB), through src/preprocess.py with the session calibration -> step 7."""
    out = []
    for r in man:
        n = r["image_id"].rsplit("_", 1)[-1]
        bgr = cv2.imread(str(ROOT / "data/raw" / r["session_id"] / str(r["freq_hz"]) / f"IMG_{n}.png"),
                         cv2.IMREAD_UNCHANGED)
        rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA if bgr.shape[2] == 3 else cv2.COLOR_BGRA2RGBA)
        rgba[..., :3] = np.clip(rgba[..., :3].astype(np.float32) * factor, 0, 255).astype(np.uint8)
        calib = json.loads((ROOT / "data/calib" / f"{r['session_id']}.json").read_text())
        out.append(pp.preprocess_stages(rgba, calib).resized)
    return torch.from_numpy(M.to_unit(np.stack(out))).unsqueeze(1).to(DEV)


def corners_only(X):
    c = 14
    m = torch.zeros_like(X)
    for sl in [(slice(None, c), slice(None, c)), (slice(None, c), slice(-c, None)),
               (slice(-c, None), slice(None, c)), (slice(-c, None), slice(-c, None))]:
        m[..., sl[0], sl[1]] = 1
    return X * m


def main():
    R.determinism()
    OUT.mkdir(parents=True, exist_ok=True)
    D = R.Data()
    X, Y = D.X[DEV], D.Y[DEV]
    y, groups = D.y, D.groups
    splits = json.loads((ROOT / "runs/protocol_v1/splits.json").read_text())
    N = len(y)
    light = {f: lighting_images(D.man, f) for f in (0.6, 1.4)}
    Xc = corners_only(X)

    oof = {s: np.full(N, -1) for s in SEEDS}
    oof_d4 = np.full((N, 8), -1)
    oof_light = {f: np.full(N, -1) for f in light}
    oof_corners = np.full(N, -1)
    log = []
    t0 = time.perf_counter()
    for o in splits["outer"]:
        tr, te = np.array(o["train"]), np.array(o["test"])
        for seed in SEEDS:
            ep, inner_best = epochs_from_inner(X, Y, groups, o["inner"], seed, CFG)
            net = M.train_net(CFG, X, Y, groups, tr, epochs=ep, seed=seed, device=DEV)
            oof[seed][te] = M.predict_net(net, X, te).argmax(1)
            log.append({"fold": o["fold"], "seed": seed, "inner_best_epochs": inner_best, "refit_epochs": ep})
            if seed == DEPLOY_SEED:
                for k in range(8):
                    oof_d4[te, k] = M.predict_net(net, M.d4(X, k), te).argmax(1)
                for f, Xl in light.items():
                    oof_light[f][te] = M.predict_net(net, Xl, te).argmax(1)
        epc, _ = epochs_from_inner(Xc, Y, groups, o["inner"], DEPLOY_SEED, CFG)
        netc = M.train_net(CFG, Xc, Y, groups, tr, epochs=epc, seed=DEPLOY_SEED, device=DEV)
        oof_corners[te] = M.predict_net(netc, Xc, te).argmax(1)
        print(f"fold {o['fold']} done  acc(seed42)={np.mean(oof[42][te] == y[te]):.3f}  "
              f"{time.perf_counter() - t0:.0f}s", flush=True)

    k42 = int((oof[42] == y).sum())
    res = {
        "headline_accuracy_seed42": k42 / N, "correct": k42, "n": N,
        "wilson95_frame_level": wilson(k42, N),
        "seed42": {"weighted": metrics(y, oof[42], groups), "unweighted": metrics(y, oof[42])},
        "accuracy_other_seeds": {s: float(np.mean(oof[s] == y)) for s in SEEDS},
        "robustness": {
            "d4_test_accuracy_mean": float(np.mean(oof_d4 == y[:, None])),
            "d4_test_accuracy_per_symmetry": [float(np.mean(oof_d4[:, k] == y)) for k in range(8)],
            "lighting_x0.6_accuracy": float(np.mean(oof_light[0.6] == y)),
            "lighting_x1.4_accuracy": float(np.mean(oof_light[1.4] == y)),
        },
        "control_corners_only": metrics(y, oof_corners, groups),
        "bootstrap_groups": bootstrap_groups(y, {"seed42": oof[42], "corners": oof_corners}, groups),
    }
    with open(OUT / "oof_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "group_id", "y_true_hz"] + [f"seed{s}_hz" for s in SEEDS]
                   + ["light0.6_hz", "light1.4_hz", "corners_hz"])
        for i in range(N):
            w.writerow([D.ids[i], groups[i], CLASSES_HZ[y[i]]] + [CLASSES_HZ[oof[s][i]] for s in SEEDS]
                       + [CLASSES_HZ[oof_light[0.6][i]], CLASSES_HZ[oof_light[1.4][i]], CLASSES_HZ[oof_corners[i]]])

    # --- the app run: seed 42, all 125 photos, epochs from the all-data inner splits ---------
    ep_final, ep_inner = epochs_from_inner(X, Y, groups, splits["final_inner"], DEPLOY_SEED, CFG)
    final = M.train_net(CFG, X, Y, groups, np.arange(N), epochs=ep_final, seed=DEPLOY_SEED, device=DEV)
    run_dir = write_app_run(final, ep_final, res)
    res["final"] = {"run": run_dir.name, "refit_epochs": ep_final, "inner_best_epochs": ep_inner}
    out = {"experiment_id": "chladni_v2_wittenberg_v1_20261108", "seed": DEPLOY_SEED,
           "prereg_sha256": sha256_file(PREREG), "splits_sha256": sha256_file(ROOT / "runs/protocol_v1/splits.json"),
           "config": CFG, "device": DEV, "torch": torch.__version__, "status": "exploratory (v2 designed after v1, same data)",
           "results": res, "fit_log": log}
    (OUT / "metrics.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    (OUT / "config.json").write_text(json.dumps({"config": CFG, "seeds": SEEDS}, indent=2), encoding="utf-8")
    r = res
    print(json.dumps({"accuracy_seed42": r["headline_accuracy_seed42"], "wilson": r["wilson95_frame_level"],
                      "macro_f1_w": r["seed42"]["weighted"]["macro_f1"],
                      "recall": {h: v["recall"] for h, v in r["seed42"]["weighted"]["per_class"].items()},
                      "other_seeds": r["accuracy_other_seeds"], "robustness": r["robustness"],
                      "corners_macro_f1": r["control_corners_only"]["macro_f1"],
                      "final_run": run_dir.name}, indent=1))


def write_app_run(final, epochs, res):
    """runs/<ts>/ in the format src/api.py and tasks.py check-model read for a CNN run."""
    used = [{"index": i + 1, "freq_hz": hz} for i, hz in enumerate(CLASSES_HZ)]  # config/classes.json indices
    full = json.loads((ROOT / "config/classes.json").read_text("utf-8"))["classes"]
    assert all({"index": c["index"], "freq_hz": c["freq_hz"]} in
               [{"index": f["index"], "freq_hz": f["freq_hz"]} for f in full] for c in used)
    config = pp.load_config()
    sig = opencv_signature()
    run_dir = runs_module.new_run_dir()
    net = final["net"].cpu()
    torch.save({"model": "wittenberg_v1", "n_classes": len(used), "classes": used,
                "state_dict": net.state_dict(), "preprocess_sha256": runs_module.preprocess_hash(config),
                "preprocess_version": config["version"],
                **{k: v for k, v in sig.items() if not k.startswith("opencv_dist")}}, run_dir / "model.pt")
    (run_dir / "config.json").write_text(json.dumps({
        "model": "wittenberg_v1", "n_classes": len(used), "classes": used,
        "hyperparameters": {**CFG, "epochs": epochs, "label_smoothing": 0.05, "cosine_to": 1e-5},
        "augmentation": "A2 = D4 symmetry + A1 (experiments/protocol_v2/preregister.yaml)",
        "trained_on": "all 125 accepted photos of 20261108-03/04 (protocol p.27 final refit)",
        "preprocess_version": config["version"], "preprocess_sha256": runs_module.preprocess_hash(config),
        "preprocess_config": config, **sig, "seed": DEPLOY_SEED,
        "determinism": {"seed": DEPLOY_SEED, "deterministic_algorithms": True}}, indent=2) + "\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps({
        "run": run_dir.name, "model": "wittenberg_v1", "seed": DEPLOY_SEED, "synthetic": False,
        "do_not_report": None,
        "out_of_fold_estimate": {
            "source": "runs/protocol_v2/metrics.json",
            "accuracy": res["headline_accuracy_seed42"], "wilson95_frame_level": res["wilson95_frame_level"],
            "macro_f1_group_weighted": res["seed42"]["weighted"]["macro_f1"],
            "status": "exploratory — v2 designed after v1 on the same photos; confirm on a new session"},
    }, indent=2) + "\n", encoding="utf-8")
    with open(run_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss"]); w.writeheader(); w.writerows(final["history"])
    runs_module.write_latest(run_dir)
    return run_dir


if __name__ == "__main__":
    main()
