"""Campaign v3: the v2 recipe on preprocessing 1.2.0 (border discarded), plus session-cue
controls. Writes runs/protocol_v3/ and the app run (seed 42, all 125 photos).

    python experiments/protocol_v3/run_v3.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "protocol_v2"))
import run_v2 as V  # noqa: E402  (brings protocol_v1 + src onto the path)
from common import CLASSES_HZ, bootstrap_groups, metrics, sha256_file  # noqa: E402
import preprocess as pp  # noqa: E402

OUT = ROOT / "runs" / "protocol_v3"
PREREG = HERE / "preregister.yaml"
V.OUT = OUT
V.PREREG = PREREG


def shuffled(X, seed=123):
    g = torch.Generator().manual_seed(seed)
    flat = X.flatten(1).clone()
    for i in range(flat.shape[0]):
        flat[i] = flat[i, torch.randperm(flat.shape[1], generator=g).to(flat.device)]
    return flat.view_as(X)


def main():
    assert pp.load_config()["version"] == "1.2.0", "v3 needs preprocessing 1.2.0"
    V.R.determinism()
    OUT.mkdir(parents=True, exist_ok=True)
    D = V.R.Data()
    X, Y, y, groups, N = D.X[V.DEV], D.Y[V.DEV], D.y, D.groups, len(D.y)
    assert float(X[..., :8, :].abs().sum()) == 0.0, "border band is not empty: re-ingest under 1.2.0"
    splits = json.loads((ROOT / "runs/protocol_v1/splits.json").read_text())
    light = {f: V.lighting_images(D.man, f) for f in (0.6, 1.4)}
    Xs = shuffled(X)
    hog = V.M.hog_features((X[:, 0].cpu().numpy() * 255).round().astype(np.uint8))

    oof = {s: np.full(N, -1) for s in V.SEEDS}
    oof_d4 = np.full((N, 8), -1)
    oof_light = {f: np.full(N, -1) for f in light}
    oof_shuf = np.full(N, -1)
    oof_hog = np.full(N, -1)
    log, t0 = [], time.perf_counter()
    for o in splits["outer"]:
        tr, te = np.array(o["train"]), np.array(o["test"])
        for seed in V.SEEDS:
            ep, ib = V.epochs_from_inner(X, Y, groups, o["inner"], seed, V.CFG)
            net = V.M.train_net(V.CFG, X, Y, groups, tr, epochs=ep, seed=seed, device=V.DEV)
            oof[seed][te] = V.M.predict_net(net, X, te).argmax(1)
            log.append({"fold": o["fold"], "seed": seed, "inner_best_epochs": ib, "refit_epochs": ep})
            if seed == V.DEPLOY_SEED:
                for k in range(8):
                    oof_d4[te, k] = V.M.predict_net(net, V.M.d4(X, k), te).argmax(1)
                for f, Xl in light.items():
                    oof_light[f][te] = V.M.predict_net(net, Xl, te).argmax(1)
        eps, _ = V.epochs_from_inner(Xs, Y, groups, o["inner"], V.DEPLOY_SEED, V.CFG)
        nets = V.M.train_net(V.CFG, Xs, Y, groups, tr, epochs=eps, seed=V.DEPLOY_SEED, device=V.DEV)
        oof_shuf[te] = V.M.predict_net(nets, Xs, te).argmax(1)
        oof_hog[te] = V.M.fit_linear({"id": "hog", "family": "B1", "C": 0.1, "head": "logreg"},
                                     hog, y, groups, tr, te)["pred"]
        print(f"fold {o['fold']} done  acc(seed42)={np.mean(oof[42][te] == y[te]):.3f}  "
              f"{time.perf_counter() - t0:.0f}s", flush=True)

    k42 = int((oof[42] == y).sum())
    res = {
        "headline_accuracy_seed42": k42 / N, "correct": k42, "n": N,
        "wilson95_frame_level": V.wilson(k42, N),
        "seed42": {"weighted": metrics(y, oof[42], groups), "unweighted": metrics(y, oof[42])},
        "accuracy_other_seeds": {s: float(np.mean(oof[s] == y)) for s in V.SEEDS},
        "robustness": {
            "d4_test_accuracy_mean": float(np.mean(oof_d4 == y[:, None])),
            "d4_test_accuracy_per_symmetry": [float(np.mean(oof_d4[:, k] == y)) for k in range(8)],
            "lighting_x0.6_accuracy": float(np.mean(oof_light[0.6] == y)),
            "lighting_x1.4_accuracy": float(np.mean(oof_light[1.4] == y)),
        },
        "control_pixel_shuffle": metrics(y, oof_shuf, groups),
        "reference_hog_lr_1.2.0": metrics(y, oof_hog, groups),
        "bootstrap_groups": bootstrap_groups(y, {"seed42": oof[42], "shuffle": oof_shuf, "hog": oof_hog}, groups),
    }
    with open(OUT / "oof_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "group_id", "y_true_hz"] + [f"seed{s}_hz" for s in V.SEEDS]
                   + ["light0.6_hz", "light1.4_hz", "shuffle_hz", "hog_hz"])
        for i in range(N):
            w.writerow([D.ids[i], groups[i], CLASSES_HZ[y[i]]] + [CLASSES_HZ[oof[s][i]] for s in V.SEEDS]
                       + [CLASSES_HZ[oof_light[0.6][i]], CLASSES_HZ[oof_light[1.4][i]],
                          CLASSES_HZ[oof_shuf[i]], CLASSES_HZ[oof_hog[i]]])

    ep_final, ep_inner = V.epochs_from_inner(X, Y, groups, splits["final_inner"], V.DEPLOY_SEED, V.CFG)
    final = V.M.train_net(V.CFG, X, Y, groups, np.arange(N), epochs=ep_final, seed=V.DEPLOY_SEED, device=V.DEV)
    run_dir = V.write_app_run(final, ep_final, res)
    m = json.loads((run_dir / "metrics.json").read_text())
    m["out_of_fold_estimate"]["source"] = "runs/protocol_v3/metrics.json"
    m["preprocess_version"] = "1.2.0"
    (run_dir / "metrics.json").write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8")
    res["final"] = {"run": run_dir.name, "refit_epochs": ep_final, "inner_best_epochs": ep_inner}
    (OUT / "metrics.json").write_text(json.dumps({
        "experiment_id": "chladni_v3_session_independence_20261108", "seed": V.DEPLOY_SEED,
        "prereg_sha256": sha256_file(PREREG), "splits_sha256": sha256_file(ROOT / "runs/protocol_v1/splits.json"),
        "preprocess_version": "1.2.0", "config": V.CFG, "device": V.DEV, "torch": torch.__version__,
        "status": "exploratory (same photos as v1/v2)", "results": res, "fit_log": log}, indent=2, default=str),
        encoding="utf-8")
    (OUT / "config.json").write_text(json.dumps({"config": V.CFG, "seeds": V.SEEDS, "preprocess": "1.2.0"}, indent=2),
                                     encoding="utf-8")
    r = res
    rec = lambda mm: {h: round(v["recall"], 3) for h, v in mm["per_class"].items()}  # noqa: E731
    print(json.dumps({"accuracy_seed42": r["headline_accuracy_seed42"], "wilson": r["wilson95_frame_level"],
                      "macro_f1_w": r["seed42"]["weighted"]["macro_f1"], "recall": rec(r["seed42"]["weighted"]),
                      "confusion": r["seed42"]["unweighted"]["confusion"],
                      "other_seeds": r["accuracy_other_seeds"], "robustness": r["robustness"],
                      "shuffle_macro_f1": r["control_pixel_shuffle"]["macro_f1"], "shuffle_recall": rec(r["control_pixel_shuffle"]),
                      "hog_macro_f1": r["reference_hog_lr_1.2.0"]["macro_f1"], "hog_recall": rec(r["reference_hog_lr_1.2.0"]),
                      "bootstrap": r["bootstrap_groups"]["seed42"], "final_run": run_dir.name}, indent=1))


if __name__ == "__main__":
    main()
