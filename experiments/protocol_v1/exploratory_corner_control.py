"""EXPLORATORY, post hoc (added after seeing the background-ring result; not pre-registered).
Corners-only control: keep four 14x14 corner patches (ArUco markers + table), zero the rest.
Same outer splits, HOG+LR C=1. Within session 03 corners carry no pattern, so 497/863/1218
should sit near chance; if 2241 is still recognised, that is the session shortcut."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import models as M
from common import ROOT, CLASSES_HZ, load_manifest, load_images, metrics
OUT = ROOT / "runs" / "protocol_v1"
man = load_manifest(); imgs = load_images(man)
y = np.array([r["y"] for r in man]); g = np.array([r["group_id"] for r in man])
c = 14
mask = np.zeros((128, 128), bool)
mask[:c, :c] = mask[:c, -c:] = mask[-c:, :c] = mask[-c:, -c:] = True
X = M.hog_features(np.where(mask, imgs, 0).astype(np.uint8))
splits = json.loads((OUT / "splits.json").read_text())
oof = np.full(len(y), -1)
cfg = {"id": "corners", "family": "BG", "C": 1.0, "head": "logreg"}
for o in splits["outer"]:
    te = np.array(o["test"]); oof[te] = M.fit_linear(cfg, X, y, g, np.array(o["train"]), te)["pred"]
m = metrics(y, oof, g)
s03 = y < 3
m03 = (oof[s03] == y[s03]).mean()
is2241 = metrics((y == 3).astype(int), (oof == 3).astype(int))  # uses 4-class labels; only cols 0/1 meaningful
res = {"exploratory": True, "macro_f1_4class_weighted": m["macro_f1"],
       "recall": {h: v["recall"] for h, v in m["per_class"].items()},
       "accuracy_within_session03_3class": float(m03), "chance_within_session03": 1/3,
       "confusion_weighted": m["confusion"]}
(OUT / "exploratory_corner_control.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=1))
