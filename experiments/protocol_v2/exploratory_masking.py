"""EXPLORATORY (post hoc): does the deployed recipe rely on the corners/border?
Refits the seed-42 network per outer fold (same epochs rule) and evaluates the test images
(a) unchanged, (b) corners blanked, (c) only the plate interior kept (outer 8 px blanked)."""
import json, sys, numpy as np, torch
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import run_v2 as V
from common import CLASSES_HZ
V.R.determinism()
D = V.R.Data(); X, Y = D.X[V.DEV], D.Y[V.DEV]; y, g = D.y, D.groups
splits = json.loads((V.ROOT / "runs/protocol_v1/splits.json").read_text())
no_corners = X - V.corners_only(X)
interior = torch.zeros_like(X); interior[..., 8:-8, 8:-8] = X[..., 8:-8, 8:-8]
pred = {k: np.full(len(y), -1) for k in ("unchanged", "no_corners", "interior_only")}
for o in splits["outer"]:
    tr, te = np.array(o["train"]), np.array(o["test"])
    ep, _ = V.epochs_from_inner(X, Y, g, o["inner"], 42, V.CFG)
    net = V.M.train_net(V.CFG, X, Y, g, tr, epochs=ep, seed=42, device=V.DEV)
    for k, Xk in (("unchanged", X), ("no_corners", no_corners), ("interior_only", interior)):
        pred[k][te] = V.M.predict_net(net, Xk, te).argmax(1)
out = {k: {"accuracy": float(np.mean(p == y)),
           "recall": {str(h): float(np.mean(p[y == c] == c)) for c, h in enumerate(CLASSES_HZ)}}
       for k, p in pred.items()}
(V.ROOT / "runs/protocol_v2/exploratory_masking.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
