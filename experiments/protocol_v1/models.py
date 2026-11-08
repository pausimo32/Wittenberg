"""Candidate families: B0 dummy, B1 HOG+LR, C1 small CNN (protocol p.15), W1 wittenberg_v1
(src/model.py, docs/04 recipe), F1/F2 frozen ImageNet embeddings + linear head."""
from __future__ import annotations

import math
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from common import ROOT, group_weights

sys.path.insert(0, str(ROOT / "src"))
from model import build_model  # noqa: E402  the project's wittenberg_v1, not a copy
from preprocess import _scale as scale_step  # noqa: E402  step 8, not a copy
import preprocess as _pp  # noqa: E402

N_CLASSES = 4
OUTPUT_CFG = _pp.load_config()["output"]


def to_unit(images_u8: np.ndarray) -> np.ndarray:
    """(N,128,128) uint8 -> (N,128,128) float32 in [0,1] through preprocess step 8."""
    return np.stack([scale_step(a, OUTPUT_CFG).reshape(128, 128) for a in images_u8])


# --- fixed feature extractors (no learned parameters -> may be cached globally) -------------

from hoglr import hog_features  # noqa: E402,F401  the one HOG (src/hoglr.py), not a copy


def backbone(name: str):
    import torchvision.models as tvm
    if name == "mobilenet_v3_small":
        m = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        dim, w = 576, "MobileNet_V3_Small_Weights.IMAGENET1K_V1"
    elif name == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)
        dim, w = 1280, "EfficientNet_B0_Weights.IMAGENET1K_V1"
    else:
        raise ValueError(name)
    m.eval()
    net = nn.Sequential(m.features, m.avgpool, nn.Flatten()).eval()
    return net, dim, w


IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def imagenet_input(unit: torch.Tensor) -> torch.Tensor:
    """(N,128,128) in [0,1] -> (N,3,224,224): whole plate kept (no crop), replicated gray."""
    x = unit.unsqueeze(1)
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
    return (x.repeat(1, 3, 1, 1) - IMNET_MEAN) / IMNET_STD


@torch.no_grad()
def embeddings(name: str, unit: np.ndarray) -> tuple[np.ndarray, str]:
    net, dim, wid = backbone(name)
    feats = []
    for i in range(0, len(unit), 16):
        feats.append(net(imagenet_input(torch.from_numpy(unit[i:i + 16]))).numpy())
    E = np.concatenate(feats)
    assert E.shape[1] == dim, (E.shape, dim)
    return E.astype(np.float32), wid


# --- linear heads -------------------------------------------------------------------------

def fit_linear(cfg: dict, X, y, groups, tr, te):
    """StandardScaler + LR/LinearSVC, both fitted on train only with group weights (mean 1)."""
    if cfg["family"] == "B0":
        strat = "most_frequent" if "most_frequent" in cfg["id"] else "stratified"
        clf = DummyClassifier(strategy=strat, random_state=42).fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])
        return {"pred": p.argmax(1), "scores": p, "score_type": "prob"}
    head = (LogisticRegression(C=cfg["C"], max_iter=5000) if cfg.get("head", "logreg") == "logreg"
            else LinearSVC(C=cfg["C"], max_iter=10000))
    pipe = make_pipeline(StandardScaler(), head)
    w = group_weights(np.asarray(groups)[tr]); w = w / w.mean()
    name = pipe.steps[1][0]
    pipe.fit(X[tr], y[tr], standardscaler__sample_weight=w, **{f"{name}__sample_weight": w})
    assert list(pipe.classes_) == list(range(N_CLASSES))
    if hasattr(pipe, "predict_proba"):
        s, st = pipe.predict_proba(X[te]), "prob"
    else:
        s, st = pipe.decision_function(X[te]), "margin"  # never used for log-loss/Brier
    return {"pred": s.argmax(1), "scores": s, "score_type": st, "model": pipe}


# --- small CNN, protocol p.15, exactly ------------------------------------------------------

class SmallChladniCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                                 nn.GroupNorm(4, cout), nn.ReLU(), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(1, 16), block(16, 32), block(32, 64), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(64, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


RECIPES = {
    "C1": dict(batch=4, max_epochs=100, patience=12, wd=1e-3, smoothing=0.0, cosine=False, norm=True),
    "W1": dict(batch=32, max_epochs=80, patience=12, wd=1e-2, smoothing=0.05, cosine=True, norm=False),
}


def make_net(family):
    return SmallChladniCNN(N_CLASSES) if family == "C1" else build_model("wittenberg_v1", N_CLASSES)


def augment_a1(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """A1 on (B,1,128,128) in [0,1]: rot ±5°, shift ±2% (p=.5); brightness/contrast 0.9–1.1 (p=.5)."""
    B, dev = x.shape[0], x.device
    u = lambda *s: torch.rand(*s, generator=g)  # noqa: E731  CPU generator -> reproducible
    geom = (u(B) < 0.5).float()
    ang = (u(B) * 2 - 1) * math.radians(5) * geom
    tx = (u(B) * 2 - 1) * 0.04 * geom  # grid coords span 2 -> 2% of the side = 0.04
    ty = (u(B) * 2 - 1) * 0.04 * geom
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1).to(dev)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    photo = (u(B) < 0.5).float()
    b = (1 + (u(B) * 0.2 - 0.1) * photo).view(B, 1, 1, 1).to(dev)
    c = (1 + (u(B) * 0.2 - 0.1) * photo).view(B, 1, 1, 1).to(dev)
    m = x.mean(dim=(2, 3), keepdim=True)
    return ((x - m) * c + m * b).clamp(0, 1)


def d4(x: torch.Tensor, k: int) -> torch.Tensor:
    """Symmetry k in 0..7 of the square on (B,1,H,W): k%4 quarter turns, k>=4 adds a reflection.
    Exact pixel permutations — no interpolation."""
    y = torch.rot90(x, k % 4, dims=(2, 3))
    return torch.flip(y, dims=(3,)) if k >= 4 else y


def augment_a2(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """A2 = a uniformly drawn D4 symmetry per image, then A1."""
    ks = torch.randint(0, 8, (x.shape[0],), generator=g)
    x = torch.cat([d4(x[i:i + 1], int(k)) for i, k in enumerate(ks)])
    return augment_a1(x, g)


def train_net(cfg, X, y, groups, tr, va=None, epochs=None, seed=42, device="cpu"):
    """Train one network. With `va`: early stopping on group-weighted val loss, returns best
    epoch + val probs. With `epochs`: fixed-length refit on `tr`, no validation peeking."""
    R = {**RECIPES[cfg["family"]], **{k: cfg[k] for k in ("batch",) if k in cfg}}
    torch.manual_seed(seed); np.random.seed(seed)
    net = make_net(cfg["family"]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=R["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=R["max_epochs"], eta_min=1e-5) if R["cosine"] else None
    crit = nn.CrossEntropyLoss(label_smoothing=R["smoothing"])
    g = torch.Generator().manual_seed(seed)
    prep = (lambda t: (t - 0.5) / 0.5) if R["norm"] else (lambda t: t)
    Xtr, ytr = X[tr], y[tr]
    if va is not None:
        Xva, yva = prep(X[va]), y[va]
        wva = torch.tensor(group_weights(np.asarray(groups)[va]), dtype=torch.float32, device=device)
        wva = wva / wva.sum()
    n_epochs = epochs if epochs is not None else R["max_epochs"]
    best = (math.inf, 0, None)
    since = 0
    history = []
    for ep in range(1, n_epochs + 1):
        net.train()
        perm = torch.randperm(len(tr), generator=g)
        ep_loss = 0.0
        for i in range(0, len(tr), R["batch"]):
            idx = perm[i:i + R["batch"]].to(device)
            xb = Xtr[idx]
            if cfg["aug"] == "A1":
                xb = augment_a1(xb, g)
            elif cfg["aug"] == "A2":
                xb = augment_a2(xb, g)
            loss = crit(net(prep(xb)), ytr[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            ep_loss += float(loss.detach()) * len(idx)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
        if sched:
            sched.step()
        history.append({"epoch": ep, "train_loss": ep_loss / len(tr)})
        if va is None:
            continue
        net.eval()
        with torch.no_grad():
            vl = (F.cross_entropy(net(Xva), yva, reduction="none") * wva).sum().item()
        if vl < best[0]:
            best = (vl, ep, {k: v.detach().clone() for k, v in net.state_dict().items()}); since = 0
        else:
            since += 1
            if since >= R["patience"]:
                break
    if va is not None:
        net.load_state_dict(best[2])
        net.eval()
        with torch.no_grad():
            p = F.softmax(net(Xva), 1).cpu().numpy()
        return {"best_epoch": best[1], "val_probs": p, "val_loss": best[0]}
    return {"net": net.eval(), "prep": prep, "history": history}


@torch.no_grad()
def predict_net(trained, X, idx):
    return F.softmax(trained["net"](trained["prep"](X[idx])), 1).cpu().numpy()


# --- cost, measured before any fitting -----------------------------------------------------

def measure_latency(families=("B1", "C1", "W1", "F1", "F2"), reps=50) -> dict:
    torch.set_num_threads(4)
    x8 = (np.random.default_rng(0).random((1, 128, 128)) * 255).astype(np.uint8)
    unit = torch.from_numpy(to_unit(x8))
    out = {}
    for fam in families:
        if fam == "B1":
            lr = make_pipeline(StandardScaler(), LogisticRegression()).fit(
                np.random.rand(8, 1764), np.arange(8) % 4)
            fn = lambda: lr.predict_proba(hog_features(x8))  # noqa: E731
            size = 1764 * 4 * 3 * 8
        elif fam in ("C1", "W1"):
            net = make_net(fam).eval()
            fn = lambda net=net: net(unit.unsqueeze(1))  # noqa: E731
            size = sum(p.numel() for p in net.parameters()) * 4
        else:
            net, dim, _ = backbone("mobilenet_v3_small" if fam == "F1" else "efficientnet_b0")
            fn = lambda net=net: net(imagenet_input(unit))  # noqa: E731
            size = sum(p.numel() for p in net.parameters()) * 4 + dim * 4 * 3 * 8
        with torch.no_grad():
            for _ in range(5):
                fn()
            ts = []
            for _ in range(reps):
                t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000)
        out[fam] = {"median_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95)),
                    "artifact_bytes": int(size)}
    return out
