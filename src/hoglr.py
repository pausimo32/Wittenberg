"""HOG + logistic regression — the model ADR 0009's selection rule picked on 2026-11-08.

**Input.** Step 7 of `src/preprocess.py` (`Stages.resized`, the 128x128 uint8 edge map), the
array `data/processed/` holds. HOG is this classifier's feature extractor, not a
preprocessing step: `preprocess.py`, its config and its hash are untouched (CLAUDE.md rule 1).

**One HOG.** `HOG_PARAMS` and `hog_features` are the only definition. The protocol campaign
(`experiments/protocol_v1/`) imports them; nothing else may build a `cv2.HOGDescriptor`.

**Training uses scikit-learn; serving does not.** `fit` runs `StandardScaler` +
`LogisticRegression` (lbfgs, multinomial) with per-image weights `1/n_group` normalised to
mean 1 — exactly the protocol's B1 candidate. The fitted state is four arrays (scaler mean and
scale, coefficients, intercepts) written to `model.npz`. `HogLogReg.predict_proba` is
`softmax(((x - mean) / scale) @ coef.T + intercept)`, which is what scikit-learn computes for
a multinomial logistic regression; `tests/test_hoglr.py` pins the two together. No pickle
travels with the app, and the app does not depend on the scikit-learn version.

**Explanation is exact, not approximate.** The logit of class k is a sum over features
`z_j * coef_kj` plus the intercept. Grouping the 1764 terms by the HOG block they came from
gives each 32x32 block its exact share of the decision (`block_contributions`), which the
viewer paints over the plate in place of the CNN's Grad-CAM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

MODEL_NAME = "hog_lr"
MODEL_FILE = "model.npz"

#: win 128x128, block 32x32, stride 16, cell 16x16, 9 orientation bins -> 7x7 blocks x 36.
HOG_PARAMS = {"win_size": [128, 128], "block_size": [32, 32], "block_stride": [16, 16],
              "cell_size": [16, 16], "n_bins": 9}
_HOG = cv2.HOGDescriptor(
    tuple(HOG_PARAMS["win_size"]), tuple(HOG_PARAMS["block_size"]),
    tuple(HOG_PARAMS["block_stride"]), tuple(HOG_PARAMS["cell_size"]), HOG_PARAMS["n_bins"],
)
BLOCKS_PER_SIDE = 7
BLOCK_LEN = 36
N_FEATURES = BLOCKS_PER_SIDE * BLOCKS_PER_SIDE * BLOCK_LEN  # 1764
C_DEFAULT = 0.1


class HogLRError(RuntimeError):
    """The model file is missing, malformed or does not match the input."""


def hog_features(resized: np.ndarray) -> np.ndarray:
    """`(128,128)` or `(N,128,128)` uint8 step-7 arrays -> `(N, 1764)` float32."""
    a = np.asarray(resized)
    if a.ndim == 2:
        a = a[None]
    if a.dtype != np.uint8 or a.shape[1:] != (128, 128):
        raise HogLRError(f"hog_features expects uint8 (N,128,128), got {a.dtype} {a.shape}")
    return np.stack([_HOG.compute(np.ascontiguousarray(x)).ravel() for x in a]).astype(np.float32)


def block_origin(block: int) -> tuple[int, int]:
    """OpenCV orders blocks column-major: block = bx * 7 + by. Returns pixel (x, y)."""
    bx, by = divmod(block, BLOCKS_PER_SIDE)
    return bx * HOG_PARAMS["block_stride"][0], by * HOG_PARAMS["block_stride"][1]


class HogLogReg:
    """The fitted model, in numpy only."""

    def __init__(self, mean, scale, coef, intercept, classes: list[dict[str, Any]], meta=None):
        self.mean = np.asarray(mean, np.float64)
        self.scale = np.asarray(scale, np.float64)
        self.coef = np.asarray(coef, np.float64)
        self.intercept = np.asarray(intercept, np.float64)
        self.classes = [{"index": int(c["index"]), "freq_hz": int(c["freq_hz"])} for c in classes]
        self.meta = dict(meta or {})
        if self.coef.shape != (len(self.classes), N_FEATURES) or self.mean.shape != (N_FEATURES,):
            raise HogLRError(
                f"model arrays do not fit {len(self.classes)} classes x {N_FEATURES} features: "
                f"coef {self.coef.shape}, mean {self.mean.shape}"
            )

    def standardise(self, features: np.ndarray) -> np.ndarray:
        return (np.asarray(features, np.float64) - self.mean) / self.scale

    def logits(self, features: np.ndarray) -> np.ndarray:
        return self.standardise(features) @ self.coef.T + self.intercept

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        z = self.logits(features)
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def block_contributions(self, features: np.ndarray, k: int) -> np.ndarray:
        """`(7,7)` [by, bx]: each block's exact term in the logit of class position `k`,
        relative to the mean image (where every standardised feature is 0)."""
        terms = self.standardise(features).reshape(-1)[:N_FEATURES] * self.coef[k]
        per_block = terms.reshape(BLOCKS_PER_SIDE * BLOCKS_PER_SIDE, BLOCK_LEN).sum(axis=1)
        return per_block.reshape(BLOCKS_PER_SIDE, BLOCKS_PER_SIDE).T  # [bx,by] -> [by,bx]

    # --- persistence -------------------------------------------------------------------------

    def save(self, run_dir: Path | str) -> Path:
        path = Path(run_dir) / MODEL_FILE
        np.savez(path, mean=self.mean, scale=self.scale, coef=self.coef, intercept=self.intercept,
                 classes=np.array(json.dumps(self.classes)), meta=np.array(json.dumps(self.meta)),
                 hog_params=np.array(json.dumps(HOG_PARAMS)))
        return path

    @classmethod
    def load(cls, run_dir: Path | str) -> "HogLogReg":
        path = Path(run_dir) / MODEL_FILE
        try:
            with np.load(path, allow_pickle=False) as z:
                if json.loads(str(z["hog_params"])) != HOG_PARAMS:
                    raise HogLRError(f"{path} was fitted with different HOG parameters")
                return cls(z["mean"], z["scale"], z["coef"], z["intercept"],
                           json.loads(str(z["classes"])), json.loads(str(z["meta"])))
        except (OSError, KeyError, ValueError) as exc:
            raise HogLRError(f"{path} cannot be loaded ({exc})") from exc


def fit(resized: np.ndarray, labels: np.ndarray, groups: np.ndarray,
        classes: list[dict[str, Any]], C: float = C_DEFAULT) -> HogLogReg:
    """Fit on step-7 arrays. `labels` are positions into `classes`; `groups` the ADR 0009
    unit, used for weights `1/n_group` (mean 1) on both the scaler and the classifier."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = hog_features(resized)
    y = np.asarray(labels)
    if sorted(set(y.tolist())) != list(range(len(classes))):
        raise HogLRError("every class must have training images")
    _, inv, counts = np.unique(np.asarray(groups), return_inverse=True, return_counts=True)
    w = 1.0 / counts[inv]
    w = w / w.mean()
    scaler = StandardScaler().fit(X, sample_weight=w)
    clf = LogisticRegression(C=C, max_iter=5000).fit(scaler.transform(X), y, sample_weight=w)
    import sklearn
    coef, intercept = clf.coef_, clf.intercept_
    if coef.shape[0] == 1:
        # Two classes: scikit-learn stores one logit z for class 1, p1 = sigmoid(z). Softmax
        # over logits [0, z] is the same probability, so the numpy path stays one formula.
        coef = np.vstack([np.zeros_like(coef), coef])
        intercept = np.concatenate([[0.0], intercept])
    return HogLogReg(scaler.mean_, scaler.scale_, coef, intercept, classes,
                     meta={"C": C, "solver": clf.solver, "n_iter": int(np.max(clf.n_iter_)),
                           "sklearn": sklearn.__version__, "n_train": int(len(y))})


# --- display (viewer stages; never re-enters the pipeline) -----------------------------------

def hog_glyphs(resized: np.ndarray, scale: int = 4) -> np.ndarray:
    """Per 16x16 cell, one line per orientation bin, brightness = gradient energy. Uint8."""
    a = np.asarray(resized, np.float32)
    gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=1)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    ang = np.mod(ang, 180.0)
    n, cell = HOG_PARAMS["n_bins"], HOG_PARAMS["cell_size"][0]
    side = 128 // cell
    hist = np.zeros((side, side, n), np.float32)
    bins = np.minimum((ang / (180.0 / n)).astype(int), n - 1)
    for cy in range(side):
        for cx in range(side):
            sl = (slice(cy * cell, (cy + 1) * cell), slice(cx * cell, (cx + 1) * cell))
            hist[cy, cx] = np.bincount(bins[sl].ravel(), mag[sl].ravel(), n)
    hist /= max(float(hist.max()), 1e-9)
    s = cell * scale
    out = np.zeros((side * s, side * s), np.uint8)
    for cy in range(side):
        for cx in range(side):
            c = np.array([cx * s + s / 2, cy * s + s / 2])
            for b in range(n):
                v = hist[cy, cx, b]
                if v < 0.05:
                    continue
                t = np.deg2rad(b * 180.0 / n + 90.0)  # draw the edge, perpendicular to the gradient
                d = np.array([np.cos(t), np.sin(t)]) * (s / 2 - 1)
                p, q = (c - d).astype(int), (c + d).astype(int)
                cv2.line(out, tuple(p), tuple(q), int(255 * v), 1, cv2.LINE_AA)
    return out


def contribution_map(model: HogLogReg, features: np.ndarray, k: int, size: int) -> np.ndarray:
    """`(size,size)` in [0,1]: positive block terms spread over each block's 32x32 footprint."""
    blocks = np.clip(model.block_contributions(features, k), 0, None)
    heat = np.zeros((128, 128), np.float64)
    bs = HOG_PARAMS["block_size"][0]
    for by in range(BLOCKS_PER_SIDE):
        for bx in range(BLOCKS_PER_SIDE):
            x, y = block_origin(bx * BLOCKS_PER_SIDE + by)
            heat[y:y + bs, x:x + bs] += blocks[by, bx]
    if heat.max() > 0:
        heat /= heat.max()
    return cv2.resize(heat.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
