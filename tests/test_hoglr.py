"""`src/hoglr.py` — the numpy inference must be scikit-learn's, and the explanation exact."""

import json

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import hoglr
from groups import preparation_groups

CLASSES = [{"index": 1, "freq_hz": 497}, {"index": 2, "freq_hz": 863}, {"index": 4, "freq_hz": 2241}]


def edge_maps(n_per_class=6, seed=0):
    """Three visibly different line families on a 128x128 edge map, with jitter."""
    rng = np.random.default_rng(seed)
    imgs, labels, groups = [], [], []
    for c in range(3):
        for i in range(n_per_class):
            a = np.zeros((128, 128), np.uint8)
            off = int(rng.integers(-4, 5))
            if c == 0:
                a[:, 40 + off] = a[:, 88 + off] = 255
            elif c == 1:
                a[40 + off, :] = a[88 + off, :] = 255
            else:
                np.fill_diagonal(a[:, :], 255)
            a[rng.integers(0, 128, 30), rng.integers(0, 128, 30)] = 255
            imgs.append(a); labels.append(c); groups.append(f"g{i % 3}")
    return np.stack(imgs), np.array(labels), np.array(groups)


@pytest.fixture(scope="module")
def fitted():
    X, y, g = edge_maps()
    return hoglr.fit(X, y, g, CLASSES), X, y, g


def test_numpy_probabilities_equal_scikit_learns(fitted):
    model, X, y, g = fitted
    F = hoglr.hog_features(X)
    _, inv, counts = np.unique(g, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inv]; w /= w.mean()
    sc = StandardScaler().fit(F, sample_weight=w)
    ref = LogisticRegression(C=hoglr.C_DEFAULT, max_iter=5000).fit(sc.transform(F), y, sample_weight=w)
    np.testing.assert_allclose(model.predict_proba(F), ref.predict_proba(sc.transform(F)), atol=1e-9)


def test_two_classes_give_scikit_learns_binary_probabilities():
    X, y, g = edge_maps()
    keep = y < 2
    model = hoglr.fit(X[keep], y[keep], g[keep], CLASSES[:2])
    F = hoglr.hog_features(X[keep])
    _, inv, counts = np.unique(g[keep], return_inverse=True, return_counts=True)
    w = 1.0 / counts[inv]; w /= w.mean()
    sc = StandardScaler().fit(F, sample_weight=w)
    ref = LogisticRegression(C=hoglr.C_DEFAULT, max_iter=5000).fit(sc.transform(F), y[keep], sample_weight=w)
    np.testing.assert_allclose(model.predict_proba(F), ref.predict_proba(sc.transform(F)), atol=1e-9)


def test_block_contributions_sum_to_the_logit_minus_the_intercept(fitted):
    model, X, _, _ = fitted
    F = hoglr.hog_features(X[:1])
    for k in range(len(CLASSES)):
        total = model.block_contributions(F, k).sum()
        assert total == pytest.approx(model.logits(F)[0, k] - model.intercept[k], abs=1e-8)


def test_block_layout_is_column_major():
    """A line inside the top-right cell only touches the blocks whose footprint covers it."""
    a = np.zeros((128, 128), np.uint8)
    a[4:12, 120] = 255  # cell (cx=7, cy=0)
    f = hoglr.hog_features(a)[0].reshape(49, 36)
    lit = {b for b in range(49) if np.abs(f[b]).sum() > 0}
    assert lit == {6 * 7 + 0}
    assert hoglr.block_origin(42) == (96, 0)


def test_save_load_round_trip_without_pickle(fitted, tmp_path):
    model, X, _, _ = fitted
    model.save(tmp_path)
    again = hoglr.HogLogReg.load(tmp_path)
    F = hoglr.hog_features(X)
    np.testing.assert_array_equal(again.predict_proba(F), model.predict_proba(F))
    assert again.classes == CLASSES
    with np.load(tmp_path / hoglr.MODEL_FILE, allow_pickle=False) as z:  # loads with pickle off
        assert set(z.files) >= {"mean", "scale", "coef", "intercept", "classes"}


def test_load_refuses_a_model_fitted_with_other_hog_parameters(fitted, tmp_path):
    model, _, _, _ = fitted
    path = model.save(tmp_path)
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["hog_params"] = np.array(json.dumps({**hoglr.HOG_PARAMS, "n_bins": 12}))
    np.savez(path, **arrays)
    with pytest.raises(hoglr.HogLRError, match="different HOG parameters"):
        hoglr.HogLogReg.load(tmp_path)


def test_hog_features_reject_a_float_tensor():
    with pytest.raises(hoglr.HogLRError):
        hoglr.hog_features(np.zeros((128, 128), np.float32))


def test_preparation_groups_follow_reseed_every_and_merge_a_tiny_tail(tmp_path):
    session = tmp_path / "raw" / "20990101-01"
    session.mkdir(parents=True)
    (session / "session.json").write_text(json.dumps({"station": {"reseed_every": 3}}))
    rows = [{"session_id": "20990101-01", "freq_hz": 497, "image_id": f"20990101-01_497_{n:04d}"}
            for n in (1, 2, 3, 4, 5, 6, 8)]  # 7 accepted frames, a gap at 7
    g = preparation_groups(rows, tmp_path)
    assert [g[r["image_id"]][-3:] for r in rows] == ["b00"] * 3 + ["b01"] * 4
