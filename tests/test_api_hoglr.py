"""`/api/*` serving a `hog_lr` run (ADR 0009). Every fixture is built here, in tmp dirs."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import api
import hoglr
import runs as runs_module
from calibrate import calibrate, load_calibration_config
from environment import opencv_signature
from imageio import decode_bytes
from preprocess import preprocess_stages

FULL_LIST = [{"index": i, "freq_hz": f} for i, f in enumerate([165, 497, 863, 1218, 2241])]
SUBSET = FULL_LIST[1:]  # 165 Hz has no captures: the model covers indices 1-4 only


def golden_png() -> bytes:
    rng = np.random.default_rng(42)
    image = rng.integers(20, 60, size=(600, 800, 3), dtype=np.uint8)
    image = np.dstack([image, np.full((600, 800, 1), 255, np.uint8)])
    cv2.fillConvexPoly(image, np.array([[120, 90], [520, 70], [560, 430], [90, 470]], np.int32),
                       (210, 210, 210, 255))
    for offset in range(-160, 200, 70):
        cv2.line(image, (260 + offset, 90), (300 + offset, 450), (25, 25, 25, 255), 6)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA))
    assert ok
    return buf.tobytes()


def build_hog_run(runs_root, config, classes=SUBSET, model_classes=None):
    rng = np.random.default_rng(0)
    imgs = (rng.random((len(classes) * 4, 128, 128)) > 0.97).astype(np.uint8) * 255
    for i in range(len(imgs)):
        imgs[i, :, 10 + 25 * (i % len(classes))] = 255  # a class-specific vertical line
    labels = np.arange(len(imgs)) % len(classes)
    groups = np.array([f"g{i // len(classes)}" for i in range(len(imgs))])
    model = hoglr.fit(imgs, labels, groups, model_classes or classes)
    run_dir = runs_module.new_run_dir(runs_root, stamp="20261108-120000")
    model.save(run_dir)
    (run_dir / "config.json").write_text(json.dumps({
        "model": "hog_lr", "classes": model.classes,
        "preprocess_version": config["version"],
        "preprocess_sha256": runs_module.preprocess_hash(config), **opencv_signature(),
    }), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps({"synthetic": False}), encoding="utf-8")
    classes_path = runs_root / "classes.json"
    classes_path.write_text(json.dumps({"classes": FULL_LIST}), encoding="utf-8")
    runs_module.write_latest(run_dir, runs_root)
    return run_dir, classes_path


@pytest.fixture(scope="module")
def served(tmp_path_factory, repo_root):
    config = json.loads((repo_root / "config" / "preprocess.json").read_text("utf-8"))
    root = tmp_path_factory.mktemp("hogruns")
    run_dir, classes_path = build_hog_run(root, config)
    app = api.create_app(runs_root=root, classes_path=classes_path, site_dir=root / "no-site")
    return {"client": TestClient(app), "run_dir": run_dir, "config": config, "root": root,
            "classes_path": classes_path}


def post(client, payload, **params):
    return client.post("/api/predict", files={"file": ("g.png", payload, "image/png")}, params=params)


def test_health_names_the_hog_model_on_cpu(served):
    body = served["client"].get("/api/health").json()
    assert body["model"] == "hog_lr" and body["device"] == "cpu"
    assert [c["freq_hz"] for c in body["classes"]] == [497, 863, 1218, 2241]


def test_predict_matches_the_direct_numpy_call(served):
    payload = golden_png()
    decoded = decode_bytes(payload, "g.png")
    found = calibrate(decoded.pixels, "upload", load_calibration_config())
    calib = ({"roi_mode": "homography", "corners": found.corners} if found.corners is not None
             else {"roi_mode": "center_crop"})
    stages = preprocess_stages(decoded.pixels, calib, served["config"])
    p = hoglr.HogLogReg.load(served["run_dir"]).predict_proba(hoglr.hog_features(stages.resized))[0]
    order = [int(i) for i in np.argsort(-p)[:3]]

    body = post(served["client"], payload).json()
    assert [t["p"] for t in body["top3"]] == [float(p[i]) for i in order]
    assert [t["freq_hz"] for t in body["top3"]] == [SUBSET[i]["freq_hz"] for i in order]
    # The frozen index from config/classes.json, not the position in the model's subset.
    assert [t["class_index"] for t in body["top3"]] == [SUBSET[i]["index"] for i in order]


def test_stages_are_the_pipeline_then_hog_contribution_probabilities(served):
    body = post(served["client"], golden_png()).json()
    assert [s["n"] for s in body["stages"]] == list(range(1, 12))
    assert [s["key"] for s in body["stages"][8:]] == ["hog", "contribution", "probabilities"]
    assert all(s["png_base64"] for s in body["stages"] if s["kind"] == "image")
    assert all(s["ms"] > 0 for s in body["stages"][:10])
    bars = body["stages"][-1]["bars"]
    assert sum(b["p"] for b in bars) == pytest.approx(1.0)
    assert [b["class_index"] for b in bars] == [1, 2, 3, 4]


def test_stages_false_returns_only_the_edge_map(served):
    body = post(served["client"], golden_png(), stages="false").json()
    assert [s["key"] for s in body["stages"]] == ["resized"]


def test_startup_refuses_a_subset_whose_index_means_another_frequency(served, tmp_path):
    wrong = [{"index": 1, "freq_hz": 497}, {"index": 2, "freq_hz": 870}]
    root = tmp_path / "r"
    root.mkdir()
    _, classes_path = build_hog_run(root, served["config"], classes=wrong)
    with pytest.raises(api.StartupRefused, match="class list has changed"):
        api.create_app(runs_root=root, classes_path=classes_path)


def test_startup_refuses_a_hog_run_without_its_model(served, tmp_path):
    root = tmp_path / "r2"
    root.mkdir()
    run_dir, classes_path = build_hog_run(root, served["config"])
    (run_dir / hoglr.MODEL_FILE).unlink()
    with pytest.raises(api.StartupRefused, match="model.npz"):
        api.create_app(runs_root=root, classes_path=classes_path)


def test_startup_refuses_a_hog_run_with_a_mutated_preprocess_config(served, tmp_path):
    root = tmp_path / "r3"
    root.mkdir()
    stale = {**served["config"], "canny": {**served["config"]["canny"], "threshold1": 51}}
    _, classes_path = build_hog_run(root, stale)
    with pytest.raises(api.StartupRefused, match="different"):
        api.create_app(runs_root=root, classes_path=classes_path)


def test_elapsed_stays_within_the_cpu_budget(served):
    post(served["client"], golden_png())  # warm-up
    body = post(served["client"], golden_png()).json()
    assert body["elapsed_ms"] < 150, body["timings_ms"]
