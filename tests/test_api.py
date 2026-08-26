"""F5 — the serving API. `docs/05-webapp-spec.md` §3, §6.

The two tests that matter are the consistency check and the startup guard, and they test
opposite halves of the same property: that what the app serves is what the checkpoint was
trained on.

**Consistency, not parity.** `test_predict_matches_the_direct_call` asserts *exact*
equality — same probabilities, same edge map bytes — between `/api/predict` and calling
`imageio.decode_bytes` + `preprocess.preprocess_stages` + the model by hand. There is one
implementation, so any difference is the API having grown a step of its own, which is the
failure `docs/adr/0004` reorganised the architecture to make impossible. Rounding tolerance
would hide exactly the drift the test exists to catch, so there is none.

**The guard.** `test_startup_refuses_a_mutated_config` edits the real
`config/preprocess.json` and asserts the app refuses to start, with both hashes in the
message. It restores the file byte-for-byte in a fixture teardown, including on failure.
"""

from __future__ import annotations

import base64
import copy
import json
import time

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

import api
import runs as runs_module
from calibrate import calibrate, load_calibration_config
from dataset import to_model_input
from environment import opencv_signature
from imageio import decode_bytes
from model import build_model
from preprocess import preprocess_stages

SEED = 42

#: §7: "If `elapsed_ms` exceeds 150 ms, something is wrong — most likely the model is being
#: reloaded per request instead of once at startup." Asserted on CPU, the supported device.
ELAPSED_BUDGET_MS = 150


# --------------------------------------------------------------------------- #
# a run to serve
# --------------------------------------------------------------------------- #

def build_run(runs_root, config, classes, metrics, stamp="20260825-120000"):
    """A finished run directory: an untrained `wittenberg_v1` with a valid provenance.

    The weights are random and it does not matter — none of these tests asserts *what* the
    model answers, only that the API answers what the same weights answer when driven by
    hand. Training a real checkpoint here would make the suite slow and the assertions no
    stronger. `metrics` is a parameter because `do_not_report` is what distinguishes a run
    the TDR may quote from one it may not, and both have to be served.
    """
    run_dir = runs_module.new_run_dir(runs_root, stamp=stamp)

    torch.manual_seed(SEED)
    model = build_model("wittenberg_v1", len(classes))
    signature = opencv_signature()

    torch.save(
        {
            "model": "wittenberg_v1",
            "n_classes": len(classes),
            "classes": classes,
            "state_dict": model.state_dict(),
            "preprocess_sha256": runs_module.preprocess_hash(config),
            "preprocess_version": config["version"],
            **{k: v for k, v in signature.items() if not k.startswith("opencv_dist")},
        },
        run_dir / "model.pt",
    )
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "model": "wittenberg_v1",
                "n_classes": len(classes),
                "classes": classes,
                "preprocess_version": config["version"],
                "preprocess_sha256": runs_module.preprocess_hash(config),
                **signature,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "history.csv").write_text("epoch\n1\n", encoding="utf-8")

    classes_path = runs_root / "classes.json"
    classes_path.write_text(
        json.dumps({"version": "1.0.0", "n_classes": len(classes), "classes": classes}),
        encoding="utf-8",
    )

    runs_module.write_latest(run_dir, runs_root)
    return {
        "runs_root": runs_root,
        "run_dir": run_dir,
        "classes_path": classes_path,
        "classes": classes,
        "config": config,
    }


@pytest.fixture(scope="module")
def served_run(tmp_path_factory, repo_root):
    """A run whose numbers are reportable: no `do_not_report` in its metrics."""
    from conftest_f4a import class_list

    return build_run(
        tmp_path_factory.mktemp("runs"),
        json.loads((repo_root / "config" / "preprocess.json").read_text("utf-8")),
        class_list(),
        {"run": "20260825-120000", "synthetic": False},
    )


@pytest.fixture(scope="module")
def app(served_run):
    return api.create_app(
        runs_root=served_run["runs_root"],
        classes_path=served_run["classes_path"],
        web_dist=served_run["runs_root"] / "no-frontend-yet",
    )


@pytest.fixture(scope="module")
def client(app):
    return TestClient(app)


@pytest.fixture(scope="module")
def golden_png(tmp_path_factory) -> bytes:
    """The reference upload: the synthetic capture, PNG-encoded in memory.

    Built with the same generator as `conftest.plate_photo` (seed 42) rather than the
    fixture itself, which is function-scoped.
    """
    rng = np.random.default_rng(SEED)
    image = rng.integers(20, 60, size=(600, 800, 3), dtype=np.uint8)
    image = np.dstack([image, np.full((600, 800, 1), 255, np.uint8)])
    cv2.fillConvexPoly(
        image,
        np.array([[120, 90], [520, 70], [560, 430], [90, 470]], dtype=np.int32),
        (210, 210, 210, 255),
    )
    for offset in range(-160, 200, 70):
        cv2.line(image, (260 + offset, 90), (300 + offset, 450), (25, 25, 25, 255), 6)
    for offset in range(-120, 160, 90):
        cv2.line(image, (110, 240 + offset), (560, 220 + offset), (25, 25, 25, 255), 6)

    ok, buffer = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA))
    assert ok
    return buffer.tobytes()


def post(client, payload: bytes, **params):
    return client.post(
        "/api/predict", files={"file": ("golden.png", payload, "image/png")}, params=params
    )


# --------------------------------------------------------------------------- #
# §3 — /api/health
# --------------------------------------------------------------------------- #

def test_health_reports_the_run_the_version_and_the_device(client, served_run):
    body = client.get("/api/health").json()
    assert body["model_run"] == served_run["run_dir"].name
    assert body["preprocess_version"] == served_run["config"]["version"]
    assert body["classes"] == served_run["classes"]
    assert body["model"] == "wittenberg_v1"

    # ADR 0005: CUDA if it happens to be there, CPU otherwise, and CPU is a pass.
    assert body["device"] == ("cuda" if torch.cuda.is_available() else "cpu")


def test_health_reports_the_hash_the_gate_passed_on(client, served_run):
    body = client.get("/api/health").json()
    assert body["preprocess_sha256"] == runs_module.preprocess_hash(served_run["config"])
    assert body["opencv"]["version"] == opencv_signature()["opencv_version"]


def test_health_says_nothing_to_qualify_a_reportable_run(client):
    body = client.get("/api/health").json()
    assert body["do_not_report"] is None
    assert body["synthetic"] is False


def test_health_carries_the_do_not_report_of_a_synthetic_run(tmp_path, served_run):
    """F6 hangs a visible banner on this field, so it has to survive the trip verbatim.

    That a synthetic run *gets* the field is `test_train.py`'s assertion; this is the other
    half — that serving one does not quietly drop it. A ~100 % accuracy shown with nothing
    on screen to qualify it is the app misleading a participant.
    """
    warning = (
        "This run used tools/synth_capture.py sessions. No number from this file may "
        "appear in the TDR (docs/00-master-plan.md F2.5)."
    )
    run = build_run(
        tmp_path,
        served_run["config"],
        served_run["classes"],
        {"run": "synthetic", "synthetic": True, "do_not_report": warning},
    )
    app = api.create_app(runs_root=run["runs_root"], classes_path=run["classes_path"])

    body = TestClient(app).get("/api/health").json()
    assert body["do_not_report"] == warning
    assert body["synthetic"] is True


def test_startup_refuses_a_run_whose_metrics_cannot_be_read(tmp_path, served_run):
    """Unreadable provenance defaults to refusal, never to 'probably reportable'."""
    run = build_run(
        tmp_path, served_run["config"], served_run["classes"], {"run": "broken"}
    )
    (run["run_dir"] / "metrics.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(api.StartupRefused, match="metrics.json cannot be read"):
        api.create_app(runs_root=run["runs_root"], classes_path=run["classes_path"])


# --------------------------------------------------------------------------- #
# §6 — the consistency check
# --------------------------------------------------------------------------- #

def direct(served_run, payload: bytes):
    """What `/api/predict` must reproduce: the pipeline and the model, driven by hand."""
    config = served_run["config"]
    decoded = decode_bytes(payload, "golden.png")

    found = calibrate(decoded.pixels, "upload", load_calibration_config())
    calib = (
        {"roi_mode": "homography", "corners": found.corners}
        if found.corners is not None
        else {"roi_mode": "center_crop"}
    )

    stages = preprocess_stages(decoded.pixels, calib, config)

    checkpoint = torch.load(
        served_run["run_dir"] / "model.pt", map_location="cpu", weights_only=False
    )
    model = build_model(checkpoint["model"], checkpoint["n_classes"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    x = to_model_input(torch.from_numpy(stages.resized).unsqueeze(0), config, "cpu")
    probabilities = torch.softmax(model(x), dim=1)[0].detach().cpu().numpy()
    return stages, probabilities


@pytest.mark.skipif(torch.cuda.is_available(), reason="the direct call runs on CPU")
def test_predict_matches_the_direct_call(client, served_run, golden_png):
    stages, probabilities = direct(served_run, golden_png)
    body = post(client, golden_png).json()

    order = [int(i) for i in np.argsort(-probabilities)[:3]]
    assert [t["class_index"] for t in body["top3"]] == order
    # Exact, not approximate — see the module docstring.
    assert [t["p"] for t in body["top3"]] == [float(probabilities[i]) for i in order]
    assert [t["freq_hz"] for t in body["top3"]] == [
        served_run["classes"][i]["freq_hz"] for i in order
    ]

    edge_stage = next(s for s in body["stages"] if s["n"] == 7)
    ok, expected = cv2.imencode(".png", stages.resized)
    assert ok
    # The image shown IS the one that produced the answer (§8): same bytes, not a re-run.
    assert edge_stage["png_base64"] == base64.b64encode(expected.tobytes()).decode("ascii")


def test_predict_returns_all_thirteen_stages(client, golden_png):
    body = post(client, golden_png).json()
    assert [s["n"] for s in body["stages"]] == list(range(1, 14))

    images = [s for s in body["stages"] if s["kind"] == "image"]
    assert len(images) == 12
    assert all(s["png_base64"] for s in images)

    probabilities = body["stages"][-1]
    assert probabilities["kind"] == "bars"
    assert sum(bar["p"] for bar in probabilities["bars"]) == pytest.approx(1.0)


def test_every_stage_carries_its_own_measured_time(client, golden_png):
    """§8: 'Show the real measured time of each step next to it'."""
    body = post(client, golden_png).json()
    measured = {s["n"]: s["ms"] for s in body["stages"]}

    assert all(measured[n] > 0.0 for n in range(1, 13)), measured
    # Stage 13 is a bar chart of a number already computed, not a step with a cost.
    assert measured[13] is None


def test_stage_times_are_measured_inside_the_phase_that_contains_them(
    client, golden_png
):
    """The per-step numbers come from the real run, so they must fit inside its phase.

    If a stage ever reported more time than the phase it happened in, the stage number
    would be coming from somewhere other than the call that produced the image.
    """
    body = post(client, golden_png).json()
    steps = sum(s["ms"] for s in body["stages"] if 2 <= s["n"] <= 8)

    assert steps <= body["timings_ms"]["preprocess"]
    assert sum(s["ms"] for s in body["stages"] if 9 <= s["n"] <= 11) <= (
        body["timings_ms"]["inference"]
    )


def test_stage_one_is_capped_for_display_and_says_the_original_size(
    client, golden_png
):
    """A 12 Mpx capture encodes to 24 MB and 459 ms uncapped — a 3x breach of §7's budget.

    The cap is on the picture sent to be looked at, never on the array the pipeline reads, and
    the response carries both sizes so the UI can show the real one rather than implying the
    PNG is full-size.
    """
    body = post(client, golden_png).json()
    capture = next(s for s in body["stages"] if s["n"] == 1)

    assert capture["shape"][:2] == [600, 800]  # what preprocess_stages received
    assert max(capture["display_shape"][:2]) <= api.DISPLAY_MAX_SIDE
    assert capture["display_shape"] != capture["shape"]

    # A stage that is already small is encoded as it is, and says nothing about a display copy.
    assert next(s for s in body["stages"] if s["n"] == 6)["display_shape"] is None


def test_stage_parameters_come_from_the_config_not_from_the_ui(
    client, served_run, golden_png
):
    """§8: 'the parameters read from config/preprocess.json, not retyped into the UI'."""
    body = post(client, golden_png).json()
    canny = next(s for s in body["stages"] if s["key"] == "edges")
    assert canny["call"] == "cv2.Canny"
    assert canny["params"] == served_run["config"]["canny"]


def test_stages_false_returns_only_the_final_edge_map(client, golden_png):
    body = post(client, golden_png, stages="false").json()
    assert len(body["stages"]) == 1
    assert body["stages"][0]["key"] == "resized"
    assert body["top3"]


def test_an_upload_without_markers_reports_the_roi_mode_it_fell_back_to(
    client, golden_png
):
    body = post(client, golden_png).json()
    assert body["roi_mode"] == "center_crop"
    assert body["markers_found"] == 0


# --------------------------------------------------------------------------- #
# §7 — the performance budget
# --------------------------------------------------------------------------- #

def test_elapsed_ms_stays_within_the_budget(client, golden_png):
    post(client, golden_png)  # warm-up: the first forward pass allocates
    body = post(client, golden_png, stages="false").json()
    assert body["elapsed_ms"] < ELAPSED_BUDGET_MS, (
        f"{body['elapsed_ms']} ms — §7: most likely the model is being reloaded per "
        f"request instead of once at startup. timings: {body['timings_ms']}"
    )


def test_the_model_is_loaded_once_not_per_request(client, golden_png):
    """The observable consequence of loading once: request two is not slower than one."""
    post(client, golden_png, stages="false")
    times = []
    for _ in range(3):
        started = time.perf_counter()
        post(client, golden_png, stages="false")
        times.append(time.perf_counter() - started)
    assert max(times) < 1.0


# --------------------------------------------------------------------------- #
# §3 — the startup gate
# --------------------------------------------------------------------------- #

@pytest.fixture
def restorable_config(repo_root):
    """Lets a test edit the real `config/preprocess.json` and always puts it back."""
    path = repo_root / "config" / "preprocess.json"
    original = path.read_bytes()
    try:
        yield path
    finally:
        path.write_bytes(original)
        api._current_preprocess_config(None)  # drop the mutated copy from the read cache


def test_startup_refuses_a_mutated_config(served_run, restorable_config):
    mutated = json.loads(restorable_config.read_text("utf-8"))
    mutated["canny"]["threshold1"] = 51
    restorable_config.write_text(json.dumps(mutated, indent=2), encoding="utf-8")

    with pytest.raises(api.StartupRefused) as caught:
        api.create_app(
            runs_root=served_run["runs_root"], classes_path=served_run["classes_path"]
        )

    message = str(caught.value)
    assert runs_module.preprocess_hash(served_run["config"]) in message  # the checkpoint's
    assert runs_module.preprocess_hash(mutated) in message  # the working copy's
    assert "REFUSING TO START" in message


def test_startup_refuses_a_renumbered_class_list(served_run, tmp_path):
    """The class index is frozen; a regenerated list means a different frequency per index."""
    renumbered = copy.deepcopy(served_run["classes"])
    renumbered[0] = {**renumbered[0], "freq_hz": renumbered[0]["freq_hz"] + 5}
    path = tmp_path / "classes.json"
    path.write_text(json.dumps({"classes": renumbered}), encoding="utf-8")

    with pytest.raises(api.StartupRefused, match="class list has changed"):
        api.create_app(runs_root=served_run["runs_root"], classes_path=path)


def test_startup_refuses_when_the_class_list_is_missing(served_run, tmp_path):
    with pytest.raises(api.StartupRefused, match="not found"):
        api.create_app(
            runs_root=served_run["runs_root"], classes_path=tmp_path / "absent.json"
        )


def test_startup_refuses_an_unfinished_run(served_run):
    (served_run["run_dir"] / "history.csv").rename(
        served_run["run_dir"] / "history.csv.bak"
    )
    try:
        with pytest.raises(api.StartupRefused, match="not a finished run"):
            api.create_app(
                runs_root=served_run["runs_root"], classes_path=served_run["classes_path"]
            )
    finally:
        (served_run["run_dir"] / "history.csv.bak").rename(
            served_run["run_dir"] / "history.csv"
        )


# --------------------------------------------------------------------------- #
# §3 — the rules that are not about pixels
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# §3 — serving the built frontend
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def app_with_frontend(served_run, tmp_path_factory):
    """The app with a stand-in `web/dist`. F6's real build is checked by `tasks.py app`."""
    dist = tmp_path_factory.mktemp("dist")
    (dist / "index.html").write_text("<!doctype html><div id=root></div>", encoding="utf-8")
    (dist / "assets").mkdir()
    (dist / "assets" / "index.js").write_text("console.log(1)", encoding="utf-8")
    return TestClient(
        api.create_app(
            runs_root=served_run["runs_root"],
            classes_path=served_run["classes_path"],
            web_dist=dist,
        )
    )


@pytest.mark.parametrize("path", ["/", "/analitza", "/dades", "/apren/qualsevol"])
def test_client_routes_survive_a_reload(app_with_frontend, path):
    """`BrowserRouter` puts real URLs in the address bar with no file behind them.

    Without the fallback these 404 — invisible while clicking, and immediate the moment
    someone reloads or is handed a link. Mid-session that is one participant's data lost.
    """
    response = app_with_frontend.get(path)
    assert response.status_code == 200
    assert "<div id=root>" in response.text


def test_a_missing_asset_still_404s(app_with_frontend):
    """A broken build must fail loudly, not serve HTML where a script was expected."""
    assert app_with_frontend.get("/assets/does-not-exist.js").status_code == 404
    assert app_with_frontend.get("/assets/index.js").status_code == 200


def test_the_api_is_not_shadowed_by_the_static_mount(app_with_frontend):
    assert app_with_frontend.get("/api/health").json()["model_run"]
    # An unknown /api path is an API error, never the SPA shell.
    assert app_with_frontend.get("/api/nope").status_code == 404
    assert "<div id=root>" not in app_with_frontend.get("/api/nope").text


def test_no_cors_wildcard(app):
    assert not any(
        "CORS" in type(m.cls).__name__ or "CORS" in getattr(m.cls, "__name__", "")
        for m in app.user_middleware
    )


def test_an_upload_is_never_written_to_disk(client, served_run, golden_png):
    before = sorted(p.name for p in served_run["run_dir"].parent.rglob("*"))
    post(client, golden_png)
    assert sorted(p.name for p in served_run["run_dir"].parent.rglob("*")) == before


def test_a_file_that_is_not_an_image_is_refused_not_crashed(client):
    response = post(client, b"this is not a PNG")
    assert response.status_code == 415


def test_an_empty_upload_is_refused(client):
    assert post(client, b"").status_code == 400
