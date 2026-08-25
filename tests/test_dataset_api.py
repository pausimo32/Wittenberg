"""F6 — `/dades`'s API. `docs/05-webapp-spec.md` §9.

Every test here builds its own `data/` and `drafts/` under `tmp_path` and points the app at
them. Nothing in this file reads or writes the repository's real dataset.

The app is also given a throwaway `web/dist` rather than the repository's build, for one
reason: `test_an_api_path_with_no_route_is_still_a_404` is only a test at all while the SPA
fallback is actually mounted, and `web/dist` is a git-ignored build artefact that may or may
not be on the machine running the suite. A two-line `index.html` under `tmp_path` makes the
mount unconditional, so the assertion cannot pass by the fallback simply being absent.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from test_drafts import HEADER, SESSION, a_jpeg, dir_fingerprint, marker_board

import api
import drafts


@pytest.fixture
def roots(tmp_path):
    data = tmp_path / "data"
    (data / "raw").mkdir(parents=True)
    (data / "metadata.csv").write_text(HEADER, "utf-8")
    return data, tmp_path / "drafts"


def make_app(tmp_path, name, config, data_root, drafts_root):
    """The real app, pointed at a throwaway dataset. Reuses `test_api.py::build_run`."""
    from conftest_f4a import class_list
    from test_api import build_run

    built = build_run(tmp_path / name, config, class_list(), {"do_not_report": None})

    dist = tmp_path / name / "dist"
    dist.mkdir(parents=True, exist_ok=True)
    (dist / "index.html").write_text("<!doctype html><title>app</title>", "utf-8")

    return api.create_app(
        argv=[],
        runs_root=built["runs_root"],
        classes_path=built["classes_path"],
        web_dist=dist,
        data_root=data_root,
        drafts_root=drafts_root,
    )


@pytest.fixture
def client(roots, tmp_path, config):
    data_root, drafts_root = roots
    with TestClient(make_app(tmp_path, "runs", config, data_root, drafts_root)) as c:
        yield c


def test_an_api_path_with_no_route_is_still_a_404(client):
    """The SPA fallback must not answer for /api. This is the route the trap was found on."""
    response = client.get("/api/drafts/20260830-01/nothing")

    assert response.status_code == 404
    assert not response.text.lstrip().lower().startswith("<!doctype")


def test_the_spa_fallback_still_answers_a_view_route(client):
    """The other half of the same rule: `/dades` is a real URL with no file behind it."""
    response = client.get("/dades")

    assert response.status_code == 200
    assert response.text.lstrip().lower().startswith("<!doctype")


def test_create_list_and_read_a_draft(client, roots):
    _data_root, drafts_root = roots

    created = client.post("/api/drafts", json=SESSION)
    assert created.status_code == 200, created.text
    assert created.json()["draft_id"] == "20260830-01"
    assert created.json()["path"] == str(drafts_root / "20260830-01")

    listed = client.get("/api/drafts").json()["drafts"]
    assert [d["draft_id"] for d in listed] == ["20260830-01"]

    read = client.get("/api/drafts/20260830-01").json()
    assert read["session"] == SESSION
    assert read["calibrated"] is False
    assert read["calibration"] is None
    assert read["frames"] == {}
    assert read["existing_rows"] == 0


def test_a_draft_that_is_not_there_is_a_404_with_a_code(client):
    response = client.get("/api/drafts/20260830-99")

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "no_draft"


def test_the_listing_carries_the_size_on_disk(client):
    """Requirement 9. Three open drafts are a few gigabytes; the one you can delete is the one
    you can tell apart."""
    client.post("/api/drafts", json=SESSION)

    listed = client.get("/api/drafts").json()["drafts"][0]

    assert listed["bytes"] > 0
    assert listed["duplicates"] == 0
    assert listed["pending"] == 0


def test_the_summary_counts_pending_duplicates_and_rejections_apart(client, roots):
    """A duplicate is not a rejection: the gate never saw it. Folding the two together would
    make a batch someone retried read as a lighting problem."""
    _data_root, drafts_root = roots
    client.post("/api/drafts", json=SESSION)
    draft = drafts.read(drafts_root, "20260830-01")

    accepted = drafts.store_frame(draft, 440, "IMG_0001.JPG", a_jpeg(1))
    drafts.record_verdict(draft, accepted, "ok", None, {})
    rejected = drafts.store_frame(draft, 440, "IMG_0002.JPG", a_jpeg(2))
    drafts.record_verdict(draft, rejected, "blurred", "blurred", {})
    drafts.store_frame(draft, 620, "IMG_0003.JPG", a_jpeg(3))  # never gated
    with pytest.raises(drafts.DraftDuplicate):
        drafts.store_frame(draft, 620, "AGAIN.JPG", a_jpeg(1))

    listed = client.get("/api/drafts").json()["drafts"][0]

    assert listed["total"] == 3
    assert listed["accepted"] == 1
    assert listed["rejected"] == 1
    assert listed["pending"] == 1
    assert listed["duplicates"] == 1
    assert listed["classes"] == [440, 620]


def test_draft_survives_a_new_client(client, roots, tmp_path, config):
    """Requirement 2. A reload is a new client; the draft is on the server, so it is found."""
    client.post("/api/drafts", json=SESSION)
    data_root, drafts_root = roots

    fresh = make_app(tmp_path, "runs2", config, data_root, drafts_root)
    with TestClient(fresh) as other:
        ids = [d["draft_id"] for d in other.get("/api/drafts").json()["drafts"]]

    assert ids == ["20260830-01"]


def test_an_incomplete_session_log_is_a_422_with_a_code(client):
    response = client.post(
        "/api/drafts", json={k: v for k, v in SESSION.items() if k != "operator"}
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_session_log"
    assert "operator" in response.json()["detail"]["message"]


def test_a_malformed_session_id_is_a_422_and_writes_nothing(client, roots):
    _data_root, drafts_root = roots

    response = client.post("/api/drafts", json={**SESSION, "session_id": "../escape"})

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_session_log"
    assert not drafts_root.exists() or list(drafts_root.iterdir()) == []


def test_a_second_draft_for_the_same_session_is_refused(client):
    client.post("/api/drafts", json=SESSION)

    response = client.post("/api/drafts", json=SESSION)

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "draft_exists"


def test_a_synthetic_session_alongside_a_real_one_is_refused(client, roots):
    """`docs/05` §9, acceptance test 4 — at the door, before anything is uploaded."""
    data_root, drafts_root = roots
    (data_root / "metadata.csv").write_text(
        HEADER
        + "20260101-01_440_0001,20260101-01,440,train,sand,6.0,210,lamp_diffuse,3.0,"
        "homography,ok,\n",
        "utf-8",
    )

    response = client.post(
        "/api/drafts",
        json={**SESSION, "session_id": "SYNTH-20260830-01", "synthetic": True},
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "mixed_provenance"
    # Refused at the door: the check runs before anything is created, so a session that can
    # never be committed never costs an upload — nor leaves a directory to clean up.
    assert not (drafts_root / "SYNTH-20260830-01").exists()


def test_discard_removes_the_draft_and_leaves_data_alone(client, roots):
    data_root, _drafts_root = roots
    before = dir_fingerprint(data_root)
    client.post("/api/drafts", json=SESSION)

    assert client.delete("/api/drafts/20260830-01").status_code == 204

    assert client.get("/api/drafts").json()["drafts"] == []
    assert dir_fingerprint(data_root) == before


def test_discarding_a_draft_that_is_not_there_is_a_404(client):
    assert client.delete("/api/drafts/20260830-99").status_code == 404


def test_the_serving_half_is_untouched(client):
    """The router is mounted on the same app; §3's routes still answer, and first."""
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["model"] == "wittenberg_v1"


def a_png(image) -> bytes:
    import cv2

    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return buffer.tobytes()


def upload(client, url, data: bytes, name: str = "IMG_4711.jpg", kind: str = "image/jpeg"):
    return client.post(url, files={"file": (name, data, kind)})


def test_calibration_returns_the_quad_and_a_preview(client, roots):
    from calibrate import load_calibration_config

    data_root, _drafts_root = roots
    before = dir_fingerprint(data_root)
    client.post("/api/drafts", json=SESSION)
    board = marker_board(load_calibration_config())

    response = upload(
        client, "/api/drafts/20260830-01/calibration",
        a_png(board[..., :3]), "calib.png", "image/png",
    )

    assert response.status_code == 200, response.text
    assert response.json()["calibration"]["ok"] is True
    assert response.json()["calibration"]["n_markers"] == 4
    assert response.json()["preview"]
    assert client.get("/api/drafts/20260830-01").json()["calibrated"] is True
    # `data/calib/` gets its copy at commit; a draft that is abandoned here leaves nothing.
    assert dir_fingerprint(data_root) == before


def test_calibration_without_markers_reports_how_many_were_found(client):
    import numpy as np

    client.post("/api/drafts", json=SESSION)

    response = upload(
        client, "/api/drafts/20260830-01/calibration",
        a_png(np.full((400, 400, 3), 255, np.uint8)), "blank.png", "image/png",
    )

    assert response.status_code == 200
    assert response.json()["calibration"]["ok"] is False
    assert response.json()["calibration"]["n_markers"] == 0
    # The picture comes back even though the detection failed: "no markers found" is what the
    # person at the rig has to act on, and they act on it by looking at the frame.
    assert response.json()["preview"]
    assert client.get("/api/drafts/20260830-01").json()["calibrated"] is False


def test_the_quad_is_drawn_where_the_markers_are(client):
    """The preview is the only evidence anyone has that the calibration is right.

    A capture is larger than the preview, so the quad has to be scaled by exactly the factor
    the thumbnail used. Asserting that a PNG came back would pass with the outline drawn
    somewhere the markers are not — at full-resolution coordinates, for one, which on this
    frame lands almost entirely outside the picture.
    """
    import base64

    import cv2
    import numpy as np

    from calibrate import load_calibration_config
    from dataset_api import CALIBRATION_PREVIEW_MAX_SIDE

    client.post("/api/drafts", json=SESSION)
    # Bigger than the preview, so `thumbnail` really resizes and the scale is under test.
    side = CALIBRATION_PREVIEW_MAX_SIDE * 3
    board = cv2.resize(
        marker_board(load_calibration_config())[..., :3], (side, side),
        interpolation=cv2.INTER_NEAREST,
    )

    body = upload(
        client, "/api/drafts/20260830-01/calibration", a_png(board), "calib.png", "image/png"
    ).json()

    preview = cv2.imdecode(
        np.frombuffer(base64.b64decode(body["preview"]), np.uint8), cv2.IMREAD_COLOR
    )
    assert max(preview.shape[:2]) == CALIBRATION_PREVIEW_MAX_SIDE
    scale = preview.shape[1] / side
    expected = np.array(body["calibration"]["corners"], np.float64) * scale

    ys, xs = np.nonzero(np.all(preview == (0, 190, 0), axis=2))
    assert xs.size, "no quad was drawn on the preview"
    # The outline is 2 px thick, so its extent may exceed the corners by a pixel either way.
    assert abs(xs.min() - expected[:, 0].min()) <= 2
    assert abs(xs.max() - expected[:, 0].max()) <= 2
    assert abs(ys.min() - expected[:, 1].min()) <= 2
    assert abs(ys.max() - expected[:, 1].max()) <= 2
