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

from itertools import pairwise

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


# --------------------------------------------------------------------------- #
# the frames — the order, the live gate, the thumbnails
# --------------------------------------------------------------------------- #

def calibrated_draft(client):
    from calibrate import load_calibration_config

    client.post("/api/drafts", json=SESSION)
    upload(
        client,
        "/api/drafts/20260830-01/calibration",
        a_png(marker_board(load_calibration_config())[..., :3]),
        "calib.png",
        "image/png",
    )


def test_frames_refused_before_calibration(client, roots):
    """Requirement 3, enforced by the server. The ROI the gate measures does not exist yet, and
    a rule that lives only in a component is one component away from not being a rule."""
    _data_root, drafts_root = roots
    client.post("/api/drafts", json=SESSION)

    response = upload(client, "/api/drafts/20260830-01/frames?freq_hz=440", a_jpeg())

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "not_calibrated"
    # And the bytes were not kept either: refusing after storing would leave a frame nobody
    # asked for in a draft whose owner was told no.
    assert not (drafts_root / "20260830-01" / "raw").exists()


def test_a_frame_is_gated_as_it_arrives(client):
    calibrated_draft(client)

    response = upload(client, "/api/drafts/20260830-01/frames?freq_hz=440", a_jpeg())

    assert response.status_code == 200, response.text
    record = response.json()
    assert record["name"] == "IMG_0001.jpg"
    assert record["original_filename"] == "IMG_4711.jpg"
    assert record["quality"] in {"ok", "blurred", "partial", "reject"}
    assert record["quality"] != "pending"
    assert (record["reason"] is None) == (record["quality"] == "ok")


def test_the_live_gate_is_the_ingest_gate(client, roots):
    """`docs/05` §9, acceptance test 1 — identical verdicts, asserted rather than assumed.

    Not "the same thresholds as": the route calls `ingest.assess_frame`, so a discrepancy would
    have to be a discrepancy inside one function.
    """
    from calibrate import read_calibration
    from ingest import gate_frame
    from preprocess import load_config
    from quality import load_gate

    _data_root, drafts_root = roots
    calibrated_draft(client)
    for i in range(4):
        upload(client, "/api/drafts/20260830-01/frames?freq_hz=440", a_jpeg(i), f"IMG_{i}.jpg")

    draft = drafts.read(drafts_root, "20260830-01")
    calib = read_calibration(draft.root / "calibration.json")
    records = drafts.frame_records(draft)
    assert len(records) == 4
    for record in records:
        path = drafts.frame_path(draft, record)
        quality, reason, _metrics = gate_frame(
            path, calib, calib["roi_mode"], load_config(), load_gate()
        )
        assert (quality, reason) == (record["quality"], record["reason"])


def test_a_rejected_frame_carries_a_thumbnail(client):
    calibrated_draft(client)

    record = upload(client, "/api/drafts/20260830-01/frames?freq_hz=440", a_jpeg()).json()

    if record["reason"] is None:
        pytest.skip("this synthetic frame passed the gate; the thumbnail path needs a reject")
    assert record["preview"] is True
    image = client.get(f"/api/drafts/20260830-01/preview/440/{record['name']}?kind=original")
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"


def test_a_partial_frame_gets_no_edge_preview(client, roots):
    """Without four markers there is no rectified plate, so there is no edge map that means
    anything — and offering one would be showing a rectification that did not happen."""
    calibrated_draft(client)

    # A frame with no markers at all: the session is calibrated, this capture is not usable.
    import numpy as np

    blank = a_png(np.full((400, 400, 3), 128, np.uint8))
    record = upload(
        client, "/api/drafts/20260830-01/frames?freq_hz=440", blank, "blank.png", "image/png"
    ).json()

    assert record["reason"] == "partial"
    assert record["preview"] is True
    assert (
        client.get(
            f"/api/drafts/20260830-01/preview/440/{record['name']}?kind=original"
        ).status_code
        == 200
    )
    assert (
        client.get(f"/api/drafts/20260830-01/preview/440/{record['name']}?kind=edges").status_code
        == 404
    )


def test_the_same_folder_dropped_twice_is_refused(client):
    calibrated_draft(client)
    upload(client, "/api/drafts/20260830-01/frames?freq_hz=440", a_jpeg())

    again = upload(client, "/api/drafts/20260830-01/frames?freq_hz=440", a_jpeg())

    assert again.status_code == 409
    assert again.json()["detail"]["code"] == "duplicate_frame"


def test_the_same_capture_under_a_new_name_is_its_own_category(client):
    """Requirement 8, at the route. Not `rejected`: the gate never saw it, and a retried batch
    must not read as a lighting problem."""
    calibrated_draft(client)
    data = a_jpeg()
    upload(client, "/api/drafts/20260830-01/frames?freq_hz=440", data, "IMG_1.jpg")

    again = upload(client, "/api/drafts/20260830-01/frames?freq_hz=620", data, "IMG_2.jpg")

    assert again.status_code == 409
    assert again.json()["detail"]["code"] == "duplicate_content"
    summary = client.get("/api/drafts/20260830-01").json()
    assert summary["duplicates"] == 1
    assert summary["total"] == 1
    assert summary["accepted"] + summary["rejected"] == 1


def test_a_text_file_is_refused_before_it_is_stored(client, roots):
    calibrated_draft(client)
    _data_root, drafts_root = roots

    response = client.post(
        "/api/drafts/20260830-01/frames?freq_hz=440",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 415
    assert response.json()["detail"]["code"] == "unsupported_image"
    assert not (drafts_root / "20260830-01" / "raw" / "440").exists()


def test_a_frame_can_be_taken_back_out(client):
    calibrated_draft(client)
    record = upload(client, "/api/drafts/20260830-01/frames?freq_hz=440", a_jpeg()).json()

    dropped = client.delete(f"/api/drafts/20260830-01/frames/440/{record['name']}")

    assert dropped.status_code == 204
    assert client.get("/api/drafts/20260830-01").json()["total"] == 0
    assert (
        client.get(
            f"/api/drafts/20260830-01/preview/440/{record['name']}?kind=original"
        ).status_code
        == 404
    )


def test_a_preview_name_cannot_escape_the_draft(client):
    """The one path segment that is not a validated session id."""
    calibrated_draft(client)

    response = client.get("/api/drafts/20260830-01/preview/440/..%2F..%2Fdraft.json")

    assert response.status_code == 404


def test_an_unknown_preview_kind_is_a_404(client):
    calibrated_draft(client)

    assert (
        client.get("/api/drafts/20260830-01/preview/440/IMG_0001.jpg?kind=secret").status_code
        == 404
    )


def test_uploading_a_frame_writes_nothing_under_data(client, roots):
    """`docs/05` §9: uploading is not committing."""
    data_root, _drafts_root = roots
    calibrated_draft(client)
    before = dir_fingerprint(data_root)

    upload(client, "/api/drafts/20260830-01/frames?freq_hz=440", a_jpeg())

    assert dir_fingerprint(data_root) == before


# --------------------------------------------------------------------------- #
# the commit — stage, ingest, and only then a session
# --------------------------------------------------------------------------- #

def fill(client, classes=(440, 620), per_class=3):
    calibrated_draft(client)
    n = 0
    for freq in classes:
        for _ in range(per_class):
            upload(
                client,
                f"/api/drafts/20260830-01/frames?freq_hz={freq}",
                a_jpeg(n),
                f"IMG_{n}.jpg",
            )
            n += 1
    return n


def wait_for_commit(client, draft_id="20260830-01", limit=4000):
    for _ in range(limit):
        status = client.get(f"/api/drafts/{draft_id}/commit").json()
        if status["status"] in {"done", "failed"}:
            return status
    raise AssertionError("the commit never finished")


def test_commit_ingests_the_session_and_clears_the_draft(client, roots):
    data_root, drafts_root = roots
    total = fill(client)

    started = client.post("/api/drafts/20260830-01/commit?jobs=1")
    assert started.status_code == 202, started.text
    status = wait_for_commit(client)

    assert status["status"] == "done", status
    assert status["report"]["total"] == total
    assert (data_root / "raw" / "20260830-01" / "session.json").is_file()
    assert (data_root / "calib" / "20260830-01.json").is_file()
    assert (data_root / "metadata.csv").is_file()
    assert not (drafts_root / "20260830-01").exists()
    assert not (data_root / "raw" / ".incoming-20260830-01").exists()


def test_the_committed_rows_are_the_ones_ingest_would_have_written(client, roots):
    """The commit calls `ingest_session`; this is what says so. Every row carries the session
    id, the class it was labelled with, and a `quality` from the §6 vocabulary."""
    from metadata import QUALITY_VALUES, read_metadata

    data_root, _drafts_root = roots
    total = fill(client)
    client.post("/api/drafts/20260830-01/commit?jobs=1")
    wait_for_commit(client)

    rows = read_metadata(data_root / "metadata.csv")

    assert len(rows) == total
    assert {r["session_id"] for r in rows} == {"20260830-01"}
    assert {int(r["freq_hz"]) for r in rows} == {440, 620}
    assert all(r["quality"] in QUALITY_VALUES for r in rows)
    assert all(r["material"] == "sand" for r in rows)     # the session log became the columns


def test_a_failed_commit_leaves_no_session_and_can_be_resumed(client, roots, monkeypatch):
    """Requirement 6, end to end. A disk that fills up or one corrupt capture must not leave a
    half-moved session that nobody can name."""
    import dataset_api

    data_root, drafts_root = roots
    total = fill(client)

    real = dataset_api.ingest_session

    def explode(*_args, **_kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(dataset_api, "ingest_session", explode)
    client.post("/api/drafts/20260830-01/commit?jobs=1")
    failed = wait_for_commit(client)

    assert failed["status"] == "failed"
    assert failed["error"]["type"] == "OSError"
    assert not (data_root / "raw" / "20260830-01").exists()
    assert not (data_root / "calib" / "20260830-01.json").exists()
    assert (data_root / "raw" / ".incoming-20260830-01").is_dir()
    assert (drafts_root / "20260830-01").is_dir()

    monkeypatch.setattr(dataset_api, "ingest_session", real)
    client.post("/api/drafts/20260830-01/commit?jobs=1")
    done = wait_for_commit(client)

    assert done["status"] == "done", done
    assert done["report"]["total"] == total
    assert (data_root / "raw" / "20260830-01" / "session.json").is_file()
    assert not (data_root / "raw" / ".incoming-20260830-01").exists()
    assert not (drafts_root / "20260830-01").exists()


def test_an_interrupted_commit_is_reported_as_interrupted(client, roots, tmp_path, config, monkeypatch):
    """The server restarted mid-commit: no job in this process, a draft stuck in `committing`.
    Saying "idle" there would invite someone to start again over a staged directory."""
    import dataset_api

    data_root, drafts_root = roots
    fill(client)
    monkeypatch.setattr(
        dataset_api,
        "ingest_session",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("boom")),
    )
    client.post("/api/drafts/20260830-01/commit?jobs=1")
    wait_for_commit(client)

    fresh = make_app(tmp_path, "restarted", config, data_root, drafts_root)
    with TestClient(fresh) as other:
        status = other.get("/api/drafts/20260830-01/commit").json()

    assert status["status"] == "interrupted"
    assert ".incoming-20260830-01" in status["error"]["message"]


def test_discarding_a_failed_commit_takes_the_staged_captures(client, roots, monkeypatch):
    import dataset_api

    data_root, _drafts_root = roots
    fill(client)
    monkeypatch.setattr(
        dataset_api,
        "ingest_session",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("boom")),
    )
    client.post("/api/drafts/20260830-01/commit?jobs=1")
    wait_for_commit(client)

    assert client.delete("/api/drafts/20260830-01").status_code == 204

    assert not (data_root / "raw" / ".incoming-20260830-01").exists()
    assert list((data_root / "raw").iterdir()) == []


def test_commit_refuses_an_existing_session_without_replace(client):
    """Requirement 4. Idempotent means repeating does not duplicate, not that repeating was
    what anyone wanted."""
    fill(client)
    client.post("/api/drafts/20260830-01/commit?jobs=1")
    wait_for_commit(client)
    fill(client)   # the same session id, a second draft

    refused = client.post("/api/drafts/20260830-01/commit?jobs=1")

    assert refused.status_code == 409
    assert refused.json()["detail"]["code"] == "session_exists"
    assert refused.json()["detail"]["existing_rows"] > 0


def test_a_refused_commit_does_not_touch_the_session_it_refused_to_replace(client, roots):
    """The refusal happens before anything moves, so the afternoon that is already in the
    dataset is exactly as it was."""
    data_root, drafts_root = roots
    fill(client)
    client.post("/api/drafts/20260830-01/commit?jobs=1")
    wait_for_commit(client)
    before = dir_fingerprint(data_root)

    fill(client)
    client.post("/api/drafts/20260830-01/commit?jobs=1")

    assert dir_fingerprint(data_root) == before
    assert (drafts_root / "20260830-01" / "raw").is_dir()   # its captures are still its own


def test_re_ingesting_a_session_twice_produces_one_set_of_rows(client, roots):
    """`docs/05` §9, acceptance test 3."""
    from metadata import read_metadata

    data_root, _drafts_root = roots
    total = fill(client)
    client.post("/api/drafts/20260830-01/commit?jobs=1")
    wait_for_commit(client)
    fill(client)
    client.post("/api/drafts/20260830-01/commit?replace=true&jobs=1")
    wait_for_commit(client)

    rows = read_metadata(data_root / "metadata.csv")

    assert len(rows) == total
    assert len({r["image_id"] for r in rows}) == total


def test_commit_reports_progress(client):
    """Requirement 5: six hundred files are not six, so the number has to move."""
    total = fill(client, per_class=4)
    client.post("/api/drafts/20260830-01/commit?jobs=1")

    seen = []
    for _ in range(4000):
        status = client.get("/api/drafts/20260830-01/commit").json()
        seen.append(status["done"])
        if status["status"] in {"done", "failed"}:
            break

    assert seen[-1] == total
    assert all(a <= b for a, b in pairwise(seen))


def test_commit_refuses_a_draft_with_no_frames(client):
    calibrated_draft(client)

    response = client.post("/api/drafts/20260830-01/commit")

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "no_frames"


def test_commit_refuses_an_uncalibrated_draft(client):
    client.post("/api/drafts", json=SESSION)

    response = client.post("/api/drafts/20260830-01/commit")

    assert response.status_code == 409
    assert response.json()["detail"]["code"] in {"no_frames", "not_calibrated"}


def test_a_second_commit_while_one_is_running_is_refused(client):
    fill(client)
    client.post("/api/drafts/20260830-01/commit?jobs=1")

    second = client.post("/api/drafts/20260830-01/commit?jobs=1")

    # Either the first is still running (409) or it finished between the two calls, in which
    # case the draft is gone and this is a 404. Both are refusals; neither starts a second job.
    assert second.status_code in (404, 409)
    if second.status_code == 409:
        assert second.json()["detail"]["code"] == "commit_running"
    wait_for_commit(client)


def test_commit_status_is_idle_for_a_draft_nobody_has_committed(client):
    calibrated_draft(client)

    status = client.get("/api/drafts/20260830-01/commit").json()

    assert status["status"] == "idle"
    assert status["report"] is None


def test_split_runs_by_session(client, roots):
    fill(client)
    client.post("/api/drafts/20260830-01/commit?jobs=1")
    wait_for_commit(client)

    response = client.post("/api/dataset/split")

    # One session cannot be partitioned three ways; the refusal is the correct answer and it
    # has to be legible rather than a 500.
    assert response.status_code in (200, 409)
    if response.status_code == 409:
        assert response.json()["detail"]["code"] == "split_failed"
    else:
        assert response.json()["seed"] == 42
