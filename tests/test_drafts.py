"""F6 — the draft store. `docs/05-webapp-spec.md` §9.

A draft lives at `drafts/<session_id>/`, outside `data/`, and that is the whole design: "a
half-finished session leaves no trace in `data/`" is then true because nothing was ever written
there, not because something was cleaned up afterwards.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import drafts
from ingest import IngestError

SESSION = {
    "session_id": "20260830-01",
    "date": "2026-08-30",
    "operator": "PS",
    "plate": "alu-200",
    "material": "sand",
    "grain_mass_g": 6.0,
    "distance_mm": 210,
    "lighting": "lamp_diffuse",
    "dwell_s": 3.0,
    "notes": "",
}


@pytest.fixture
def drafts_root(tmp_path) -> Path:
    return tmp_path / "drafts"


def test_create_writes_the_session_log_and_nothing_else(drafts_root):
    draft = drafts.create(drafts_root, SESSION)

    assert draft.root == drafts_root / "20260830-01"
    assert json.loads((draft.root / "session.json").read_text("utf-8")) == SESSION
    assert draft.status == "new"


def test_create_refuses_an_incomplete_session_log_and_leaves_no_directory(drafts_root):
    incomplete = {k: v for k, v in SESSION.items() if k != "grain_mass_g"}

    with pytest.raises(IngestError, match="grain_mass_g"):
        drafts.create(drafts_root, incomplete)

    assert not (drafts_root / "20260830-01").exists()


def test_create_refuses_a_second_draft_for_the_same_session(drafts_root):
    drafts.create(drafts_root, SESSION)

    with pytest.raises(drafts.DraftConflict, match="already open"):
        drafts.create(drafts_root, SESSION)


def test_create_refuses_a_malformed_session_id(drafts_root):
    with pytest.raises(ValueError):
        drafts.create(drafts_root, {**SESSION, "session_id": "../escape"})


@pytest.mark.parametrize(
    "escape",
    ["../escape", "..\\escape", "20260830-01/../../etc", "C:/Windows", "/etc/passwd"],
)
def test_a_traversal_id_is_refused_by_the_path_check_itself(drafts_root, escape):
    """Requirement 7. `validate_session_id` refuses these today as a side effect of being a
    *format* rule, and that is the problem: the day someone wants a friendlier identifier they
    will relax a format validator and open a path hole without knowing one was closed there.
    `_within` is the boundary said out loud, so neither intention can be broken by touching
    the other."""
    with pytest.raises(drafts.DraftError):
        drafts._within(drafts_root, escape)


def test_within_accepts_what_it_should(drafts_root):
    assert drafts._within(drafts_root, "20260830-01") == (drafts_root / "20260830-01").resolve()


def test_listing_and_read_round_trip(drafts_root):
    drafts.create(drafts_root, SESSION)
    drafts.create(drafts_root, {**SESSION, "session_id": "20260830-02"})

    ids = [d.draft_id for d in drafts.listing(drafts_root)]

    assert ids == ["20260830-01", "20260830-02"]
    assert drafts.read(drafts_root, "20260830-02").session_id == "20260830-02"


def test_listing_is_empty_when_there_is_no_drafts_directory(drafts_root):
    assert drafts.listing(drafts_root) == []


def test_discard_removes_the_whole_draft(drafts_root):
    drafts.create(drafts_root, SESSION)

    drafts.discard(drafts_root, "20260830-01")

    assert not (drafts_root / "20260830-01").exists()
    assert drafts.listing(drafts_root) == []


# --------------------------------------------------------------------------- #
# the frames — the bytes arrive raw, and the same bytes arrive once
# --------------------------------------------------------------------------- #

def a_jpeg(seed: int = 3) -> bytes:
    """Real JPEG bytes, so the test is about a file format and not about a byte string."""
    import cv2
    import numpy as np

    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, size=(64, 48, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return buffer.tobytes()


def test_stored_bytes_are_the_uploaded_bytes(drafts_root):
    """THE test of this task. A browser upload that gets recoded survives Canny as spurious
    contours and cannot be detected by looking at the file afterwards."""
    draft = drafts.create(drafts_root, SESSION)
    data = a_jpeg()

    record = drafts.store_frame(draft, 440, "IMG_4711.JPG", data)

    stored = drafts.frame_path(draft, record).read_bytes()
    assert hashlib.sha256(stored).hexdigest() == hashlib.sha256(data).hexdigest()
    assert record["sha256"] == hashlib.sha256(data).hexdigest()


def test_stored_frames_land_where_ingest_looks_for_them(drafts_root):
    draft = drafts.create(drafts_root, SESSION)

    first = drafts.store_frame(draft, 440, "IMG_4711.JPG", a_jpeg(1))
    second = drafts.store_frame(draft, 440, "IMG_4712.JPG", a_jpeg(2))

    assert drafts.frame_path(draft, first) == draft.raw / "440" / "IMG_0001.jpg"
    assert drafts.frame_path(draft, second) == draft.raw / "440" / "IMG_0002.jpg"


def test_a_stored_frame_is_pending_until_it_is_gated(drafts_root):
    """A frame whose bytes are on disk but whose verdict never arrived must be visible as
    something, not as nothing."""
    draft = drafts.create(drafts_root, SESSION)

    record = drafts.store_frame(draft, 440, "IMG_4711.JPG", a_jpeg())

    assert record["quality"] == "pending"
    reread = drafts.frame_records(drafts.read(drafts_root, "20260830-01"), 440)
    assert reread[0]["quality"] == "pending"


def test_the_same_file_dropped_twice_is_refused(drafts_root):
    draft = drafts.create(drafts_root, SESSION)
    drafts.store_frame(draft, 440, "IMG_4711.JPG", a_jpeg())

    with pytest.raises(drafts.DuplicateName, match="already"):
        drafts.store_frame(draft, 440, "IMG_4711.JPG", a_jpeg())


def test_the_same_bytes_under_a_new_name_are_a_duplicate(drafts_root):
    """Requirement 8. Six hundred uploads over a connection that stumbles means someone
    retries a batch; a repeated frame trips no threshold and quietly unbalances the classes,
    and with bursts already near-identical to each other, nobody catches it by looking."""
    draft = drafts.create(drafts_root, SESSION)
    data = a_jpeg()
    drafts.store_frame(draft, 440, "IMG_4711.JPG", data)

    with pytest.raises(drafts.DuplicateContent, match="same bytes"):
        drafts.store_frame(draft, 620, "IMG_9999.JPG", data)

    assert len(drafts.frame_records(draft)) == 1
    assert not (draft.raw / "620").exists()
    duplicates = drafts.duplicate_records(drafts.read(drafts_root, "20260830-01"))
    assert [d["original_filename"] for d in duplicates] == ["IMG_9999.JPG"]
    assert duplicates[0]["matches"] == "IMG_0001.jpg"


def test_an_unsupported_suffix_never_reaches_the_draft(drafts_root):
    draft = drafts.create(drafts_root, SESSION)

    with pytest.raises(drafts.DraftError, match="suffix"):
        drafts.store_frame(draft, 440, "notes.txt", b"hello")

    assert not (draft.raw / "440").exists()


def test_records_survive_a_reread(drafts_root):
    draft = drafts.create(drafts_root, SESSION)
    record = drafts.store_frame(draft, 440, "IMG_4711.JPG", a_jpeg())
    drafts.record_verdict(draft, record, "blurred", "blurred", {"laplacian_variance": 4.0})

    reread = drafts.read(drafts_root, "20260830-01")
    records = drafts.frame_records(reread, 440)

    assert [r["name"] for r in records] == ["IMG_0001.jpg"]
    assert records[0]["reason"] == "blurred"
    assert records[0]["original_filename"] == "IMG_4711.JPG"
    assert records[0]["metrics"] == {"laplacian_variance": 4.0}


def test_a_verdict_for_a_frame_that_is_not_there_is_an_error(drafts_root):
    draft = drafts.create(drafts_root, SESSION)

    with pytest.raises(drafts.DraftError, match="no frame"):
        drafts.record_verdict(draft, {"freq_hz": 440, "name": "IMG_0001.jpg"}, "ok", None, {})


def test_concurrent_stores_do_not_lose_each_other(drafts_root):
    """The frames log is a read-modify-write and the upload route runs in a thread pool.
    Without the lock this loses rows, and it loses them silently."""
    from concurrent.futures import ThreadPoolExecutor

    draft = drafts.create(drafts_root, SESSION)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(
            pool.map(
                lambda i: drafts.store_frame(draft, 440, f"IMG_{i}.JPG", a_jpeg(i)), range(24)
            )
        )

    records = drafts.frame_records(drafts.read(drafts_root, "20260830-01"), 440)
    assert len(records) == 24
    assert len({r["name"] for r in records}) == 24
    assert len({r["sha256"] for r in records}) == 24


def test_remove_frame_takes_the_record_and_the_previews_too(drafts_root):
    import numpy as np

    draft = drafts.create(drafts_root, SESSION)
    record = drafts.store_frame(draft, 440, "IMG_4711.JPG", a_jpeg())
    path = drafts.frame_path(draft, record)
    drafts.write_previews(
        draft, path, 440, np.zeros((40, 40, 4), np.uint8), np.zeros((40, 40), np.uint8)
    )
    drafts.record_verdict(draft, record, "blurred", "blurred", {})

    drafts.remove_frame(draft, 440, "IMG_0001.jpg")

    assert drafts.frame_records(draft, 440) == []
    assert not path.exists()
    assert not drafts.preview_path(draft, 440, "IMG_0001.jpg", "edges").exists()


def test_a_removal_does_not_make_the_next_frame_collide(drafts_root):
    """The stored names are the image_ids ingest will derive, so a hole in the numbering is
    left alone — renumbering would change the identity of frames already gated."""
    draft = drafts.create(drafts_root, SESSION)
    drafts.store_frame(draft, 440, "A.JPG", a_jpeg(1))
    drafts.store_frame(draft, 440, "B.JPG", a_jpeg(2))
    drafts.remove_frame(draft, 440, "IMG_0001.jpg")

    record = drafts.store_frame(draft, 440, "C.JPG", a_jpeg(3))

    assert record["name"] == "IMG_0003.jpg"
    assert {r["name"] for r in drafts.frame_records(draft, 440)} == {
        "IMG_0002.jpg",
        "IMG_0003.jpg",
    }


def test_previews_are_written_for_a_partial_frame_without_an_edge_map(drafts_root):
    """Without four markers there is no rectified plate, so there is no edge map that means
    anything. Showing one anyway would be showing a rectification that did not happen."""
    import numpy as np

    draft = drafts.create(drafts_root, SESSION)
    record = drafts.store_frame(draft, 440, "IMG_4711.JPG", a_jpeg())
    path = drafts.frame_path(draft, record)

    drafts.write_previews(draft, path, 440, np.zeros((40, 40, 4), np.uint8), None)

    assert drafts.preview_path(draft, 440, record["name"], "original").is_file()
    assert not drafts.preview_path(draft, 440, record["name"], "edges").exists()


def test_size_on_disk_counts_the_captures(drafts_root):
    draft = drafts.create(drafts_root, SESSION)
    data = a_jpeg()
    drafts.store_frame(draft, 440, "IMG_4711.JPG", data)

    assert drafts.size_on_disk(draft) >= len(data)
