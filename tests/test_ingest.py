"""Contract tests for `src/ingest.py` and `src/metadata.py` — docs/02 §4 and §6.

Sessions here are built small (two classes, a handful of frames) with the same renderer
`tools/synth_capture.py` uses, so what the tests ingest is what the tool produces.

Four properties carry the weight:

* **Idempotence.** Re-ingesting a session replaces its rows and never duplicates them.
  A duplicated session would inflate one condition's weight in training and nothing
  downstream would notice.
* **No mixed provenance.** Synthetic sessions and real ones cannot share a
  `metadata.csv`. A synthetic frame that reached a training set would put a number in
  the TDR that describes a renderer rather than a plate.
* **Rejects are counted, never deleted.** They stay in `data/raw/`.
* **No silent `center_crop`.** A homography session with a three-marker frame rejects
  that frame; it does not quietly re-crop it.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from synth_capture import render_frame

from calibrate import calibrate, write_calibration
from ingest import IngestError, MixedProvenanceError, ingest_session
from metadata import (
    METADATA_COLUMNS,
    QUALITY_VALUES,
    SPLIT_NONE,
    SPLIT_UNASSIGNED,
    MetadataError,
    is_synthetic,
    read_metadata,
    validate_session_id,
)

SHAPES = {420: (1, 2), 610: (1, 3)}

CONDITIONS = {
    "material": "sand",
    "grain_mass_g": 2.0,
    "distance_mm": 210,
    "lighting": "lamp_diffuse",
    "dwell_s": 3.0,
}


# --------------------------------------------------------------------------- #
# a session on disk
# --------------------------------------------------------------------------- #

def build_session(
    root: Path,
    session_id: str,
    frames: int = 3,
    bad_frame_markers: dict[int, int] | None = None,
    with_calib: bool = True,
    conditions: dict | None = None,
) -> Path:
    """Write a miniature but structurally complete raw session.

    `bad_frame_markers` maps a frame index to how many markers it should carry, so a
    test can plant a three-marker frame exactly where it wants one.
    """
    rng = np.random.default_rng(7)
    bad = bad_frame_markers or {}
    session_dir = root / "raw" / session_id
    good_frame = None

    for freq, (n, m) in SHAPES.items():
        d = session_dir / str(freq)
        d.mkdir(parents=True, exist_ok=True)
        for k in range(frames):
            frame = render_frame(n, m, rng, "lamp_diffuse", n_markers=bad.get(k, 4))
            if good_frame is None and bad.get(k, 4) == 4:
                good_frame = frame
            cv2.imwrite(str(d / f"IMG_{k:04d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA))

    (session_dir / "session.json").write_text(
        json.dumps({"session_id": session_id, **(conditions or CONDITIONS)}), encoding="utf-8"
    )
    if with_calib:
        write_calibration(calibrate(good_frame, session_id=session_id), root / "calib")
    return session_dir


@pytest.fixture
def data_root(tmp_path) -> Path:
    return tmp_path / "data"


def rows_of(data_root: Path) -> list[dict]:
    return read_metadata(data_root / "metadata.csv")


# --------------------------------------------------------------------------- #
# the metadata file
# --------------------------------------------------------------------------- #

def test_ingest_writes_one_row_per_raw_image(data_root):
    """§4, literally: one row per raw image — rejected frames included."""
    build_session(data_root, "SYNTH-01", frames=3, bad_frame_markers={1: 3})
    report = ingest_session("SYNTH-01", data_root)

    rows = rows_of(data_root)
    assert report.total == 6                     # 2 classes x 3 frames
    assert len(rows) == report.total
    assert len(rows) > report.accepted           # some were rejected and still have a row


def test_rejected_frames_have_a_row(data_root):
    build_session(data_root, "SYNTH-01", frames=3, bad_frame_markers={1: 3})
    report = ingest_session("SYNTH-01", data_root)

    rejected = [r for r in rows_of(data_root) if r["quality"] != "ok"]
    assert len(rejected) == report.rejected
    assert rejected, "a rejected frame must still be recorded in the committed file"


def test_quality_column_uses_the_spec_vocabulary(data_root):
    build_session(data_root, "SYNTH-01", frames=3, bad_frame_markers={1: 3})
    ingest_session("SYNTH-01", data_root)
    seen = {r["quality"] for r in rows_of(data_root)}
    assert seen <= set(QUALITY_VALUES)
    assert len(seen) > 1, "the column is pointless if it only ever says ok"


def test_a_rejected_row_carries_the_gate_reason(data_root):
    """`quality` collapses two reasons to `reject`; the notes keep them apart."""
    build_session(data_root, "SYNTH-01", frames=3, bad_frame_markers={1: 3})
    ingest_session("SYNTH-01", data_root)
    partial = [r for r in rows_of(data_root) if r["quality"] == "partial"]
    assert partial
    assert all("gate:partial" in r["notes"] for r in partial)


def test_rejected_rows_have_no_split(data_root):
    """Empty, not `unassigned` — a rejected frame is in no partition at all."""
    build_session(data_root, "SYNTH-01", frames=3, bad_frame_markers={1: 3})
    ingest_session("SYNTH-01", data_root)
    for r in rows_of(data_root):
        if r["quality"] != "ok":
            assert r["split"] == SPLIT_NONE


def test_accepted_and_rejected_rows_account_for_every_raw_frame(data_root):
    build_session(data_root, "SYNTH-01", frames=4, bad_frame_markers={1: 3, 2: 0})
    report = ingest_session("SYNTH-01", data_root)
    rows = rows_of(data_root)
    assert sum(1 for r in rows if r["quality"] == "ok") == report.accepted
    assert sum(1 for r in rows if r["quality"] != "ok") == report.rejected


def test_image_ids_are_unique(data_root):
    build_session(data_root, "SYNTH-01", frames=4)
    ingest_session("SYNTH-01", data_root)
    ids = [r["image_id"] for r in rows_of(data_root)]
    assert len(ids) == len(set(ids))


def test_image_id_follows_the_spec_format(data_root):
    build_session(data_root, "SYNTH-01", frames=2)
    ingest_session("SYNTH-01", data_root)
    for r in rows_of(data_root):
        assert r["image_id"].startswith(f"{r['session_id']}_{r['freq_hz']}_")


def test_columns_match_the_spec(data_root):
    build_session(data_root, "SYNTH-01", frames=2)
    ingest_session("SYNTH-01", data_root)
    with (data_root / "metadata.csv").open(newline="", encoding="utf-8") as fh:
        assert next(csv.reader(fh)) == list(METADATA_COLUMNS)


def test_conditions_come_from_the_session_log(data_root):
    build_session(data_root, "SYNTH-01", frames=2)
    ingest_session("SYNTH-01", data_root)
    r = rows_of(data_root)[0]
    assert r["material"] == "sand"
    assert r["lighting"] == "lamp_diffuse"
    assert float(r["grain_mass_g"]) == 2.0
    assert int(r["distance_mm"]) == 210


def test_split_is_left_for_the_splitter(data_root):
    """Ingest never assigns a split — `tasks.py split` does, by session."""
    build_session(data_root, "SYNTH-01", frames=2)
    ingest_session("SYNTH-01", data_root)
    accepted = [r for r in rows_of(data_root) if r["quality"] == "ok"]
    assert {r["split"] for r in accepted} == {SPLIT_UNASSIGNED}


def test_missing_session_log_is_refused(data_root):
    build_session(data_root, "SYNTH-01", frames=2)
    (data_root / "raw" / "SYNTH-01" / "session.json").unlink()
    with pytest.raises(IngestError, match="session log"):
        ingest_session("SYNTH-01", data_root)


# --------------------------------------------------------------------------- #
# idempotence
# --------------------------------------------------------------------------- #

def test_reingesting_replaces_rows_and_never_duplicates(data_root):
    build_session(data_root, "SYNTH-01", frames=3)
    first = ingest_session("SYNTH-01", data_root)
    before = rows_of(data_root)

    second = ingest_session("SYNTH-01", data_root)
    after = rows_of(data_root)

    assert second.accepted == first.accepted
    assert second.total == first.total
    assert len(after) == len(before)
    assert [r["image_id"] for r in after] == [r["image_id"] for r in before]


def test_reingesting_one_session_leaves_the_others_alone(data_root):
    build_session(data_root, "SYNTH-01", frames=2)
    build_session(data_root, "SYNTH-02", frames=2)
    ingest_session("SYNTH-01", data_root)
    ingest_session("SYNTH-02", data_root)
    untouched = [r for r in rows_of(data_root) if r["session_id"] == "SYNTH-02"]

    ingest_session("SYNTH-01", data_root)
    assert [r for r in rows_of(data_root) if r["session_id"] == "SYNTH-02"] == untouched
    assert {r["session_id"] for r in rows_of(data_root)} == {"SYNTH-01", "SYNTH-02"}


def test_reingest_after_frames_are_removed_shrinks_the_session(data_root):
    build_session(data_root, "SYNTH-01", frames=4)
    ingest_session("SYNTH-01", data_root)
    n_before = len(rows_of(data_root))

    for p in sorted((data_root / "raw" / "SYNTH-01" / "420").glob("*.png"))[:2]:
        p.unlink()
    ingest_session("SYNTH-01", data_root)
    assert len(rows_of(data_root)) < n_before


# --------------------------------------------------------------------------- #
# provenance — synthetic and real must never share a metadata.csv
# --------------------------------------------------------------------------- #

def test_synthetic_and_real_sessions_are_told_apart():
    assert is_synthetic("SYNTH-01")
    assert not is_synthetic("20260916-01")


def test_a_real_session_cannot_join_synthetic_ones(data_root):
    build_session(data_root, "SYNTH-01", frames=2)
    build_session(data_root, "20260916-01", frames=2)
    ingest_session("SYNTH-01", data_root)

    with pytest.raises(MixedProvenanceError, match="SYNTH"):
        ingest_session("20260916-01", data_root)


def test_a_synthetic_session_cannot_join_real_ones(data_root):
    build_session(data_root, "20260916-01", frames=2)
    build_session(data_root, "SYNTH-01", frames=2)
    ingest_session("20260916-01", data_root)

    with pytest.raises(MixedProvenanceError, match="SYNTH"):
        ingest_session("SYNTH-01", data_root)


def test_the_refusal_leaves_the_existing_metadata_untouched(data_root):
    build_session(data_root, "SYNTH-01", frames=2)
    build_session(data_root, "20260916-01", frames=2)
    ingest_session("SYNTH-01", data_root)
    before = (data_root / "metadata.csv").read_bytes()

    with pytest.raises(MixedProvenanceError):
        ingest_session("20260916-01", data_root)
    assert (data_root / "metadata.csv").read_bytes() == before


@pytest.mark.parametrize("bad", ["nope", "2026-09-16", "SYNTH_01", "20260916", "synth-01"])
def test_malformed_session_ids_are_refused(bad):
    with pytest.raises(MetadataError):
        validate_session_id(bad)


def test_ingest_refuses_a_malformed_session_id(data_root):
    """`IngestError` subclasses `MetadataError`, so this is catchable either way."""
    (data_root / "raw" / "not-a-session").mkdir(parents=True)
    with pytest.raises(IngestError):
        ingest_session("not-a-session", data_root)


@pytest.mark.parametrize("good", ["SYNTH-01", "20260916-01", "20261231-99"])
def test_well_formed_session_ids_are_accepted(good):
    validate_session_id(good)


# --------------------------------------------------------------------------- #
# the quality gate, end to end
# --------------------------------------------------------------------------- #

def test_rejected_frames_stay_in_raw(data_root):
    build_session(data_root, "SYNTH-01", frames=3, bad_frame_markers={1: 3})
    before = sorted(p.name for p in (data_root / "raw" / "SYNTH-01").rglob("*.png"))
    report = ingest_session("SYNTH-01", data_root)

    after = sorted(p.name for p in (data_root / "raw" / "SYNTH-01").rglob("*.png"))
    assert after == before
    assert report.rejected > 0


def test_a_three_marker_frame_is_partial_not_center_cropped(data_root):
    build_session(data_root, "SYNTH-01", frames=3, bad_frame_markers={1: 3})
    report = ingest_session("SYNTH-01", data_root)

    assert report.by_reason.get("partial", 0) == 2       # one per class
    assert report.roi_mode == "homography"
    assert all(r["roi_mode"] == "homography" for r in rows_of(data_root))


def test_report_breaks_rejections_down_by_reason(data_root):
    build_session(data_root, "SYNTH-01", frames=3, bad_frame_markers={2: 0})
    ingest_session("SYNTH-01", data_root)

    report = json.loads((data_root / "ingest_report.json").read_text("utf-8"))
    session = report["sessions"]["SYNTH-01"]
    assert session["total"] == 6
    assert session["accepted"] + sum(session["rejected"].values()) == session["total"]
    # The report rounds to 4 dp so it stays readable; that is the only slack allowed.
    assert session["rejection_rate"] == pytest.approx(
        sum(session["rejected"].values()) / session["total"], abs=1e-4
    )


def test_report_records_the_thresholds_that_were_applied(data_root):
    """A yield figure is meaningless without the gate that produced it."""
    build_session(data_root, "SYNTH-01", frames=2)
    ingest_session("SYNTH-01", data_root)
    report = json.loads((data_root / "ingest_report.json").read_text("utf-8"))
    assert report["gate"]["laplacian_variance_min"] == 100.0
    assert report["gate"]["edge_density"]["interior_margin_frac"] == 0.02


def test_report_lists_every_rejected_image_by_path(data_root):
    """Rejects are counted *and* traceable — someone has to be able to go and look."""
    build_session(data_root, "SYNTH-01", frames=3, bad_frame_markers={1: 3})
    ingest_session("SYNTH-01", data_root)
    report = json.loads((data_root / "ingest_report.json").read_text("utf-8"))

    listed = report["sessions"]["SYNTH-01"]["rejected_images"]
    assert listed
    for item in listed:
        assert (data_root / item["path"]).exists()
        assert item["reason"]


def test_report_replaces_rather_than_appends_on_reingest(data_root):
    build_session(data_root, "SYNTH-01", frames=3)
    ingest_session("SYNTH-01", data_root)
    ingest_session("SYNTH-01", data_root)

    report = json.loads((data_root / "ingest_report.json").read_text("utf-8"))
    assert report["sessions"]["SYNTH-01"]["total"] == 6
    assert report["totals"]["total"] == 6


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #

def test_a_session_without_calibration_falls_back_to_center_crop_and_says_so(data_root):
    """Allowed by the ingest skill — but flagged, never silent."""
    build_session(data_root, "SYNTH-01", frames=2, with_calib=False)
    report = ingest_session("SYNTH-01", data_root)

    assert report.roi_mode == "center_crop"
    assert report.flags
    assert any("calibration" in f for f in report.flags)
    assert all(r["roi_mode"] == "center_crop" for r in rows_of(data_root))


def test_uncalibrated_session_does_not_reject_frames_as_partial(data_root):
    """With no homography there is no marker requirement — §6 rule 2 is conditional."""
    build_session(data_root, "SYNTH-01", frames=2, with_calib=False, )
    report = ingest_session("SYNTH-01", data_root)
    assert report.by_reason.get("partial", 0) == 0


# --------------------------------------------------------------------------- #
# processed output
# --------------------------------------------------------------------------- #

def test_processed_images_are_written_for_accepted_frames_only(data_root):
    build_session(data_root, "SYNTH-01", frames=3, bad_frame_markers={1: 3})
    report = ingest_session("SYNTH-01", data_root)
    written = list((data_root / "processed").rglob("*.png"))
    assert len(written) == report.accepted


def test_processed_images_are_at_model_resolution(data_root):
    build_session(data_root, "SYNTH-01", frames=2)
    ingest_session("SYNTH-01", data_root)
    p = next((data_root / "processed").rglob("*.png"))
    assert cv2.imread(str(p), cv2.IMREAD_UNCHANGED).shape == (128, 128)


def test_reingest_does_not_leave_orphaned_processed_images(data_root):
    build_session(data_root, "SYNTH-01", frames=4)
    ingest_session("SYNTH-01", data_root)
    for p in sorted((data_root / "raw" / "SYNTH-01" / "420").glob("*.png"))[:2]:
        p.unlink()
    report = ingest_session("SYNTH-01", data_root)
    assert len(list((data_root / "processed").rglob("*.png"))) == report.accepted


def test_unknown_session_is_refused(data_root):
    (data_root / "raw").mkdir(parents=True)
    with pytest.raises(IngestError, match="not found"):
        ingest_session("SYNTH-09", data_root)
