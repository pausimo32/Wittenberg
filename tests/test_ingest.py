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
from ingest import (
    REQUIRED_SESSION_KEYS,
    IngestError,
    MixedProvenanceError,
    ingest_session,
)
from metadata import (
    METADATA_COLUMNS,
    QUALITY_VALUES,
    SPLIT_NONE,
    SPLIT_UNASSIGNED,
    MetadataError,
    is_synthetic,
    read_metadata,
    session_date,
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

#: The rest of `session.json`. Ingest refuses a session without every one of these, so
#: the helper writes a complete log and the tests that care remove a key on purpose.
PROVENANCE = {"operator": "test", "plate": "test-200x200", "notes": ""}


def session_log(session_id: str, conditions: dict | None = None) -> dict:
    log = {
        "session_id": session_id,
        "date": session_date(session_id),
        **PROVENANCE,
        **(conditions or CONDITIONS),
    }
    if is_synthetic(session_id):
        log["synthetic"] = True
    return log


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
        json.dumps(session_log(session_id, conditions)), encoding="utf-8"
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
    build_session(data_root, "SYNTH-20260101-01", frames=3, bad_frame_markers={1: 3})
    report = ingest_session("SYNTH-20260101-01", data_root)

    rows = rows_of(data_root)
    assert report.total == 6                     # 2 classes x 3 frames
    assert len(rows) == report.total
    assert len(rows) > report.accepted           # some were rejected and still have a row


def test_rejected_frames_have_a_row(data_root):
    build_session(data_root, "SYNTH-20260101-01", frames=3, bad_frame_markers={1: 3})
    report = ingest_session("SYNTH-20260101-01", data_root)

    rejected = [r for r in rows_of(data_root) if r["quality"] != "ok"]
    assert len(rejected) == report.rejected
    assert rejected, "a rejected frame must still be recorded in the committed file"


def test_quality_column_uses_the_spec_vocabulary(data_root):
    build_session(data_root, "SYNTH-20260101-01", frames=3, bad_frame_markers={1: 3})
    ingest_session("SYNTH-20260101-01", data_root)
    seen = {r["quality"] for r in rows_of(data_root)}
    assert seen <= set(QUALITY_VALUES)
    assert len(seen) > 1, "the column is pointless if it only ever says ok"


def test_a_rejected_row_carries_the_gate_reason(data_root):
    """`quality` collapses two reasons to `reject`; the notes keep them apart."""
    build_session(data_root, "SYNTH-20260101-01", frames=3, bad_frame_markers={1: 3})
    ingest_session("SYNTH-20260101-01", data_root)
    partial = [r for r in rows_of(data_root) if r["quality"] == "partial"]
    assert partial
    assert all("gate:partial" in r["notes"] for r in partial)


def test_rejected_rows_have_no_split(data_root):
    """Empty, not `unassigned` — a rejected frame is in no partition at all."""
    build_session(data_root, "SYNTH-20260101-01", frames=3, bad_frame_markers={1: 3})
    ingest_session("SYNTH-20260101-01", data_root)
    for r in rows_of(data_root):
        if r["quality"] != "ok":
            assert r["split"] == SPLIT_NONE


def test_accepted_and_rejected_rows_account_for_every_raw_frame(data_root):
    build_session(data_root, "SYNTH-20260101-01", frames=4, bad_frame_markers={1: 3, 2: 0})
    report = ingest_session("SYNTH-20260101-01", data_root)
    rows = rows_of(data_root)
    assert sum(1 for r in rows if r["quality"] == "ok") == report.accepted
    assert sum(1 for r in rows if r["quality"] != "ok") == report.rejected


def test_image_ids_are_unique(data_root):
    build_session(data_root, "SYNTH-20260101-01", frames=4)
    ingest_session("SYNTH-20260101-01", data_root)
    ids = [r["image_id"] for r in rows_of(data_root)]
    assert len(ids) == len(set(ids))


def test_image_id_follows_the_spec_format(data_root):
    build_session(data_root, "SYNTH-20260101-01", frames=2)
    ingest_session("SYNTH-20260101-01", data_root)
    for r in rows_of(data_root):
        assert r["image_id"].startswith(f"{r['session_id']}_{r['freq_hz']}_")


def test_columns_match_the_spec(data_root):
    build_session(data_root, "SYNTH-20260101-01", frames=2)
    ingest_session("SYNTH-20260101-01", data_root)
    with (data_root / "metadata.csv").open(newline="", encoding="utf-8") as fh:
        assert next(csv.reader(fh)) == list(METADATA_COLUMNS)


def test_conditions_come_from_the_session_log(data_root):
    build_session(data_root, "SYNTH-20260101-01", frames=2)
    ingest_session("SYNTH-20260101-01", data_root)
    r = rows_of(data_root)[0]
    assert r["material"] == "sand"
    assert r["lighting"] == "lamp_diffuse"
    assert float(r["grain_mass_g"]) == 2.0
    assert int(r["distance_mm"]) == 210


def test_split_is_left_for_the_splitter(data_root):
    """Ingest never assigns a split — `tasks.py split` does, by session."""
    build_session(data_root, "SYNTH-20260101-01", frames=2)
    ingest_session("SYNTH-20260101-01", data_root)
    accepted = [r for r in rows_of(data_root) if r["quality"] == "ok"]
    assert {r["split"] for r in accepted} == {SPLIT_UNASSIGNED}


def test_missing_session_log_is_refused(data_root):
    build_session(data_root, "SYNTH-20260101-01", frames=2)
    (data_root / "raw" / "SYNTH-20260101-01" / "session.json").unlink()
    with pytest.raises(IngestError, match="session.json"):
        ingest_session("SYNTH-20260101-01", data_root)


@pytest.mark.parametrize("key", REQUIRED_SESSION_KEYS)
def test_every_required_session_key_is_refused_when_absent(data_root, key):
    """A hard block, not a warning.

    Five of these keys are columns of `metadata.csv` (§4) and none of them can be
    reconstructed from the images four days later. A warning here reads as "carry on"
    and the session is ingested without them, which is not recoverable once the rig has
    been dismantled.
    """
    build_session(data_root, "SYNTH-20260101-01", frames=2)
    path = data_root / "raw" / "SYNTH-20260101-01" / "session.json"
    log = json.loads(path.read_text("utf-8"))
    del log[key]
    path.write_text(json.dumps(log), encoding="utf-8")

    with pytest.raises(IngestError, match=key):
        ingest_session("SYNTH-20260101-01", data_root)


@pytest.mark.parametrize("key", [k for k in REQUIRED_SESSION_KEYS if k != "notes"])
def test_an_empty_required_value_is_refused(data_root, key):
    build_session(data_root, "SYNTH-20260101-01", frames=2)
    path = data_root / "raw" / "SYNTH-20260101-01" / "session.json"
    log = json.loads(path.read_text("utf-8"))
    log[key] = ""
    path.write_text(json.dumps(log), encoding="utf-8")

    with pytest.raises(IngestError, match=key):
        ingest_session("SYNTH-20260101-01", data_root)


def test_empty_notes_are_accepted(data_root):
    """`notes` must be present; empty means "nothing anomalous", which is information."""
    build_session(data_root, "SYNTH-20260101-01", frames=2)
    report = ingest_session("SYNTH-20260101-01", data_root)
    assert report.total > 0


def test_a_session_log_copied_from_another_session_is_refused(data_root):
    build_session(data_root, "SYNTH-20260101-01", frames=2)
    path = data_root / "raw" / "SYNTH-20260101-01" / "session.json"
    log = json.loads(path.read_text("utf-8"))
    log["session_id"] = "SYNTH-20260102-01"
    path.write_text(json.dumps(log), encoding="utf-8")

    with pytest.raises(IngestError, match="copy-paste"):
        ingest_session("SYNTH-20260101-01", data_root)


def test_a_date_that_disagrees_with_the_session_id_is_refused(data_root):
    """§2 makes one session one day. Two dates means two sessions or a typo."""
    build_session(data_root, "SYNTH-20260101-01", frames=2)
    path = data_root / "raw" / "SYNTH-20260101-01" / "session.json"
    log = json.loads(path.read_text("utf-8"))
    log["date"] = "2026-01-09"
    path.write_text(json.dumps(log), encoding="utf-8")

    with pytest.raises(IngestError, match="date"):
        ingest_session("SYNTH-20260101-01", data_root)


def test_a_hyphenated_date_is_accepted(data_root):
    """`2026-01-01` and `20260101` are the same day written two ways."""
    build_session(data_root, "SYNTH-20260101-01", frames=2)
    path = data_root / "raw" / "SYNTH-20260101-01" / "session.json"
    log = json.loads(path.read_text("utf-8"))
    log["date"] = "2026-01-01"
    path.write_text(json.dumps(log), encoding="utf-8")

    assert ingest_session("SYNTH-20260101-01", data_root).total > 0


def test_a_synth_session_without_the_synthetic_flag_is_refused(data_root):
    """The flag is what keeps a rendered frame from later reading as a measurement."""
    build_session(data_root, "SYNTH-20260101-01", frames=2)
    path = data_root / "raw" / "SYNTH-20260101-01" / "session.json"
    log = json.loads(path.read_text("utf-8"))
    del log["synthetic"]
    path.write_text(json.dumps(log), encoding="utf-8")

    with pytest.raises(IngestError, match="synthetic"):
        ingest_session("SYNTH-20260101-01", data_root)


def test_a_real_session_flagged_synthetic_is_refused(data_root):
    build_session(data_root, "20260916-01", frames=2)
    path = data_root / "raw" / "20260916-01" / "session.json"
    log = json.loads(path.read_text("utf-8"))
    log["synthetic"] = True
    path.write_text(json.dumps(log), encoding="utf-8")

    with pytest.raises(IngestError, match="synthetic"):
        ingest_session("20260916-01", data_root)


def test_the_report_carries_the_whole_session_log(data_root):
    """`operator` and `plate` are not metadata.csv columns, so this is where they live."""
    build_session(data_root, "SYNTH-20260101-01", frames=2)
    ingest_session("SYNTH-20260101-01", data_root)
    logged = json.loads((data_root / "ingest_report.json").read_text("utf-8"))
    logged = logged["sessions"]["SYNTH-20260101-01"]["session_log"]
    assert set(logged) == set(REQUIRED_SESSION_KEYS)
    assert logged["operator"] == "test"


# --------------------------------------------------------------------------- #
# idempotence
# --------------------------------------------------------------------------- #

def test_reingesting_replaces_rows_and_never_duplicates(data_root):
    build_session(data_root, "SYNTH-20260101-01", frames=3)
    first = ingest_session("SYNTH-20260101-01", data_root)
    before = rows_of(data_root)

    second = ingest_session("SYNTH-20260101-01", data_root)
    after = rows_of(data_root)

    assert second.accepted == first.accepted
    assert second.total == first.total
    assert len(after) == len(before)
    assert [r["image_id"] for r in after] == [r["image_id"] for r in before]


def test_reingesting_one_session_leaves_the_others_alone(data_root):
    build_session(data_root, "SYNTH-20260101-01", frames=2)
    build_session(data_root, "SYNTH-20260102-01", frames=2)
    ingest_session("SYNTH-20260101-01", data_root)
    ingest_session("SYNTH-20260102-01", data_root)
    untouched = [r for r in rows_of(data_root) if r["session_id"] == "SYNTH-20260102-01"]

    ingest_session("SYNTH-20260101-01", data_root)
    assert [r for r in rows_of(data_root) if r["session_id"] == "SYNTH-20260102-01"] == untouched
    assert {r["session_id"] for r in rows_of(data_root)} == {"SYNTH-20260101-01", "SYNTH-20260102-01"}


def test_reingest_after_frames_are_removed_shrinks_the_session(data_root):
    build_session(data_root, "SYNTH-20260101-01", frames=4)
    ingest_session("SYNTH-20260101-01", data_root)
    n_before = len(rows_of(data_root))

    for p in sorted((data_root / "raw" / "SYNTH-20260101-01" / "420").glob("*.png"))[:2]:
        p.unlink()
    ingest_session("SYNTH-20260101-01", data_root)
    assert len(rows_of(data_root)) < n_before


# --------------------------------------------------------------------------- #
# provenance — synthetic and real must never share a metadata.csv
# --------------------------------------------------------------------------- #

def test_synthetic_and_real_sessions_are_told_apart():
    assert is_synthetic("SYNTH-20260101-01")
    assert not is_synthetic("20260916-01")


def test_a_real_session_cannot_join_synthetic_ones(data_root):
    build_session(data_root, "SYNTH-20260101-01", frames=2)
    build_session(data_root, "20260916-01", frames=2)
    ingest_session("SYNTH-20260101-01", data_root)

    with pytest.raises(MixedProvenanceError, match="SYNTH"):
        ingest_session("20260916-01", data_root)


def test_a_synthetic_session_cannot_join_real_ones(data_root):
    build_session(data_root, "20260916-01", frames=2)
    build_session(data_root, "SYNTH-20260101-01", frames=2)
    ingest_session("20260916-01", data_root)

    with pytest.raises(MixedProvenanceError, match="SYNTH"):
        ingest_session("SYNTH-20260101-01", data_root)


def test_the_refusal_leaves_the_existing_metadata_untouched(data_root):
    build_session(data_root, "SYNTH-20260101-01", frames=2)
    build_session(data_root, "20260916-01", frames=2)
    ingest_session("SYNTH-20260101-01", data_root)
    before = (data_root / "metadata.csv").read_bytes()

    with pytest.raises(MixedProvenanceError):
        ingest_session("20260916-01", data_root)
    assert (data_root / "metadata.csv").read_bytes() == before


@pytest.mark.parametrize(
    "bad",
    ["nope", "2026-09-16", "SYNTH_01", "20260916", "synth-01", "SYNTH-01", "SYNTH-2026-01-01"],
)
def test_malformed_session_ids_are_refused(bad):
    with pytest.raises(MetadataError):
        validate_session_id(bad)


def test_ingest_refuses_a_malformed_session_id(data_root):
    """`IngestError` subclasses `MetadataError`, so this is catchable either way."""
    (data_root / "raw" / "not-a-session").mkdir(parents=True)
    with pytest.raises(IngestError):
        ingest_session("not-a-session", data_root)


@pytest.mark.parametrize("good", ["SYNTH-20260101-01", "20260916-01", "20261231-99"])
def test_well_formed_session_ids_are_accepted(good):
    validate_session_id(good)


# --------------------------------------------------------------------------- #
# the quality gate, end to end
# --------------------------------------------------------------------------- #

def test_rejected_frames_stay_in_raw(data_root):
    build_session(data_root, "SYNTH-20260101-01", frames=3, bad_frame_markers={1: 3})
    before = sorted(p.name for p in (data_root / "raw" / "SYNTH-20260101-01").rglob("*.png"))
    report = ingest_session("SYNTH-20260101-01", data_root)

    after = sorted(p.name for p in (data_root / "raw" / "SYNTH-20260101-01").rglob("*.png"))
    assert after == before
    assert report.rejected > 0


def test_a_three_marker_frame_is_partial_not_center_cropped(data_root):
    build_session(data_root, "SYNTH-20260101-01", frames=3, bad_frame_markers={1: 3})
    report = ingest_session("SYNTH-20260101-01", data_root)

    assert report.by_reason.get("partial", 0) == 2       # one per class
    assert report.roi_mode == "homography"
    assert all(r["roi_mode"] == "homography" for r in rows_of(data_root))


def test_report_breaks_rejections_down_by_reason(data_root):
    build_session(data_root, "SYNTH-20260101-01", frames=3, bad_frame_markers={2: 0})
    ingest_session("SYNTH-20260101-01", data_root)

    report = json.loads((data_root / "ingest_report.json").read_text("utf-8"))
    session = report["sessions"]["SYNTH-20260101-01"]
    assert session["total"] == 6
    assert session["accepted"] + sum(session["rejected"].values()) == session["total"]
    # The report rounds to 4 dp so it stays readable; that is the only slack allowed.
    assert session["rejection_rate"] == pytest.approx(
        sum(session["rejected"].values()) / session["total"], abs=1e-4
    )


def test_report_records_the_thresholds_that_were_applied(data_root):
    """A yield figure is meaningless without the gate that produced it."""
    build_session(data_root, "SYNTH-20260101-01", frames=2)
    ingest_session("SYNTH-20260101-01", data_root)
    report = json.loads((data_root / "ingest_report.json").read_text("utf-8"))
    assert report["gate"]["laplacian_variance_min"] == 100.0
    assert report["gate"]["edge_density"]["interior_margin_frac"] == 0.02


def test_report_lists_every_rejected_image_by_path(data_root):
    """Rejects are counted *and* traceable — someone has to be able to go and look."""
    build_session(data_root, "SYNTH-20260101-01", frames=3, bad_frame_markers={1: 3})
    ingest_session("SYNTH-20260101-01", data_root)
    report = json.loads((data_root / "ingest_report.json").read_text("utf-8"))

    listed = report["sessions"]["SYNTH-20260101-01"]["rejected_images"]
    assert listed
    for item in listed:
        assert (data_root / item["path"]).exists()
        assert item["reason"]


def test_report_replaces_rather_than_appends_on_reingest(data_root):
    build_session(data_root, "SYNTH-20260101-01", frames=3)
    ingest_session("SYNTH-20260101-01", data_root)
    ingest_session("SYNTH-20260101-01", data_root)

    report = json.loads((data_root / "ingest_report.json").read_text("utf-8"))
    assert report["sessions"]["SYNTH-20260101-01"]["total"] == 6
    assert report["totals"]["total"] == 6


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #

def test_a_session_without_calibration_falls_back_to_center_crop_and_says_so(data_root):
    """Allowed by the ingest skill — but flagged, never silent."""
    build_session(data_root, "SYNTH-20260101-01", frames=2, with_calib=False)
    report = ingest_session("SYNTH-20260101-01", data_root)

    assert report.roi_mode == "center_crop"
    assert report.flags
    assert any("calibration" in f for f in report.flags)
    assert all(r["roi_mode"] == "center_crop" for r in rows_of(data_root))


def test_uncalibrated_session_does_not_reject_frames_as_partial(data_root):
    """With no homography there is no marker requirement — §6 rule 2 is conditional."""
    build_session(data_root, "SYNTH-20260101-01", frames=2, with_calib=False, )
    report = ingest_session("SYNTH-20260101-01", data_root)
    assert report.by_reason.get("partial", 0) == 0


# --------------------------------------------------------------------------- #
# processed output
# --------------------------------------------------------------------------- #

def test_processed_images_are_written_for_accepted_frames_only(data_root):
    build_session(data_root, "SYNTH-20260101-01", frames=3, bad_frame_markers={1: 3})
    report = ingest_session("SYNTH-20260101-01", data_root)
    written = list((data_root / "processed").rglob("*.png"))
    assert len(written) == report.accepted


def test_processed_images_are_at_model_resolution(data_root):
    build_session(data_root, "SYNTH-20260101-01", frames=2)
    ingest_session("SYNTH-20260101-01", data_root)
    p = next((data_root / "processed").rglob("*.png"))
    assert cv2.imread(str(p), cv2.IMREAD_UNCHANGED).shape == (128, 128)


def test_reingest_does_not_leave_orphaned_processed_images(data_root):
    build_session(data_root, "SYNTH-20260101-01", frames=4)
    ingest_session("SYNTH-20260101-01", data_root)
    for p in sorted((data_root / "raw" / "SYNTH-20260101-01" / "420").glob("*.png"))[:2]:
        p.unlink()
    report = ingest_session("SYNTH-20260101-01", data_root)
    assert len(list((data_root / "processed").rglob("*.png"))) == report.accepted


def test_unknown_session_is_refused(data_root):
    (data_root / "raw").mkdir(parents=True)
    with pytest.raises(IngestError, match="not found"):
        ingest_session("SYNTH-20260109-01", data_root)


# --------------------------------------------------------------------------- #
# parallelism — it must change the clock and nothing else
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("jobs", [1, 2])
def test_the_verdicts_do_not_depend_on_the_number_of_processes(data_root, tmp_path, jobs):
    """The whole licence for the pool. Ingest twice, compare the committed bytes.

    `src/preprocess.py` is a pure function and `src/quality.py` reads only its arguments, so
    frames are independent; the frame list is built in one fixed order and results are
    collected in that order. If this ever fails, the parallelism is not safe and the dataset
    depends on a machine's core count.
    """
    build_session(data_root, "SYNTH-20260101-01", frames=4, bad_frame_markers={1: 3})
    ingest_session("SYNTH-20260101-01", data_root, jobs=1)
    serial = (data_root / "metadata.csv").read_bytes()
    serial_report = json.loads((data_root / "ingest_report.json").read_text("utf-8"))["sessions"]

    ingest_session("SYNTH-20260101-01", data_root, jobs=jobs)
    assert (data_root / "metadata.csv").read_bytes() == serial
    replayed = json.loads((data_root / "ingest_report.json").read_text("utf-8"))["sessions"]
    assert replayed == serial_report


def test_the_processed_images_are_identical_too(data_root):
    """Not just the verdicts — the pixels the model will train on."""
    build_session(data_root, "SYNTH-20260101-01", frames=3)
    ingest_session("SYNTH-20260101-01", data_root, jobs=1)
    serial = {
        p.relative_to(data_root).as_posix(): p.read_bytes()
        for p in sorted((data_root / "processed").rglob("*.png"))
    }
    assert serial

    ingest_session("SYNTH-20260101-01", data_root, jobs=2)
    parallel = {
        p.relative_to(data_root).as_posix(): p.read_bytes()
        for p in sorted((data_root / "processed").rglob("*.png"))
    }
    assert parallel == serial


def test_the_pool_is_chosen_from_a_measured_frame_cost_not_a_frame_count(data_root):
    """One synthetic session — 120 frames of 600 px PNG at 19 ms — does not justify a pool.
    One real session of ~1 000 HEICs at ~560 ms plainly does. The frame counts are within an
    order of magnitude of each other; the per-frame costs are a factor of thirty apart, and
    only the second distinguishes the two cases."""
    from ingest import PARALLEL_MIN_SECONDS, is_worth_parallelising

    assert not is_worth_parallelising(120, 0.019)        # one F3a dry-run session: ~2.3 s
    assert is_worth_parallelising(1000, 0.560)           # one real session: ~9 minutes
    assert not is_worth_parallelising(10_000, 0.0)       # free frames are never worth a pool
    assert is_worth_parallelising(1, PARALLEL_MIN_SECONDS)


def test_an_empty_session_does_not_start_a_pool(data_root):
    """A session directory with no frequency folders is a mistake, not a reason to spawn."""
    from ingest import _assess_all

    assert _assess_all([], {}, "homography", None, None, jobs=8) == []


# --------------------------------------------------------------------------- #
# what /dades reuses — one implementation, not two that agree today  [F6]
# --------------------------------------------------------------------------- #

def test_gate_frame_is_the_verdict_ingest_records(data_root):
    """`docs/05` §9: the gate `/dades` applies and the gate `tasks.py ingest` applies are the
    same function, and this is where that stops being a claim."""
    from ingest import _frames_of, _session_calibration, gate_frame
    from preprocess import load_config
    from quality import load_gate

    build_session(data_root, "SYNTH-20260101-01", frames=3, bad_frame_markers={1: 3})
    calib, roi_mode, _flags = _session_calibration(data_root, "SYNTH-20260101-01")
    frames = _frames_of(data_root / "raw" / "SYNTH-20260101-01")
    report = ingest_session("SYNTH-20260101-01", data_root)
    rows = {r["image_id"]: r for r in rows_of(data_root)}

    for freq_hz, path in frames:
        quality, reason, _metrics = gate_frame(path, calib, roi_mode, load_config(), load_gate())
        image_id = f"SYNTH-20260101-01_{freq_hz}_{path.stem.split('_')[-1]}"
        assert quality == rows[image_id]["quality"]
        assert (reason is None) == (rows[image_id]["quality"] == "ok")
    assert report.total == len(frames)


def test_progress_is_called_once_per_frame(data_root):
    build_session(data_root, "SYNTH-20260101-01", frames=3)
    seen: list[tuple[int, int]] = []

    report = ingest_session(
        "SYNTH-20260101-01", data_root, jobs=1, progress=lambda d, t: seen.append((d, t))
    )

    assert [d for d, _t in seen] == list(range(1, report.total + 1))
    assert {t for _d, t in seen} == {report.total}


def test_progress_does_not_change_a_single_byte(data_root):
    """Reporting is observation. If it moved a byte it would not be reporting."""
    build_session(data_root, "SYNTH-20260101-01", frames=3, bad_frame_markers={1: 3})
    ingest_session("SYNTH-20260101-01", data_root, jobs=1)
    quiet = (data_root / "metadata.csv").read_bytes()

    ingest_session("SYNTH-20260101-01", data_root, jobs=1, progress=lambda d, t: None)

    assert (data_root / "metadata.csv").read_bytes() == quiet


def test_read_session_log_is_public_and_still_refuses(data_root):
    from ingest import read_session_log

    session_dir = data_root / "raw" / "SYNTH-20260101-01"
    session_dir.mkdir(parents=True)
    (session_dir / "session.json").write_text(
        '{"session_id": "SYNTH-20260101-01"}', encoding="utf-8"
    )

    with pytest.raises(IngestError, match="missing key"):
        read_session_log(session_dir, "SYNTH-20260101-01")


def test_assess_frame_keeps_stages_only_when_asked(data_root):
    """Default behaviour is ingest's, unchanged: a rejected frame costs no preprocess pass and
    a 48 MB RGBA array never leaves the function."""
    from ingest import _frames_of, _session_calibration, assess_frame
    from preprocess import load_config
    from quality import load_gate

    build_session(data_root, "SYNTH-20260101-01", frames=1)
    calib, roi_mode, _ = _session_calibration(data_root, "SYNTH-20260101-01")
    _freq, path = _frames_of(data_root / "raw" / "SYNTH-20260101-01")[0]

    plain = assess_frame(path, calib, roi_mode, load_config(), load_gate())
    rich = assess_frame(
        path, calib, roi_mode, load_config(), load_gate(), keep_stages=True, keep_input=True
    )

    assert plain.decoded is None
    assert rich.decoded is not None
    assert rich.stages is not None


def test_a_partial_frame_has_no_stages_even_when_they_are_asked_for(data_root):
    """Without four markers there is no rectified plate, so there is no edge map that means
    anything — and `/dades` must not be handed one to draw."""
    from ingest import _frames_of, _session_calibration, assess_frame
    from preprocess import load_config
    from quality import load_gate

    build_session(data_root, "SYNTH-20260101-01", frames=2, bad_frame_markers={1: 3})
    calib, roi_mode, _ = _session_calibration(data_root, "SYNTH-20260101-01")
    frames = _frames_of(data_root / "raw" / "SYNTH-20260101-01")

    partial = [
        assess_frame(
            path, calib, roi_mode, load_config(), load_gate(), keep_stages=True, keep_input=True
        )
        for _freq, path in frames
    ]
    partial = [a for a in partial if a.result is not None and a.result.reason == "partial"]

    assert partial, "the fixture was supposed to plant a three-marker frame"
    assert all(a.stages is None for a in partial)


def test_ingest_can_read_a_session_from_a_staging_directory(data_root):
    """The session id, and therefore every image_id, comes from the argument — never from the
    directory name. That is what lets `/dades` rename the directory after a successful ingest
    instead of before a possibly failing one."""
    build_session(data_root, "SYNTH-20260101-01", frames=2)
    staged = data_root / "raw" / ".incoming-SYNTH-20260101-01"
    (data_root / "raw" / "SYNTH-20260101-01").rename(staged)
    (data_root / "calib" / "SYNTH-20260101-01.json").rename(staged / "calibration.json")

    report = ingest_session("SYNTH-20260101-01", data_root, jobs=1, session_dir=staged)

    rows = rows_of(data_root)
    assert report.total == 4                     # 2 classes x 2 frames
    assert {r["session_id"] for r in rows} == {"SYNTH-20260101-01"}
    assert all(r["image_id"].startswith("SYNTH-20260101-01_") for r in rows)
    assert report.roi_mode == "homography"       # the staged calibration was found and used
