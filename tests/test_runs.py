"""Contract tests for `src/runs.py` — the run directory and the preprocessing hash.

`docs/03-preprocessing-spec.md` §4 calls the hash "the one guard": with a single
preprocessing implementation there is nothing to compare, and the only thing that can still
go wrong is serving a checkpoint trained under a different version of the spec. The failure
has no symptom — the images look right, the probabilities look right, and the answer is
wrong — so the check has to be a hard failure and it has to be exact.

The two things pinned here are the two ways the hash could be useless: too sensitive (it
fires on a reformat, everyone learns to ignore it) or not sensitive enough (a changed Canny
threshold slips through).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from runs import (
    OpenCVMismatch,
    PreprocessMismatch,
    RunsError,
    check_opencv,
    check_preprocess_hash,
    check_run,
    latest_run,
    missing_artefacts,
    new_run_dir,
    preprocess_hash,
    resolve_run,
    write_latest,
)

CONFIG = {
    "version": "1.1.0",
    "clahe": {"enabled": True, "clip_limit": 2.0, "tile_grid_size": [8, 8]},
    "canny": {"threshold1": 50, "threshold2": 150},
}


def finished_run(root: Path, name: str, config: dict, **overrides) -> Path:
    run_dir = root / name
    (run_dir / "report").mkdir(parents=True)
    payload = {
        "preprocess_version": config["version"],
        "preprocess_sha256": preprocess_hash(config),
        **overrides,
    }
    (run_dir / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    for artefact in ("metrics.json", "history.csv", "model.pt"):
        (run_dir / artefact).write_text("x", encoding="utf-8")
    return run_dir


# --------------------------------------------------------------------------- #
# the hash
# --------------------------------------------------------------------------- #

def test_the_same_config_hashes_the_same():
    assert preprocess_hash(CONFIG) == preprocess_hash(json.loads(json.dumps(CONFIG)))


def test_key_order_does_not_change_the_hash():
    """A reordered dict is the same config. Hashing insertion order would fire on nothing."""
    reordered = {"canny": CONFIG["canny"], "version": CONFIG["version"], "clahe": CONFIG["clahe"]}
    assert preprocess_hash(reordered) == preprocess_hash(CONFIG)


def test_reindenting_the_file_does_not_change_the_hash(tmp_path):
    """Hashing the bytes would make `git checkout` on Windows look like a Canny change."""
    path = tmp_path / "compact.json"
    path.write_text(json.dumps(CONFIG, separators=(",", ":")), encoding="utf-8")
    compact = json.loads(path.read_text("utf-8"))
    path.write_text(json.dumps(CONFIG, indent=4) + "\n", encoding="utf-8")
    pretty = json.loads(path.read_text("utf-8"))
    assert preprocess_hash(compact) == preprocess_hash(pretty)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c["canny"].update(threshold1=51),
        lambda c: c["clahe"].update(clip_limit=2.1),
        lambda c: c["clahe"].update(tile_grid_size=[8, 16]),
        lambda c: c["clahe"].update(enabled=False),
        lambda c: c.update(version="1.2.0"),
    ],
)
def test_any_changed_value_changes_the_hash(mutate):
    changed = json.loads(json.dumps(CONFIG))
    mutate(changed)
    assert preprocess_hash(changed) != preprocess_hash(CONFIG)


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #

def test_a_matching_checkpoint_passes(tmp_path):
    run_dir = finished_run(tmp_path, "20260101-000000", CONFIG)
    assert check_preprocess_hash(run_dir, CONFIG)["sha256"] == preprocess_hash(CONFIG)


def test_a_changed_threshold_is_a_hard_failure(tmp_path):
    """§4: "This is a hard failure, never a warning"."""
    run_dir = finished_run(tmp_path, "20260101-000000", CONFIG)
    changed = json.loads(json.dumps(CONFIG))
    changed["canny"]["threshold1"] = 60
    with pytest.raises(PreprocessMismatch):
        check_preprocess_hash(run_dir, changed)


def test_the_mismatch_message_names_both_versions(tmp_path):
    """§4.2: "refuses to start and prints both versions"."""
    run_dir = finished_run(tmp_path, "20260101-000000", CONFIG)
    changed = json.loads(json.dumps(CONFIG))
    changed["version"] = "1.2.0"
    with pytest.raises(PreprocessMismatch) as exc:
        check_preprocess_hash(run_dir, changed)
    assert "1.1.0" in str(exc.value)
    assert "1.2.0" in str(exc.value)


def test_a_checkpoint_with_no_hash_cannot_be_served(tmp_path):
    """Not "assume it is fine" — a checkpoint whose preprocessing cannot be verified is
    exactly as dangerous as one that mismatches."""
    run_dir = tmp_path / "20260101-000000"
    (run_dir / "report").mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"preprocess_version": "1.1.0"}), "utf-8")
    with pytest.raises(RunsError, match="preprocess_sha256"):
        check_preprocess_hash(run_dir, CONFIG)


# --------------------------------------------------------------------------- #
# the run directory and the pointer file
# --------------------------------------------------------------------------- #

def test_a_new_run_directory_has_a_report_subdirectory(tmp_path):
    run_dir = new_run_dir(tmp_path, stamp="20260101-000000")
    assert (run_dir / "report").is_dir()


def test_an_existing_run_directory_is_never_overwritten(tmp_path):
    new_run_dir(tmp_path, stamp="20260101-000000")
    with pytest.raises(RunsError, match="refusing"):
        new_run_dir(tmp_path, stamp="20260101-000000")


def test_latest_is_a_pointer_file_not_a_symlink(tmp_path):
    """CLAUDE.md: native Windows, no symlinks. `runs/latest.txt` holds a directory name."""
    run_dir = new_run_dir(tmp_path, stamp="20260101-000000")
    pointer = write_latest(run_dir, tmp_path)
    assert not pointer.is_symlink()
    assert pointer.read_text("utf-8").strip() == "20260101-000000"
    assert latest_run(tmp_path) == run_dir


def test_a_pointer_to_a_deleted_run_is_reported_not_ignored(tmp_path):
    run_dir = new_run_dir(tmp_path, stamp="20260101-000000")
    write_latest(run_dir, tmp_path)
    import shutil

    shutil.rmtree(run_dir)
    with pytest.raises(RunsError, match="does not exist"):
        latest_run(tmp_path)


def test_no_runs_at_all_names_the_target_that_makes_one(tmp_path):
    with pytest.raises(RunsError, match="tasks.py train"):
        latest_run(tmp_path)


def test_run_option_overrides_the_pointer(tmp_path):
    first = new_run_dir(tmp_path, stamp="20260101-000000")
    second = new_run_dir(tmp_path, stamp="20260102-000000")
    write_latest(second, tmp_path)
    assert resolve_run([], tmp_path) == second
    assert resolve_run(["--run", "20260101-000000"], tmp_path) == first


def test_an_unfinished_run_is_reported_by_what_it_lacks(tmp_path):
    run_dir = new_run_dir(tmp_path, stamp="20260101-000000")
    assert sorted(missing_artefacts(run_dir)) == [
        "config.json", "history.csv", "metrics.json", "model.pt",
    ]


# --------------------------------------------------------------------------- #
# the second key — the OpenCV build
# --------------------------------------------------------------------------- #

OPENCV = {"opencv_version": "4.12.0", "opencv_package": "opencv-contrib-python"}


def stamped_run(root: Path, name: str, config: dict, **overrides) -> Path:
    """A finished run carrying both keys of the §4 guard."""
    return finished_run(root, name, config, **{**OPENCV, **overrides})


def test_the_matching_build_passes(tmp_path):
    run_dir = stamped_run(tmp_path, "20260101-000000", CONFIG)
    assert check_opencv(run_dir, OPENCV)["opencv_version"] == "4.12.0"


def test_a_different_opencv_version_is_refused(tmp_path):
    """The hash still matches — that is the entire point of this key.

    docs/adr/0006: between 4.12 and 5.0 the ROI warp differs by up to 7 grey levels in every
    frame and every final tensor differs, while `config/preprocess.json` is untouched.
    """
    run_dir = stamped_run(tmp_path, "20260101-000000", CONFIG)
    check_preprocess_hash(run_dir, CONFIG)          # key one is happy
    with pytest.raises(OpenCVMismatch):             # key two is not
        check_opencv(run_dir, {**OPENCV, "opencv_version": "5.0.0"})


def test_the_message_says_the_config_is_unchanged(tmp_path):
    """Otherwise the first thing anyone does is go looking at preprocess.json."""
    run_dir = stamped_run(tmp_path, "20260101-000000", CONFIG)
    with pytest.raises(OpenCVMismatch) as exc:
        check_opencv(run_dir, {**OPENCV, "opencv_version": "5.0.0"})
    message = str(exc.value)
    assert "unchanged" in message
    assert "0006" in message
    assert "4.12.0" in message and "5.0.0" in message


def test_the_non_contrib_build_is_refused_even_at_the_same_version(tmp_path):
    """§5.3: contrib and plain are different software. Same cv2.__version__, different build."""
    run_dir = stamped_run(tmp_path, "20260101-000000", CONFIG)
    with pytest.raises(OpenCVMismatch):
        check_opencv(run_dir, {**OPENCV, "opencv_package": "opencv-python"})


def test_a_wheel_bump_that_leaves_cv2_alone_does_not_fire(tmp_path):
    """4.12.0.88 -> 4.12.0.90 keeps cv2.__version__ at 4.12.0 and changes no pixel. A guard
    that fired here would be retrained around rather than trusted."""
    run_dir = stamped_run(tmp_path, "20260101-000000", CONFIG, opencv_wheel_version="4.12.0.88")
    assert check_opencv(run_dir, {**OPENCV, "opencv_wheel_version": "4.12.0.90"})


def test_a_checkpoint_with_no_opencv_version_cannot_be_served(tmp_path):
    """Written before this key existed. Unverifiable is not the same as fine."""
    run_dir = finished_run(tmp_path, "20260101-000000", CONFIG)
    with pytest.raises(RunsError, match="opencv_version"):
        check_opencv(run_dir, OPENCV)


def test_check_run_applies_both_keys(tmp_path):
    run_dir = stamped_run(tmp_path, "20260101-000000", CONFIG)
    result = check_run(run_dir, CONFIG)
    assert result["sha256"] == preprocess_hash(CONFIG)
    assert result["opencv_version"] == "4.12.0"


def test_check_run_reads_the_live_environment_when_not_told_otherwise(tmp_path):
    """The signature has to come from the interpreter actually about to run the model."""
    from environment import opencv_signature

    run_dir = stamped_run(tmp_path, "20260101-000000", CONFIG, **opencv_signature())
    assert check_run(run_dir, CONFIG)["opencv_version"] == opencv_signature()["opencv_version"]


def test_the_signature_names_the_installed_distribution():
    from environment import installed_opencv_distributions, opencv_signature

    signature = opencv_signature()
    assert signature["opencv_package"] in installed_opencv_distributions()
    assert signature["opencv_version"]
