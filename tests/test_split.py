"""Contract tests for `src/split.py` — docs/02 §2 and docs/adr/0002.

The property under test is the one the whole project's credibility rests on: **no
session appears in more than one split**. Within a session, consecutive frames of the
same frequency are near-duplicates; if they straddle the train/test boundary the model
memorises the session and the reported accuracy means nothing. `CLAUDE.md` rule 3 calls
a random split a reported finding of the project, not a bug to work around.

So the tests below check the partition, not the accuracy: disjoint session sets, every
session in exactly one split, and the same answer every time at seed 42.
"""

from __future__ import annotations

import pytest

from metadata import (
    SPLIT_NONE,
    SPLIT_UNASSIGNED,
    SPLIT_VALUES,
    sessions_by_split,
    write_metadata,
)
from split import SEED, SplitError, assign_splits, split_dataset


def rows_for(sessions, per_session=4):
    out = []
    for s in sessions:
        for i in range(per_session):
            freq = 420 if i % 2 == 0 else 610
            out.append(
                {
                    "image_id": f"{s}_{freq}_{i:04d}",
                    "session_id": s,
                    "freq_hz": freq,
                    "split": SPLIT_UNASSIGNED,
                    "material": "sand",
                    "grain_mass_g": 2.0,
                    "distance_mm": 210,
                    "lighting": "lamp_diffuse",
                    "dwell_s": 3.0,
                    "roi_mode": "homography",
                    "quality": "ok",
                    "notes": "",
                }
            )
    return out


FOUR = ["SYNTH-01", "SYNTH-02", "SYNTH-03", "SYNTH-04"]


def with_rejects(rows, every=3, quality="blurred"):
    """Mark every Nth row rejected, the way the gate would have."""
    out = []
    for i, r in enumerate(rows):
        if i % every == 0:
            out.append({**r, "quality": quality, "split": SPLIT_NONE})
        else:
            out.append(dict(r))
    return out


# --------------------------------------------------------------------------- #
# rejected rows are present but unpartitioned
# --------------------------------------------------------------------------- #

def test_rejected_rows_keep_an_empty_split():
    """They are in metadata.csv per §4, but they are not training data."""
    assigned = assign_splits(with_rejects(rows_for(FOUR, per_session=6)))
    for r in assigned:
        if r["quality"] != "ok":
            assert r["split"] == SPLIT_NONE


def test_rejected_rows_never_enter_a_partition():
    assigned = assign_splits(with_rejects(rows_for(FOUR, per_session=6)))
    for split in SPLIT_VALUES:
        assert all(r["quality"] == "ok" for r in assigned if r["split"] == split)


def test_accepted_rows_are_still_fully_partitioned_alongside_rejects():
    assigned = assign_splits(with_rejects(rows_for(FOUR, per_session=6)))
    accepted = [r for r in assigned if r["quality"] == "ok"]
    assert accepted
    assert all(r["split"] in SPLIT_VALUES for r in accepted)


def test_rejects_do_not_change_which_session_lands_where():
    """The partition is over sessions, so the gate's yield must not move it."""
    clean = sessions_by_split(assign_splits(rows_for(FOUR, per_session=6)))
    dirty = sessions_by_split(assign_splits(with_rejects(rows_for(FOUR, per_session=6))))
    assert {k: v for k, v in clean.items() if k in SPLIT_VALUES} == {
        k: v for k, v in dirty.items() if k in SPLIT_VALUES
    }


def test_a_dataset_with_nothing_accepted_is_refused():
    all_bad = [{**r, "quality": "reject", "split": SPLIT_NONE} for r in rows_for(FOUR)]
    with pytest.raises(SplitError, match="rejected"):
        assign_splits(all_bad)


# --------------------------------------------------------------------------- #
# the partition
# --------------------------------------------------------------------------- #

def test_session_sets_are_disjoint():
    assigned = assign_splits(rows_for(FOUR))
    by_split = sessions_by_split(assigned)
    train, val, test = by_split["train"], by_split["val"], by_split["test"]
    assert train & val == set()
    assert train & test == set()
    assert val & test == set()


def test_every_session_lands_in_exactly_one_split():
    assigned = assign_splits(rows_for(FOUR))
    seen = {}
    for r in assigned:
        seen.setdefault(r["session_id"], set()).add(r["split"])
    assert all(len(v) == 1 for v in seen.values()), seen
    assert set(seen) == set(FOUR)


def test_no_row_is_left_unassigned():
    assigned = assign_splits(rows_for(FOUR))
    assert all(r["split"] in SPLIT_VALUES for r in assigned)


def test_four_sessions_split_two_one_one():
    """The target layout in §2: 2 train / 1 val / 1 test."""
    by_split = sessions_by_split(assign_splits(rows_for(FOUR)))
    assert len(by_split["train"]) == 2
    assert len(by_split["val"]) == 1
    assert len(by_split["test"]) == 1


def test_all_frames_of_a_session_share_its_split():
    """Splits are per session, never per image — `CLAUDE.md` rule 3."""
    assigned = assign_splits(rows_for(FOUR, per_session=9))
    for session in FOUR:
        splits = {r["split"] for r in assigned if r["session_id"] == session}
        assert len(splits) == 1


def test_uneven_session_sizes_do_not_split_a_session():
    """A big session must not be carved up to balance the proportions."""
    rows = rows_for(["SYNTH-01"], 40) + rows_for(["SYNTH-02", "SYNTH-03", "SYNTH-04"], 2)
    assigned = assign_splits(rows)
    for session in FOUR:
        assert len({r["split"] for r in assigned if r["session_id"] == session}) == 1


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #

def test_the_split_is_deterministic():
    a = sessions_by_split(assign_splits(rows_for(FOUR)))
    b = sessions_by_split(assign_splits(rows_for(FOUR)))
    assert a == b


def test_row_order_does_not_change_the_split():
    forward = sessions_by_split(assign_splits(rows_for(FOUR)))
    backward = sessions_by_split(assign_splits(list(reversed(rows_for(FOUR)))))
    assert forward == backward


def test_the_seed_is_forty_two():
    assert SEED == 42


def test_a_different_seed_is_allowed_but_not_the_default():
    a = sessions_by_split(assign_splits(rows_for(FOUR)))
    b = sessions_by_split(assign_splits(rows_for(FOUR), seed=7))
    assert a != b or len(FOUR) < 4      # different seed may coincide; assert it is honoured


# --------------------------------------------------------------------------- #
# refusals
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n", [1, 2])
def test_fewer_than_three_sessions_is_refused(n):
    with pytest.raises(SplitError, match="session"):
        assign_splits(rows_for(FOUR[:n]))


def test_no_rows_is_refused():
    """Zero sessions is the empty-dataset case and says so, rather than counting groups."""
    with pytest.raises(SplitError, match="no rows"):
        assign_splits([])


# --------------------------------------------------------------------------- #
# the files on disk
# --------------------------------------------------------------------------- #

def test_processed_images_are_moved_into_their_split(tmp_path):
    data_root = tmp_path / "data"
    rows = rows_for(FOUR)
    write_metadata(rows, data_root / "metadata.csv")

    for r in rows:
        p = data_root / "processed" / SPLIT_UNASSIGNED / str(r["freq_hz"])
        p.mkdir(parents=True, exist_ok=True)
        (p / f"{r['image_id'].rsplit('_', 2)[0]}_{r['image_id'].split('_')[-1]}.png").touch()

    result = split_dataset(data_root)

    assert not list((data_root / "processed" / SPLIT_UNASSIGNED).rglob("*.png"))
    for split in SPLIT_VALUES:
        moved = list((data_root / "processed" / split).rglob("*.png"))
        assert moved, split
    assert result["moved"] == len(rows)


def test_split_column_is_written_back(tmp_path):
    from metadata import read_metadata

    data_root = tmp_path / "data"
    write_metadata(rows_for(FOUR), data_root / "metadata.csv")
    split_dataset(data_root)

    rows = read_metadata(data_root / "metadata.csv")
    assert {r["split"] for r in rows} == set(SPLIT_VALUES)


def test_running_split_twice_is_idempotent(tmp_path):
    from metadata import read_metadata

    data_root = tmp_path / "data"
    write_metadata(rows_for(FOUR), data_root / "metadata.csv")
    split_dataset(data_root)
    once = read_metadata(data_root / "metadata.csv")

    split_dataset(data_root)
    assert read_metadata(data_root / "metadata.csv") == once
