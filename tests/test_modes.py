"""Contract tests for `src/modes.py` — `data/modes.csv` -> `config/classes.json`.

`docs/02` §3 step 4: keep only modes whose neighbours are >= 40 Hz away, so the classes
are physically separable. The rule is symmetric — if two modes are 25 Hz apart then
*both* are unusable, not just the higher one — and these tests pin that, because
dropping only one of the pair leaves a class the model cannot possibly separate.

The other thing pinned here is that the class index is frozen by frequency order. The
index is what a trained checkpoint stores; if it moved, an old checkpoint would answer
with a different frequency and nothing would report an error.
"""

from __future__ import annotations

import csv
import json

import pytest

from modes import (
    MIN_SEPARATION_HZ,
    ModesError,
    build_classes,
    read_modes,
    write_classes,
)

FIELDS = ["freq_hz", "width_hz", "n_nodal_lines", "symmetry", "quality", "notes"]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({**{k: "" for k in FIELDS}, **r})
    return path


def mode(freq, **kw):
    return {"freq_hz": freq, "width_hz": 10, "n_nodal_lines": 4,
            "symmetry": "diagonal", "quality": "clear", **kw}


# --------------------------------------------------------------------------- #
# the 40 Hz separation rule
# --------------------------------------------------------------------------- #

def test_well_separated_modes_are_all_kept(tmp_path):
    rows = [mode(f) for f in (400, 600, 900, 1300)]
    result = build_classes(read_modes(write_csv(tmp_path / "modes.csv", rows)))
    assert [c["freq_hz"] for c in result["classes"]] == [400, 600, 900, 1300]
    assert result["rejected"] == []


def test_a_close_pair_drops_both_members(tmp_path):
    """The rule is symmetric. Dropping only the higher one would keep a class whose
    neighbour is 25 Hz away — exactly what §3 forbids."""
    rows = [mode(f) for f in (400, 600, 2760, 2785)]
    result = build_classes(read_modes(write_csv(tmp_path / "modes.csv", rows)))
    assert [c["freq_hz"] for c in result["classes"]] == [400, 600]
    assert sorted(r["freq_hz"] for r in result["rejected"]) == [2760, 2785]


def test_rejection_records_the_offending_distance(tmp_path):
    rows = [mode(f) for f in (400, 600, 2760, 2785)]
    result = build_classes(read_modes(write_csv(tmp_path / "modes.csv", rows)))
    for r in result["rejected"]:
        assert r["nearest_neighbour_hz"] == 25
        assert "40" in r["reason"]


def test_exactly_forty_hz_apart_is_kept(tmp_path):
    """The spec says '>= 40 Hz away', so 40 is inside the rule, not outside it."""
    rows = [mode(f) for f in (400, 440)]
    result = build_classes(read_modes(write_csv(tmp_path / "modes.csv", rows)))
    assert [c["freq_hz"] for c in result["classes"]] == [400, 440]


def test_thirty_nine_hz_apart_is_rejected(tmp_path):
    rows = [mode(f) for f in (400, 439)]
    result = build_classes(read_modes(write_csv(tmp_path / "modes.csv", rows)))
    assert result["classes"] == []


def test_a_chain_of_close_modes_drops_entirely(tmp_path):
    rows = [mode(f) for f in (400, 420, 440, 900)]
    result = build_classes(read_modes(write_csv(tmp_path / "modes.csv", rows)))
    assert [c["freq_hz"] for c in result["classes"]] == [900]


# --------------------------------------------------------------------------- #
# the frozen class index
# --------------------------------------------------------------------------- #

def test_classes_are_sorted_ascending_with_dense_indices(tmp_path):
    rows = [mode(f) for f in (1300, 400, 900, 600)]        # deliberately unsorted input
    result = build_classes(read_modes(write_csv(tmp_path / "modes.csv", rows)))
    assert [c["freq_hz"] for c in result["classes"]] == [400, 600, 900, 1300]
    assert [c["index"] for c in result["classes"]] == [0, 1, 2, 3]


def test_index_order_does_not_depend_on_row_order(tmp_path):
    a = build_classes(read_modes(write_csv(tmp_path / "a.csv", [mode(f) for f in (900, 400, 600)])))
    b = build_classes(read_modes(write_csv(tmp_path / "b.csv", [mode(f) for f in (400, 600, 900)])))
    assert a["classes"] == b["classes"]


def test_written_file_declares_the_index_is_frozen(tmp_path):
    """A reader of the artefact must not have to consult the source to learn this."""
    result = build_classes(read_modes(write_csv(tmp_path / "m.csv", [mode(f) for f in (400, 600)])))
    path = write_classes(result, tmp_path / "classes.json")
    text = path.read_text("utf-8")
    assert "frozen" in text.lower()

    written = json.loads(text)
    assert written["min_separation_hz"] == MIN_SEPARATION_HZ
    assert written["n_classes"] == 2


# --------------------------------------------------------------------------- #
# malformed input
# --------------------------------------------------------------------------- #

def test_missing_column_is_refused(tmp_path):
    p = tmp_path / "modes.csv"
    p.write_text("freq_hz,width_hz\n400,10\n", encoding="utf-8")
    with pytest.raises(ModesError, match="column"):
        read_modes(p)


def test_duplicate_frequency_is_refused(tmp_path):
    p = write_csv(tmp_path / "modes.csv", [mode(400), mode(400)])
    with pytest.raises(ModesError, match="duplicate"):
        read_modes(p)


def test_non_integer_frequency_is_refused(tmp_path):
    p = write_csv(tmp_path / "modes.csv", [mode("four hundred")])
    with pytest.raises(ModesError, match="freq_hz"):
        read_modes(p)


def test_empty_file_is_refused(tmp_path):
    p = write_csv(tmp_path / "modes.csv", [])
    with pytest.raises(ModesError, match="no modes"):
        read_modes(p)


def test_missing_file_is_refused(tmp_path):
    with pytest.raises(ModesError, match="not found"):
        read_modes(tmp_path / "nope.csv")


# --------------------------------------------------------------------------- #
# the F1 phase gate
# --------------------------------------------------------------------------- #

def test_thin_mode_list_is_flagged_but_not_fatal(tmp_path):
    """`docs/00` F1 wants >= 10 modes. That is a phase gate a human clears, not a
    reason for this target to refuse to write a file — the synthetic sessions run on
    six classes on purpose."""
    result = build_classes(read_modes(write_csv(tmp_path / "m.csv", [mode(f) for f in (400, 600)])))
    assert result["warnings"]
    assert any("10" in w for w in result["warnings"])


def test_a_full_mode_list_raises_no_warning(tmp_path):
    rows = [mode(400 + 100 * i) for i in range(12)]
    result = build_classes(read_modes(write_csv(tmp_path / "m.csv", rows)))
    assert result["warnings"] == []
