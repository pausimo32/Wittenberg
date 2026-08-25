"""Contract tests for `tools/synth_capture.py` — the condition balance, mainly.

The generator's job is to produce sessions that exercise the pipeline *without* baking
in an artefact that the pipeline would then be judged on. The one that matters is
condition balance: the split assigns whole sessions, so any condition occurring in a
single session predicts the split perfectly, and a model tested on that session would be
scored on generalising to an unseen lamp angle or an unseen grain rather than to an
unseen plate.

That failure has already happened twice here — once for `lighting` and `dwell_s`, once
for `material` — so the property is pinned rather than left to the table's author.
"""

from __future__ import annotations

import sys
from collections import Counter
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from synth_capture import (
    BALANCED_CONDITIONS,
    DEFECTS,
    SESSION_CONDITIONS,
    build_modes,
    defect_for,
    render_frame,
    usable_frequencies,
)

N_SESSIONS = len(SESSION_CONDITIONS)


# --------------------------------------------------------------------------- #
# condition balance — docs/02 §2
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("condition", BALANCED_CONDITIONS)
def test_every_value_appears_in_exactly_two_sessions(condition):
    counts = Counter(s[condition] for s in SESSION_CONDITIONS)
    assert set(counts.values()) == {2}, f"{condition} is unbalanced: {dict(counts)}"


@pytest.mark.parametrize("condition", BALANCED_CONDITIONS)
def test_no_condition_value_is_unique_to_one_session(condition):
    """A value in one session only is a value that perfectly predicts the split."""
    counts = Counter(s[condition] for s in SESSION_CONDITIONS)
    assert not [v for v, n in counts.items() if n == 1], dict(counts)


def test_material_and_lighting_form_a_full_factorial():
    """Two materials x two lightings, one session per cell — no cell empty, none doubled."""
    cells = Counter((s["material"], s["lighting"]) for s in SESSION_CONDITIONS)
    assert len(cells) == N_SESSIONS
    assert set(cells.values()) == {1}


def test_holding_out_any_single_session_leaves_its_conditions_represented():
    """The property the balance exists for, checked directly for every possible holdout."""
    for held_out in range(N_SESSIONS):
        rest = [s for i, s in enumerate(SESSION_CONDITIONS) if i != held_out]
        for condition in BALANCED_CONDITIONS:
            value = SESSION_CONDITIONS[held_out][condition]
            assert any(s[condition] == value for s in rest), (
                f"holding out session {held_out + 1} removes the only {condition}={value}"
            )


def test_lighting_never_varies_within_a_session():
    """docs/02 §2 — one lighting per session, varied only between them."""
    for s in SESSION_CONDITIONS:
        assert isinstance(s["lighting"], str)


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #

def test_generated_modes_include_a_pair_the_forty_hz_rule_must_reject():
    modes = build_modes(6)
    usable = usable_frequencies(modes)
    assert len(usable) == 6
    assert len(modes) > len(usable), "nothing for `tasks.py modes` to reject"


def test_usable_modes_are_all_at_least_forty_hz_apart():
    usable = usable_frequencies(build_modes(6))
    gaps = [b - a for a, b in pairwise(usable)]
    assert all(g >= 40 for g in gaps), gaps


# --------------------------------------------------------------------------- #
# defects — one per §6 rejection reason
# --------------------------------------------------------------------------- #

def test_every_gate_reason_has_a_defect_that_triggers_it():
    """A yield report is only meaningful if all four branches of §6 are exercised."""
    names = {name for name, _period in DEFECTS}
    assert names == {"blurred", "exposure", "partial", "unsettled"}


def test_defects_are_deterministic_by_frame_index():
    assert [defect_for(i) for i in range(60)] == [defect_for(i) for i in range(60)]


def test_most_frames_are_clean():
    """The generator must not reject so much that the accepted set is unrepresentative."""
    spoilt = sum(defect_for(i) is not None for i in range(120))
    assert spoilt / 120 < 0.35


# --------------------------------------------------------------------------- #
# frames
# --------------------------------------------------------------------------- #

def test_render_is_deterministic_for_a_given_seed():
    a = render_frame(1, 2, np.random.default_rng(42), "lamp_diffuse")
    b = render_frame(1, 2, np.random.default_rng(42), "lamp_diffuse")
    assert np.array_equal(a, b)


def test_different_modes_render_differently():
    rng = np.random.default_rng(1)
    a = render_frame(1, 2, rng, "lamp_diffuse")
    b = render_frame(3, 4, np.random.default_rng(1), "lamp_diffuse")
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("n_markers", [0, 3, 4])
def test_marker_count_is_honoured(n_markers):
    from calibrate import detect_markers

    frame = render_frame(1, 2, np.random.default_rng(3), "lamp_diffuse", n_markers=n_markers)
    assert len(detect_markers(frame)) == n_markers
