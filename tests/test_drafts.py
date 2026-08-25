"""F6 — the draft store. `docs/05-webapp-spec.md` §9.

A draft lives at `drafts/<session_id>/`, outside `data/`, and that is the whole design: "a
half-finished session leaves no trace in `data/`" is then true because nothing was ever written
there, not because something was cleaned up afterwards.
"""

from __future__ import annotations

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
