"""A capture session in progress — `docs/05-webapp-spec.md` §9, steps 1 to 5.

**Why this lives outside `data/`.** §9: "Nothing is written until step 5. Uploading is not
committing. A half-finished session leaves no trace in `data/`." A draft directory under
`data/` would make that a promise kept by careful cleanup — one `except` away from being
false. `drafts/<session_id>/` makes it a property of the layout: abandoning a draft cannot
leave a trace in `data/` because nothing was ever written there.

It is the repo root and not the system temp directory on purpose. A draft is a lab afternoon;
an orphan afternoon in `%TEMP%` that nobody can find is indistinguishable from a lost one. The
path is listed by `GET /api/drafts` and shown in the view.

It is on the same volume as `data/` on purpose too: committing a draft *moves* its `raw/` tree
rather than copying it, so six hundred 12 MP captures change directory in milliseconds instead
of being read and rewritten.

The session log is validated here by `ingest.read_session_log` — the function that will refuse
the session at commit time. Validating the form with a second, friendlier copy of those rules is
how a form comes to accept a session that ingest then rejects.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ingest import SESSION_LOG, read_session_log
from metadata import validate_session_id

#: Repo root / drafts. Ignored by git; never inside `data/`.
DRAFTS_ROOT = Path(__file__).resolve().parents[1] / "drafts"

DRAFT_LOG = "draft.json"

STATUS_NEW = "new"
STATUS_CALIBRATED = "calibrated"
STATUS_COMMITTING = "committing"


class DraftError(ValueError):
    """The draft cannot do that."""


class DraftConflict(DraftError):
    """The draft cannot do that *because of state the user can see and decide about*."""


@dataclass(frozen=True)
class Draft:
    draft_id: str
    root: Path
    log: dict[str, Any]

    @property
    def session_id(self) -> str:
        return str(self.log["session_id"])

    @property
    def status(self) -> str:
        return str(self.log.get("status", STATUS_NEW))

    @property
    def raw(self) -> Path:
        """Mirrors `data/raw/<session_id>/` exactly, so committing is a directory move."""
        return self.root / "raw"

    @property
    def session(self) -> dict[str, Any]:
        return json.loads((self.root / SESSION_LOG).read_text("utf-8"))


def create(drafts_root: Path | str, session_log: Mapping[str, Any]) -> Draft:
    """Start a draft. The session log is written first and validated immediately."""
    session_id = validate_session_id(str(session_log.get("session_id", "")))
    root = _within(drafts_root, session_id)
    if root.exists():
        raise DraftConflict(
            f"a draft for session {session_id} is already open at {root} — open it or "
            f"discard it, rather than starting a second one for the same session"
        )

    root.mkdir(parents=True)
    (root / SESSION_LOG).write_text(
        json.dumps(dict(session_log), indent=2, ensure_ascii=False), "utf-8"
    )
    try:
        read_session_log(root, session_id)
    except Exception:
        # A draft that ingest would refuse is not a draft. Leave nothing behind.
        shutil.rmtree(root, ignore_errors=True)
        raise

    now = _stamp()
    return save(
        Draft(
            session_id,
            root,
            {
                "draft_id": session_id,
                "session_id": session_id,
                "status": STATUS_NEW,
                "created": now,
                "updated": now,
                "calibration": None,
            },
        )
    )


def save(draft: Draft) -> Draft:
    draft.log["updated"] = _stamp()
    (draft.root / DRAFT_LOG).write_text(
        json.dumps(draft.log, indent=2, ensure_ascii=False), "utf-8"
    )
    return draft


def listing(drafts_root: Path | str) -> list[Draft]:
    root = Path(drafts_root)
    if not root.is_dir():
        return []
    found = []
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if (child / DRAFT_LOG).is_file():
            found.append(read(root, child.name))
    return found


def read(drafts_root: Path | str, draft_id: str) -> Draft:
    root = _within(drafts_root, draft_id)
    path = root / DRAFT_LOG
    if not path.is_file():
        raise DraftError(f"no draft at {root}")
    return Draft(root.name, root, json.loads(path.read_text("utf-8")))


def discard(drafts_root: Path | str, draft_id: str) -> None:
    """Delete the draft and everything in it.

    `data/` is not touched, because nothing of this draft was ever in `data/`.
    """
    shutil.rmtree(_within(drafts_root, draft_id), ignore_errors=True)


def _within(drafts_root: Path | str, draft_id: str) -> Path:
    """The draft's directory, checked to be inside `drafts_root`. The boundary, said out loud.

    `validate_session_id` already refuses `../escape`, and today that is enough — but it refuses
    it for being a malformed *session id*, not for being a path. The day someone wants a
    friendlier identifier they will relax that format rule, and they will have no way to know
    they were also holding a path boundary shut. Three lines here keep the two intentions
    separable, so the next person cannot break one by touching the other.
    """
    root = Path(drafts_root).resolve()
    candidate = (root / str(draft_id)).resolve()
    if candidate != root and root not in candidate.parents:
        raise DraftError(f"{draft_id!r} does not name a directory inside {root}")
    return candidate


def _stamp() -> str:
    # Local time on purpose: a draft is a lab afternoon and the view shows it back to the
    # person who was in the room, the same way `runs.py` stamps a run directory.
    return datetime.now().isoformat(timespec="seconds")  # noqa: DTZ005
