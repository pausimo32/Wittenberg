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

import hashlib
import json
import os
import shutil
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from calibrate import calibrate, write_calibration
from imageio import SUPPORTED_SUFFIXES, thumbnail
from ingest import SESSION_LOG, read_session_log
from metadata import check_provenance, read_metadata, validate_session_id

#: Repo root / drafts. Ignored by git; never inside `data/`.
DRAFTS_ROOT = Path(__file__).resolve().parents[1] / "drafts"

DRAFT_LOG = "draft.json"

#: The session's geometry, under the fixed name `ingest._session_calibration` looks for
#: inside a session directory. `write_calibration` names its file after the session; a draft
#: keeps it under this name instead so that promoting the directory into `data/raw/` needs no
#: rename, and nothing downstream has to reconstruct a session id from a path.
CALIB_LOG = "calibration.json"

FRAMES_LOG = "frames.json"
DUPLICATES_LOG = "duplicates.json"

#: Rejection thumbnails, not a gallery: they exist so someone can see *why* photo three
#: failed without leaving the view. 320 px is enough to see that the plate is out of frame
#: or that the pattern is smeared, and six hundred of them cost nothing.
PREVIEW_MAX_SIDE = 320

#: A frame whose bytes are on disk but whose verdict has not come back yet. A real state,
#: not a placeholder: the upload and the gate are two steps of one request, and a frame
#: that never reached the second one must be visible as something rather than as nothing.
QUALITY_PENDING = "pending"

STATUS_NEW = "new"
STATUS_CALIBRATED = "calibrated"
STATUS_COMMITTING = "committing"


class DraftError(ValueError):
    """The draft cannot do that."""


class DraftConflict(DraftError):
    """The draft cannot do that *because of state the user can see and decide about*."""


class DraftDuplicate(DraftConflict):
    """This frame is already in the draft."""


class DuplicateName(DraftDuplicate):
    """Same filename, same class — the folder was dropped twice."""


class DuplicateContent(DraftDuplicate):
    """Same bytes as a frame already in the draft, under any name, in any class."""


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


def discard(
    drafts_root: Path | str, draft_id: str, data_root: Path | str | None = None
) -> None:
    """Delete the draft and everything in it.

    `data/` is not touched, because nothing of this draft was ever in `data/` — unless it was
    staged, in which case its captures *are* under `data/raw/.incoming-<id>` and go with it.
    Discarding a staged draft is how someone gives up on a session whose ingest kept failing,
    and leaving its captures behind would leave exactly the third thing the staging name exists
    to prevent.
    """
    root = _within(drafts_root, draft_id)
    if data_root is not None and (root / DRAFT_LOG).is_file():
        log = json.loads((root / DRAFT_LOG).read_text('utf-8'))
        if log.get('staged'):
            shutil.rmtree(staging_dir(data_root, str(log['session_id'])), ignore_errors=True)
    shutil.rmtree(root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# into data/ — either the draft exists or the session exists
# --------------------------------------------------------------------------- #

def existing_rows(data_root: Path | str, session_id: str) -> int:
    path = Path(data_root) / "metadata.csv"
    if not path.is_file():
        return 0
    return sum(1 for r in read_metadata(path) if r["session_id"] == session_id)


def check_provenance_for(data_root: Path | str, session_id: str) -> None:
    """`metadata.check_provenance`, at the door rather than at the commit.

    Finding out that a `SYNTH-*` session cannot join this dataset before uploading six hundred
    frames is the whole reason this is not left to `ingest_session`.
    """
    path = Path(data_root) / "metadata.csv"
    check_provenance(read_metadata(path) if path.is_file() else [], session_id)


def commit_guard(draft: Draft, data_root: Path | str, replace: bool) -> None:
    """Everything that must be true before a byte of `data/` changes. Raises, never warns.

    `ingest_session` is idempotent, and that is the right default for a command line where
    re-running is what you meant. From a browser it is not: an id typed one digit wrong lands on
    a session someone else captured, and idempotent then means the old rows are gone without
    anyone being asked. So the replacement is offered, not performed.
    """
    root = Path(data_root)
    check_provenance_for(root, draft.session_id)

    if not frame_records(draft):
        raise DraftConflict("this draft has no frames yet")
    if not is_calibrated(draft):
        raise DraftConflict("this draft has no validated calibration")

    rows = existing_rows(root, draft.session_id)
    if rows and not replace:
        raise DraftConflict(
            f"session {draft.session_id} already has {rows} row(s) in metadata.csv. "
            f"Committing replaces every one of them, and the captures under "
            f"data/raw/{draft.session_id} as well. Confirm the replacement or use a "
            f"different session id."
        )


def staging_dir(data_root: Path | str, session_id: str) -> Path:
    """Where a session's captures live between the draft and the session.

    A dotted name under `data/raw/`, so it is on the same volume as its destination and the last
    step is a rename rather than a copy. Ingest is handed this path explicitly, so nothing
    globs its way into it, and `.gitignore` already excludes all of `data/raw/`.
    """
    return Path(data_root) / "raw" / f".incoming-{session_id}"


def stage(draft: Draft, data_root: Path | str) -> Path:
    """Move the draft's captures under the staging name, with everything ingest will need.

    A move, not a copy: the draft is on the same volume as `data/`, so six hundred 12 MP
    captures change directory in milliseconds. On Windows a cross-volume `os.rename` fails
    outright, which is the second reason the drafts directory sits beside `data/` and not in
    `%TEMP%`.

    The `session.json` copied here is the exact file `create` validated, and `calibration.json`
    travels *inside* the staging directory rather than going to `data/calib/`: that directory is
    committed to git, and an orphan calibration for a session whose ingest then failed would be
    a file in the repository describing something that does not exist.
    """
    root = Path(data_root)
    staged = staging_dir(root, draft.session_id)
    if staged.exists():
        # A previous attempt that failed *before* moving anything. Its captures are the ones
        # about to be re-staged, so this is not a loss — `dataset_api.commit` resumes from the
        # staging directory instead of calling this when the draft is already staged.
        shutil.rmtree(staged)
    staged.parent.mkdir(parents=True, exist_ok=True)

    draft.log["status"] = STATUS_COMMITTING
    draft.log["staged"] = True
    save(draft)

    shutil.move(str(draft.raw), str(staged))
    shutil.copyfile(draft.root / SESSION_LOG, staged / SESSION_LOG)
    shutil.copyfile(draft.root / CALIB_LOG, staged / CALIB_LOG)
    return staged


def finish(draft: Draft, data_root: Path | str) -> Path:
    """Rename the staged captures to the session and place the calibration.

    Called only after `ingest_session` has returned. This is the instant the session comes into
    being: before it, `data/raw/<session_id>` and `data/calib/<session_id>.json` do not exist,
    so a failed ingest leaves a draft and not a half-session nobody can name.

    `os.replace` rather than `shutil.move`, because the destination is guaranteed absent by
    `commit_guard` — or emptied here under a confirmed replacement — and a rename within one
    directory is the closest thing to an instant this can be.

    One window remains and is not papered over: `ingest_session` writes `metadata.csv` as its
    last act, so a crash between that write and this rename leaves rows whose captures are still
    under the staging name. Re-running the commit converges, because `_clear_session_processed`
    and `replace_session_rows` both key on the session id. That is the recovery, not a claim
    that the window does not exist.
    """
    root = Path(data_root)
    staged = staging_dir(root, draft.session_id)
    session_dir = root / "raw" / draft.session_id
    if session_dir.exists():
        # Only reachable once `commit_guard(replace=True)` has passed.
        shutil.rmtree(session_dir)

    os.replace(staged, session_dir)

    calib_dir = root / "calib"
    calib_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(session_dir / CALIB_LOG, calib_dir / f"{draft.session_id}.json")

    draft.log["staged"] = False
    save(draft)
    return session_dir


# --------------------------------------------------------------------------- #
# the calibration
# --------------------------------------------------------------------------- #


def store_calibration(
    draft: Draft, image_rgba: np.ndarray, cfg: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Detect the markers in the calibration frame and, if there are four, keep it.

    `/dades` only ever produces `homography` sessions. `center_crop` is what ingest falls back
    to for a session that was captured without a marker board — a fact about a past afternoon,
    not something to offer as an option while the rig is still set up in front of you.

    A detection that finds three markers calibrates nothing: there is no quad, so the plate is
    not rectified, and `docs/02` §6 has every subsequent frame rejected as `partial` anyway. It
    is still recorded — as a failed attempt, with its `n_markers` — because that count is the
    only thing that tells the person at the rig whether to move the board or the camera.
    """
    calib = calibrate(image_rgba, draft.session_id, cfg)
    info = {
        "ok": calib.corners is not None,
        "roi_mode": calib.roi_mode,
        "n_markers": calib.n_markers,
        "marker_ids": calib.marker_ids,
        "corners": calib.corners,
        "px_per_mm": calib.px_per_mm,
        "flags": calib.flags,
    }
    if info["ok"]:
        # `write_calibration` writes `<session_id>.json`, the name `data/calib/` stores. A
        # session id is `YYYYMMDD-NN` or `SYNTH-YYYYMMDD-NN`, so that is never `CALIB_LOG`
        # already: the file is always moved onto the fixed name the draft promises.
        Path(write_calibration(calib, draft.root)).replace(draft.root / CALIB_LOG)
        draft.log["status"] = STATUS_CALIBRATED

    draft.log["calibration"] = info
    save(draft)
    return info


def calibration(draft: Draft) -> dict[str, Any] | None:
    """The last attempt, whether it succeeded or not. `None` before the first one."""
    return draft.log.get("calibration")


def is_calibrated(draft: Draft) -> bool:
    """Four markers were found and their quad is on disk — the door `POST /frames` holds shut."""
    info = draft.log.get("calibration")
    return bool(info and info.get("ok"))


# --------------------------------------------------------------------------- #
# the frames
# --------------------------------------------------------------------------- #

#: `frames.json` is a read-modify-write and the upload route runs in Starlette's thread pool
#: with three uploads in flight. Without this, two frames stored in the same instant read the
#: same log and the second write drops the first — silently, and in exactly the situation the
#: whole view exists for. One process, one lock; it is held for a file write and a JSON dump,
#: never across the 560 ms decode.
_LOG_LOCK = threading.Lock()


def store_frame(draft: Draft, freq_hz: int, filename: str, data: bytes) -> dict[str, Any]:
    """Write one upload to the draft, **byte for byte**.

    The bytes are not decoded, re-encoded, rotated, stripped of EXIF or resized on the way in.
    `docs/02` §4 calls `data/raw/` "untouched, straight off the phone", and a draft is what
    becomes `data/raw/`. Every preview this module writes is generated from the file *after*
    it has been written, so the picture on screen can never be evidence about bytes that were
    not stored.

    The stored name is `IMG_<nnnn><ext>` by arrival order within the class, because ingest
    derives `image_id` from the trailing number of the stem (`docs/02` §4) and two phones will
    happily both produce `IMG_0001`. The name the file arrived with is kept in the record.

    **The same bytes are refused, whatever they are called.** Six hundred uploads over a
    connection that stumbles means someone retries a batch. A repeated frame trips no threshold
    and quietly unbalances the classes, and with a burst whose frames are already near-identical
    to each other, nobody is going to catch it by looking. The SHA-256 is being computed for the
    integrity check anyway, so this costs nothing: the upload is recorded as a duplicate — a
    third category, neither accepted nor rejected by quality — and the bytes are not stored.

    The record is written here rather than when the verdict comes back, because the index the
    frame is numbered with and the hash it is checked against are both read from a log that the
    two uploads running beside it have not written to yet.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise DraftError(
            f"{filename}: suffix {suffix!r} is not one of {', '.join(SUPPORTED_SUFFIXES)}"
        )

    freq = int(freq_hz)
    digest = hashlib.sha256(data).hexdigest()

    with _LOG_LOCK:
        log = _frames_log(draft)
        entries = log.setdefault(str(freq), [])

        if any(r["original_filename"] == filename for r in entries):
            raise DuplicateName(
                f"{filename} is already in the {freq} Hz class of this draft — the folder was "
                f"probably dropped twice"
            )

        # Draft-wide, not class-wide: one photograph is one frequency, so the same capture
        # appearing under a second label is a mistake wherever it lands.
        for other in (r for rows in log.values() for r in rows):
            if other["sha256"] == digest:
                _record_duplicate(draft, filename, freq, digest, other)
                raise DuplicateContent(
                    f"{filename} has the same bytes as {other['name']} "
                    f"({other['original_filename']}, {other['freq_hz']} Hz) — it is the same "
                    f"capture uploaded twice, and a repeated frame is not a second observation"
                )

        freq_dir = draft.raw / str(freq)
        freq_dir.mkdir(parents=True, exist_ok=True)

        # `len(entries) + 1` is not enough on its own: `remove_frame` leaves a hole in the
        # numbering on purpose — the names are the image_ids ingest will derive, and closing a
        # gap would change the identity of frames that were already gated.
        taken = {r["name"] for r in entries}
        index = len(entries) + 1
        while f"IMG_{index:04d}{suffix}" in taken:
            index += 1
        path = freq_dir / f"IMG_{index:04d}{suffix}"
        path.write_bytes(data)

        record = {
            "name": path.name,
            "original_filename": filename,
            "freq_hz": freq,
            "quality": QUALITY_PENDING,
            "reason": None,
            "metrics": {},
            "sha256": digest,
            "preview": False,
        }
        log[str(freq)] = sorted([*entries, record], key=lambda r: r["name"])
        _write_frames_log(draft, log)
        return dict(record)


def record_verdict(
    draft: Draft,
    record: Mapping[str, Any],
    quality: str,
    reason: str | None,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Fill in one live-gate verdict, so a reload does not re-decode six hundred photos.

    An update to the row `store_frame` already wrote, never an insert: the bytes and the verdict
    arrive at different moments and only one of them can be lost.
    """
    freq, name = str(int(record["freq_hz"])), str(record["name"])
    with _LOG_LOCK:
        log = _frames_log(draft)
        updated: dict[str, Any] | None = None
        for row in log.get(freq, []):
            if row["name"] == name:
                row.update(
                    quality=quality,
                    reason=reason,
                    metrics=dict(metrics),
                    preview=preview_path(draft, int(freq), name, "original").exists(),
                )
                updated = row
        if updated is None:
            raise DraftError(f"no frame {name} in the {freq} Hz class of {draft.draft_id}")
        _write_frames_log(draft, log)
        return dict(updated)


def frame_records(draft: Draft, freq_hz: int | None = None) -> list[dict[str, Any]]:
    log = _frames_log(draft)
    if freq_hz is not None:
        return list(log.get(str(int(freq_hz)), []))
    return [r for freq in sorted(log, key=int) for r in log[freq]]


def frame_path(draft: Draft, record: Mapping[str, Any]) -> Path:
    return draft.raw / str(int(record["freq_hz"])) / str(record["name"])


def duplicate_records(draft: Draft) -> list[dict[str, Any]]:
    """Uploads refused for being bytes the draft already has.

    Kept rather than discarded: a retried batch is a fact about the afternoon, and the count is
    what tells you it happened.
    """
    path = draft.root / DUPLICATES_LOG
    return json.loads(path.read_text("utf-8")) if path.is_file() else []


def write_previews(
    draft: Draft, path: Path, freq_hz: int, original: np.ndarray, edges: np.ndarray | None
) -> None:
    """Both previews of a rejected frame, from arrays decoded out of `path`.

    `edges` is `None` for a `partial` frame: without four markers there is no rectified plate,
    so there is no edge map that means anything. Showing one anyway would be showing a
    rectification that did not happen.
    """
    target = preview_path(draft, freq_hz, path.name, "original")
    target.parent.mkdir(parents=True, exist_ok=True)
    bgr = (
        cv2.cvtColor(original, cv2.COLOR_RGBA2BGR)
        if original.ndim == 3 and original.shape[2] == 4
        else original
    )
    cv2.imwrite(str(target), thumbnail(bgr, PREVIEW_MAX_SIDE))
    if edges is not None:
        cv2.imwrite(
            str(preview_path(draft, freq_hz, path.name, "edges")),
            thumbnail(edges, PREVIEW_MAX_SIDE),
        )


def preview_path(draft: Draft, freq_hz: int, name: str, kind: str) -> Path:
    return draft.root / "preview" / str(int(freq_hz)) / f"{Path(name).stem}.{kind}.png"


def remove_frame(draft: Draft, freq_hz: int, name: str) -> None:
    """Take one frame back out.

    The names of the frames after it are left alone — they are already in `frames.json`, and
    renumbering them would change their `image_id`s.
    """
    freq = str(int(freq_hz))
    (draft.raw / freq / Path(name).name).unlink(missing_ok=True)
    for kind in ("original", "edges"):
        preview_path(draft, freq_hz, name, kind).unlink(missing_ok=True)
    with _LOG_LOCK:
        log = _frames_log(draft)
        log[freq] = [r for r in log.get(freq, []) if r["name"] != name]
        _write_frames_log(draft, log)


def size_on_disk(draft: Draft) -> int:
    """Bytes under the draft directory.

    Walked, not cached: a cached number goes stale on the first removal and then lies quietly.
    Three open drafts are a few gigabytes, and whoever has to delete one wants to know which.
    """
    return sum(p.stat().st_size for p in draft.root.rglob("*") if p.is_file())


def _record_duplicate(
    draft: Draft, filename: str, freq_hz: int, digest: str, matches: Mapping[str, Any]
) -> None:
    """Called with `_LOG_LOCK` held."""
    path = draft.root / DUPLICATES_LOG
    entries = json.loads(path.read_text("utf-8")) if path.is_file() else []
    entries.append(
        {
            "original_filename": filename,
            "freq_hz": int(freq_hz),
            "sha256": digest,
            "matches": matches["name"],
            "matches_freq_hz": matches["freq_hz"],
        }
    )
    path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), "utf-8")


def _frames_log(draft: Draft) -> dict[str, list[dict[str, Any]]]:
    path = draft.root / FRAMES_LOG
    return json.loads(path.read_text("utf-8")) if path.is_file() else {}


def _write_frames_log(draft: Draft, log: Mapping[str, Any]) -> None:
    (draft.root / FRAMES_LOG).write_text(json.dumps(log, indent=2, ensure_ascii=False), "utf-8")


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
