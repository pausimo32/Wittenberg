"""`/dades` — the routes that put a labelled session into the dataset. `docs/05` §9.

This is the **only** place in the app where a browser causes a write to `data/`, which is why
it is a module of its own rather than more routes in `src/api.py`: §3's serving half still
opens no file after startup, and keeping the two halves apart is what keeps that readable.

Three rules shape every route here:

- **The gate is `ingest.gate_frame`.** Not "the same thresholds as" — the same call. §9: "a
  discrepancy here would let the UI accept images the pipeline rejects."
- **The commit is `ingest.ingest_session`.** The route moves files into place and then calls
  the function `python tasks.py ingest` calls, with the progress hook Task 1 added.
- **No Catalan.** Refusals carry a `code`; the sentence the user reads is chosen in
  `web/src/strings.ts`. A Catalan string in a traceback helps nobody.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

import drafts as drafts_module
from ingest import IngestError
from metadata import MixedProvenanceError


class DatasetState(NamedTuple):
    """The two roots and the three configs every route here works against.

    They are parameters rather than module constants because the whole point of a test that
    exercises these routes is that it does so against a `data/` and a `drafts/` under
    `tmp_path`. A module-level `DATA_ROOT` read inside a handler would make "no test writes to
    the real dataset" a habit instead of a property.
    """

    data_root: Path
    drafts_root: Path
    calibration_config: dict[str, Any]
    preprocess_config: dict[str, Any]
    gate: dict[str, Any]


def conflict(code: str, message: str, status: int = 409) -> HTTPException:
    """A refusal the frontend can act on.

    `code` is the machine-readable half and is the only part that may be branched on; `message`
    is English, for the log and the traceback. The Catalan sentence the user reads is chosen
    from `code` in `web/src/strings.ts` — see the module docstring.
    """
    return HTTPException(status_code=status, detail={"code": code, "message": message})


def dataset_router(state: DatasetState) -> APIRouter:
    router = APIRouter(prefix="/api")

    #: `draft_id -> CommitJob`, once Task 9 writes one. Until then nothing puts anything in
    #: here, and the only reader treats a missing entry as "no commit is running".
    jobs_registry: dict[str, Any] = {}

    def load(draft_id: str) -> drafts_module.Draft:
        """The draft, or a 404 — for a bad id and a missing directory alike.

        `_within` raises `DraftError` for an id that would escape the drafts root, and that is
        deliberately the same answer as "no such draft": whether a path outside the root holds
        something is not a question this API answers.
        """
        try:
            return drafts_module.read(state.drafts_root, draft_id)
        except (drafts_module.DraftError, ValueError) as exc:
            raise HTTPException(
                status_code=404, detail={"code": "no_draft", "message": str(exc)}
            ) from exc

    @router.get("/drafts")
    def list_drafts() -> dict[str, Any]:
        return {"drafts": [_summary(d) for d in drafts_module.listing(state.drafts_root)]}

    @router.post("/drafts")
    def new_draft(session_log: dict[str, Any]) -> dict[str, Any]:
        """Step 1 — the session form becomes a draft directory.

        Provenance is checked *before* anything is created. A `SYNTH-*` session cannot join a
        dataset of real captures and never will, so finding that out now costs the user a
        refused form; finding it out at commit costs them six hundred uploads.
        """
        try:
            drafts_module.check_provenance_for(
                state.data_root, str(session_log.get("session_id", ""))
            )
        except MixedProvenanceError as exc:
            raise conflict("mixed_provenance", str(exc)) from exc

        try:
            draft = drafts_module.create(state.drafts_root, session_log)
        except drafts_module.DraftConflict as exc:
            raise conflict("draft_exists", str(exc)) from exc
        except (IngestError, ValueError) as exc:
            # `read_session_log`'s missing key and `validate_session_id`'s malformed id are
            # the same thing to the form: a session log ingest would refuse.
            raise conflict("invalid_session_log", str(exc), status=422) from exc
        return _detail(draft, state)

    @router.get("/drafts/{draft_id}")
    def get_draft(draft_id: str) -> dict[str, Any]:
        return _detail(load(draft_id), state)

    @router.delete("/drafts/{draft_id}", status_code=204)
    def delete_draft(draft_id: str) -> Response:
        load(draft_id)
        running = jobs_registry.get(draft_id)
        if running is not None and running.status == "running":
            raise conflict("commit_running", "this draft is being committed right now")
        # `data_root` so a staged draft's captures go with it — see `drafts.discard`.
        drafts_module.discard(state.drafts_root, draft_id, state.data_root)
        jobs_registry.pop(draft_id, None)
        return Response(status_code=204)

    return router


def _summary(draft: drafts_module.Draft) -> dict[str, Any]:
    records = drafts_module.frame_records(draft)
    return {
        "draft_id": draft.draft_id,
        "session_id": draft.session_id,
        "status": draft.status,
        # The native path, backslashes and all: it is for pasting into a file manager, not
        # for building a URL.
        "path": str(draft.root),
        "created": draft.log.get("created"),
        "updated": draft.log.get("updated"),
        "calibrated": drafts_module.is_calibrated(draft),
        "classes": sorted({r["freq_hz"] for r in records}),
        "total": len(records),
        # Three categories, not two. `accepted` is the gate saying yes; `rejected` is the gate
        # saying no and why; `duplicates` never reached the gate because the draft already had
        # those bytes. Folding duplicates into `rejected` would make a retried upload look like
        # a lighting problem.
        "accepted": sum(1 for r in records if r["quality"] == "ok"),
        "rejected": sum(1 for r in records if r["reason"] is not None),
        "pending": sum(1 for r in records if r["quality"] == drafts_module.QUALITY_PENDING),
        "duplicates": len(drafts_module.duplicate_records(draft)),
        # Three open drafts are a few gigabytes, and whoever has to delete one wants to know
        # which. A walk of the draft directory, not a cached number: a cached one goes stale on
        # the first removal and then lies quietly.
        "bytes": drafts_module.size_on_disk(draft),
    }


def _detail(draft: drafts_module.Draft, state: DatasetState) -> dict[str, Any]:
    by_class: dict[str, list[dict[str, Any]]] = {}
    for record in drafts_module.frame_records(draft):
        by_class.setdefault(str(record["freq_hz"]), []).append(record)
    return {
        **_summary(draft),
        "session": draft.session,
        "calibration": drafts_module.calibration(draft),
        "frames": by_class,
        # What committing would replace. Sent with every read of the draft so the view can
        # warn before the button is pressed rather than after — `commit_guard` refuses either
        # way, but a refusal at the end of an afternoon is a worse way to learn it.
        "existing_rows": drafts_module.existing_rows(state.data_root, draft.session_id),
    }
