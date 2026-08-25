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

import base64
from pathlib import Path
from typing import Annotated, Any, NamedTuple

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response

import drafts as drafts_module
from calibrate import ROI_MODE_HOMOGRAPHY
from imageio import UnsupportedImageError, decode_bytes, thumbnail
from ingest import IngestError, assess_frame, verdict_of
from metadata import MixedProvenanceError

#: The calibration preview is the one picture in this view someone squints at, so it gets
#: twice a rejection thumbnail's side and no more.
CALIBRATION_PREVIEW_MAX_SIDE = drafts_module.PREVIEW_MAX_SIDE * 2


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

    @router.post("/drafts/{draft_id}/calibration")
    def calibrate_draft(draft_id: str, file: Annotated[UploadFile, File()]) -> dict[str, Any]:
        """Step 2. A failed detection is a 200 with `ok: false`, not an error.

        Three markers found is information — it tells the person at the rig where to move the
        board — and the view needs the picture to act on it. An HTTP error would throw that
        picture away.

        Declared `def` and not `async def`, for the same reason the frames route is: decoding a
        12 MP capture and running the ArUco detector over it is most of a second of CPU, and on
        the event loop that is most of a second in which this process answers nothing at all
        (hard rule 2). Starlette runs a sync handler in a worker thread instead.

        Nothing is written under `data/`: the calibration lands inside the draft, and only the
        commit moves it. `docs/05` §9, "uploading is not committing".
        """
        draft = load(draft_id)
        data = file.file.read()
        if not data:
            raise conflict("empty_upload", "empty upload", status=400)
        try:
            decoded = decode_bytes(data, file.filename)
        except UnsupportedImageError as exc:
            raise conflict("unsupported_image", str(exc), status=415) from exc

        # The full-resolution array, every time: the geometry is measured on the pixels the
        # camera produced, and only the picture that illustrates it is ever made smaller.
        info = drafts_module.store_calibration(draft, decoded.pixels, state.calibration_config)
        return {"calibration": info, "preview": _calibration_preview(decoded.pixels, info)}

    @router.post("/drafts/{draft_id}/frames")
    def add_frame(
        draft_id: str,
        freq_hz: Annotated[int, Query(ge=1, description="the class label, in Hz")],
        file: Annotated[UploadFile, File()],
    ) -> dict[str, Any]:
        """Steps 3 and 4: store the bytes, then gate them.

        **The order is imposed here, not only in the view.** Without a validated calibration
        there is no 512 px ROI, so there is nothing for `quality.assess` to measure — and a
        rule that lives only in a React component is one component away from not being a rule.
        The refusal comes before the bytes are stored, so a draft whose owner was told no does
        not quietly acquire the frame anyway.

        **The gate is `ingest.assess_frame`.** Not the same thresholds as ingest — the same
        call, with the same config and the same session calibration. `docs/05` §9: "a
        discrepancy here would let the UI accept images the pipeline rejects", and the only way
        to make that impossible rather than unlikely is for there to be one function.

        Declared `def`, not `async def`, deliberately. Starlette then runs it in a worker
        thread, so a ~560 ms HEIC decode does not hold the event loop and the rest of the app
        keeps answering while a folder of sixty photos streams through (hard rule 2). The
        client uploads a few at a time; see `web/src/lib/upload.ts`.
        """
        draft = load(draft_id)
        if not drafts_module.is_calibrated(draft):
            raise conflict(
                "not_calibrated",
                "this draft has no validated calibration yet — the ROI the gate measures does "
                "not exist until four markers have been found",
            )

        data = file.file.read()
        if not data:
            raise conflict("empty_upload", "empty upload", status=400)

        try:
            record = drafts_module.store_frame(draft, freq_hz, file.filename or "", data)
        except drafts_module.DuplicateName as exc:
            raise conflict("duplicate_frame", str(exc)) from exc
        except drafts_module.DuplicateContent as exc:
            raise conflict("duplicate_content", str(exc)) from exc
        except drafts_module.DraftError as exc:
            raise conflict("unsupported_image", str(exc), status=415) from exc

        path = drafts_module.frame_path(draft, record)
        assessment = assess_frame(
            path,
            drafts_module.calibration(draft),
            ROI_MODE_HOMOGRAPHY,
            state.preprocess_config,
            state.gate,
            keep_stages=True,
            keep_input=True,
        )
        quality, reason, metrics = verdict_of(assessment.result)

        # Previews first: `record_verdict` records whether they exist, and a verdict claiming
        # `preview: true` about a file that is not there is worse than no preview at all.
        # Only rejections get one — §9 asks for "the reason and a thumbnail", and an accepted
        # frame has nothing to explain.
        if reason is not None and assessment.decoded is not None:
            drafts_module.write_previews(
                draft,
                path,
                freq_hz,
                assessment.decoded.pixels,
                None if assessment.stages is None else assessment.stages.edges,
            )
        return drafts_module.record_verdict(draft, record, quality, reason, metrics)

    @router.delete("/drafts/{draft_id}/frames/{freq_hz}/{name}", status_code=204)
    def drop_frame(draft_id: str, freq_hz: int, name: str) -> Response:
        drafts_module.remove_frame(load(draft_id), freq_hz, Path(name).name)
        return Response(status_code=204)

    @router.get("/drafts/{draft_id}/preview/{freq_hz}/{name}")
    def preview(draft_id: str, freq_hz: int, name: str, kind: str = "original") -> FileResponse:
        """A stored rejection thumbnail. `Path(name).name` is the traversal guard on the one
        path segment in this router that is not a validated session id."""
        if kind not in ("original", "edges"):
            raise conflict("no_preview", f"unknown preview kind {kind!r}", status=404)
        path = drafts_module.preview_path(load(draft_id), freq_hz, Path(name).name, kind)
        if not path.is_file():
            raise conflict("no_preview", f"no {kind} preview for {name}", status=404)
        return FileResponse(path, media_type="image/png")

    return router


def _calibration_preview(pixels: np.ndarray, info: dict[str, Any]) -> str:
    """The frame with the plate quad drawn on it, or the bare frame when there is no quad.

    Drawn on a *display-sized copy*: this picture is for looking at, and the calibration it
    illustrates was computed from the full-resolution array before this function was called.
    The quad is scaled by the same factor the thumbnail used, so what is outlined on screen is
    where the markers actually were — an outline drawn at the wrong scale would be the only
    evidence anyone has about the calibration, saying something that is not true.
    """
    bgr = cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGR)
    view = thumbnail(bgr, CALIBRATION_PREVIEW_MAX_SIDE)
    scale = view.shape[1] / bgr.shape[1]
    if info.get("corners"):
        # `thumbnail` returns its argument untouched when the frame is already small enough,
        # so draw on a copy rather than into the array it was handed.
        view = view.copy()
        quad = (np.array(info["corners"], np.float32) * scale).astype(np.int32)
        cv2.polylines(view, [quad], isClosed=True, color=(0, 190, 0), thickness=2)
    ok, buffer = cv2.imencode(".png", view)
    if not ok:
        raise RuntimeError("cv2.imencode failed for the calibration preview")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


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
