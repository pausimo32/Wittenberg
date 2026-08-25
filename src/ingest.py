"""`data/raw/<session_id>/` -> `data/processed/`, `data/metadata.csv`, `data/ingest_report.json`.

The pipeline for one session, in the order `.claude/skills/dataset-ingest` sets out:
decode (`imageio`), calibrate (`calibrate`), gate (`quality`), preprocess
(`preprocess`), then write the rows and the yield report.

Three things worth knowing before changing anything here.

**Where the homography comes from.** The session calibration in
`data/calib/<session_id>.json` — one calibration per session, because the rig is fixed
for the whole session (`docs/02` §2). Per-frame marker detection still happens, but only
to answer §6 rule 2: *this frame* must show four markers to be usable. Deriving the
homography per frame instead would let detector jitter warp each capture slightly
differently, and the within-class variance that produced would be indistinguishable
from real variation between frames.

**Rejects are counted, never deleted.** They stay in `data/raw/`, and every one is
listed in the report with its path, reason and measurements. `docs/02` §6 makes the
rejection rate a reported result of the project (research question S4), not an
embarrassment to be tidied away.

**The split is not assigned here.** Ingest writes `unassigned`; `tasks.py split` fills
the column by session. Because the split is unknown at ingest time, processed images
land in `data/processed/unassigned/<freq_hz>/` and the splitter moves them into the
`train`/`val`/`test` layout §4 describes.

**Frames are processed in parallel when, and only when, that is measurably faster.** Ingest is
decode-bound by roughly 20 to 1: at iPhone resolution the pipeline costs ~29 ms per frame and
the HEIC decode ~529 ms (`docs/02` §5), so ~4 000 real frames go from about 40 minutes to under
ten on a pool. The same pool is *slower* on the 600 px synthetic PNGs, whose frames cost 19 ms
— which is why the decision is made from the measured cost of the first frame rather than from
the frame count. `--jobs N` overrides it in both directions.

Determinism survives because the parallelism is over *frames*, never inside one:
`src/preprocess.py` is a pure function, `src/quality.py` reads only its arguments, the frame
list is built in one fixed order, and results are collected in that order.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

import cv2

from calibrate import (
    ROI_MODE_CENTER_CROP,
    ROI_MODE_HOMOGRAPHY,
    detect_markers,
    read_calibration,
)
from imageio import SUPPORTED_SUFFIXES, UnsupportedImageError, decode
from metadata import (
    SPLIT_NONE,
    SPLIT_UNASSIGNED,
    MetadataError,
    MixedProvenanceError,
    check_provenance,
    is_accepted,
    is_synthetic,
    read_metadata,
    replace_session_rows,
    session_date,
    validate_session_id,
    write_metadata,
)
from preprocess import load_config, preprocess_stages
from quality import assess, load_gate

SESSION_LOG = "session.json"

#: §4 requires these on every row; they describe the capture conditions and cannot be
#: reconstructed after the fact.
REQUIRED_CONDITIONS = ("material", "grain_mass_g", "distance_mm", "lighting", "dwell_s")

#: The whole of `data/raw/<session_id>/session.json`. Five of these become columns of
#: `metadata.csv`; the rest are what makes a session auditable — who shot it, on what
#: plate, on which day. Writing this file takes ten seconds at the bench and cannot be
#: done four days later from memory, which is how a dataset stops being reportable.
REQUIRED_SESSION_KEYS = (
    "session_id",
    "date",
    "operator",
    "plate",
    *REQUIRED_CONDITIONS,
    "notes",
)

#: `notes` is required to be *present* — an empty one means "nothing anomalous", which
#: is information — but every other key must carry a value.
OPTIONAL_VALUE_KEYS = ("notes",)


class IngestError(MetadataError):
    """The session cannot be ingested as it stands.

    Subclasses `MetadataError` so a caller can catch every dataset-integrity failure —
    a malformed session id, a mixed-provenance dataset, a missing session log — with one
    `except`, without having to know which module raised it.
    """


class SessionReport(NamedTuple):
    session_id: str
    total: int
    accepted: int
    rejected: int
    by_reason: dict[str, int]
    by_class: dict[str, dict[str, int]]
    roi_mode: str
    flags: list[str]

    @property
    def rejection_rate(self) -> float:
        return self.rejected / self.total if self.total else 0.0


def ingest_session(
    session_id: str,
    data_root: Path | str,
    jobs: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    session_dir: Path | str | None = None,
) -> SessionReport:
    """Ingest one session. Idempotent: re-running replaces its rows, never duplicates.

    `jobs` is how many worker processes decode and gate frames. `None` picks a sensible
    number, `1` forces everything into this process. See `_assess_all`.

    `progress(done, total)` is called after each frame, in input order. It observes and never
    decides: every byte written is the same whether it is given or not.

    `session_dir` is where the captures are, defaulting to `data/raw/<session_id>`. `/dades`
    passes a staging directory: it moves a browser-uploaded session to
    `data/raw/.incoming-<session_id>`, ingests it there, and renames it to the session's own
    name only once this function has returned — so a failure leaves the draft, not a session
    that half exists. The session id, and therefore every `image_id`, comes from the argument
    and never from the directory name, which is what makes that rename invisible.
    """
    try:
        validate_session_id(session_id)
    except MetadataError as exc:
        raise IngestError(str(exc)) from exc

    root = Path(data_root)
    session_dir = root / "raw" / session_id if session_dir is None else Path(session_dir)
    if not session_dir.is_dir():
        raise IngestError(f"{session_dir} not found")

    conditions = read_session_log(session_dir, session_id)
    calib, roi_mode, flags = _session_calibration(root, session_id, session_dir)

    existing = read_metadata(root / "metadata.csv")
    # Checked before a single file is written, so a refusal leaves the dataset as it was.
    check_provenance(existing, session_id)

    config = load_config()
    gate = load_gate()

    rows: list[dict[str, Any]] = []
    rejected_images: list[dict[str, Any]] = []
    by_reason: Counter[str] = Counter()
    by_class: dict[str, dict[str, int]] = {}

    processed_root = root / "processed" / SPLIT_UNASSIGNED
    _clear_session_processed(root, session_id)

    # The frame list is built first, in one fixed order, and every verdict is collected in
    # that same order. Whether the work ran in one process or eight then makes no difference
    # to a single byte of metadata.csv — which is the property that lets it run in eight.
    frames = _frames_of(session_dir)
    total = len(frames)
    verdicts = _assess_all(frames, calib, roi_mode, config, gate, jobs, progress)

    for (freq_hz, image_path), (quality, reason, metrics, resized) in zip(frames, verdicts):
        tally = by_class.setdefault(str(freq_hz), {"total": 0, "accepted": 0})
        tally["total"] += 1
        index = image_path.stem.split("_")[-1]
        image_id = f"{session_id}_{freq_hz}_{index}"

        if reason is not None:
            by_reason[reason] += 1
            rejected_images.append(
                {
                    "image_id": image_id,
                    "path": str(image_path.relative_to(root)).replace("\\", "/"),
                    "reason": reason,
                    "quality": quality,
                    "metrics": metrics,
                }
            )
        else:
            out = processed_root / str(freq_hz) / f"{session_id}_{index}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), resized)
            tally["accepted"] += 1

        # §4: one row per raw image. A rejected frame is still a frame that was
        # captured, and the committed file is where its verdict is visible.
        rows.append(
            {
                "image_id": image_id,
                "session_id": session_id,
                "freq_hz": freq_hz,
                "split": SPLIT_UNASSIGNED if reason is None else SPLIT_NONE,
                "material": conditions["material"],
                "grain_mass_g": conditions["grain_mass_g"],
                "distance_mm": conditions["distance_mm"],
                "lighting": conditions["lighting"],
                "dwell_s": conditions["dwell_s"],
                "roi_mode": roi_mode,
                "quality": quality,
                "notes": _notes(conditions, reason),
            }
        )

    write_metadata(replace_session_rows(existing, session_id, rows), root / "metadata.csv")

    accepted = sum(1 for r in rows if is_accepted(r))
    report = SessionReport(
        session_id=session_id,
        total=total,
        accepted=accepted,
        rejected=total - accepted,
        by_reason=dict(by_reason),
        by_class=by_class,
        roi_mode=roi_mode,
        flags=flags,
    )
    _write_report(root, report, rejected_images, conditions, gate, config)
    return report


# --------------------------------------------------------------------------- #
# one image
# --------------------------------------------------------------------------- #

#: Estimated serial wall-clock below which the pool is not worth starting. On Windows a worker
#: is a fresh interpreter that re-imports cv2 and numpy, so the pool carries a fixed cost:
#: measured here at ~2.4 s for 8 workers, which puts the break-even at ~2.7 s of serial work.
#: 8 s leaves a margin, so the pool only ever starts when it clearly wins.
PARALLEL_MIN_SECONDS = 8.0

#: More workers than cores does not help a decode-bound job enough to justify oversubscribing
#: a laptop that may also be running the app.
MAX_WORKERS = 8


def worker_count() -> int:
    return max(1, min(os.cpu_count() or 1, MAX_WORKERS))


def is_worth_parallelising(n_remaining: int, seconds_per_frame: float) -> bool:
    """Decide from a *measured* first frame, not from a guess about the frame count.

    The number of frames is the wrong criterion, and measuring showed it: 480 synthetic
    600 px PNGs at 19 ms each ran **slower** through a pool than in one process, while 4 000
    real 12 MP HEICs at ~560 ms each would go from about 40 minutes to under ten. What
    separates those two cases is the cost of one frame, which is dominated by the decode and
    which this function is handed rather than assuming.

    The first frame has to be processed anyway, so timing it is free.
    """
    return n_remaining * seconds_per_frame >= PARALLEL_MIN_SECONDS


#: Set once per worker process by `_init_worker`, so the calibration and the two configs are
#: pickled once at pool start rather than once per frame.
_WORKER: dict[str, Any] = {}


def _init_worker(calib, roi_mode: str, config, gate) -> None:
    _WORKER.update(calib=calib, roi_mode=roi_mode, config=config, gate=gate)


def _assess_worker(image_path_str: str):
    """One frame, in a worker process. Returns only what the parent needs.

    The decoded capture never crosses the process boundary — a 12 MP RGBA frame is 48 MB and
    pickling it back would cost more than the decode saved. What comes back is the verdict,
    the metrics, and the 128x128 step-7 array, which is 16 KB.
    """
    assessment = assess_frame(
        Path(image_path_str),
        _WORKER["calib"],
        _WORKER["roi_mode"],
        _WORKER["config"],
        _WORKER["gate"],
    )
    quality, reason, metrics = verdict_of(assessment.result)
    return quality, reason, metrics, (None if reason is not None else assessment.stages.resized)


def _assess_all(frames, calib, roi_mode: str, config, gate, jobs: int | None, progress=None):
    """Every frame's verdict, **in the order `frames` was given**.

    Determinism does not depend on how many processes did the work: each frame is decoded,
    gated and preprocessed independently — `src/preprocess.py` is a pure function and
    `src/quality.py` reads only its own arguments — the frame list arrives in one fixed order,
    and `Executor.map` yields results in input order. The parallelism is over *frames*, never
    inside one.

    The first frame is always done here, and timing it is what decides the rest: see
    `is_worth_parallelising`. `jobs=1` skips the pool entirely, which is what to reach for
    when something looks wrong.

    **A caller without an `if __name__ == "__main__":` guard will break here** — on Windows
    every worker re-imports the calling module, so an unguarded script re-runs itself once per
    worker. `tasks.py` is guarded; a one-off script may not be, so the pool failure is caught,
    reported, and finished serially rather than turned into eight tracebacks and no dataset.

    `progress(done, total)` is called after each frame, in input order, on the calling thread.
    It observes and never decides: the frame list, the order and every byte written are the same
    whether it is given or not.
    """
    if not frames:
        return []

    _init_worker(calib, roi_mode, config, gate)
    paths = [str(path) for _freq, path in frames]
    total = len(paths)

    def report(done: int) -> None:
        if progress is not None:
            progress(done, total)

    started = time.perf_counter()
    first = _assess_worker(paths[0])
    seconds_per_frame = time.perf_counter() - started
    report(1)
    rest = paths[1:]

    def serially():
        results = [first]
        for path in rest:
            results.append(_assess_worker(path))
            report(len(results))
        return results

    explicit = jobs is not None
    workers = max(1, int(jobs)) if explicit else worker_count()
    if not rest or workers == 1:
        return serially()
    if not explicit and not is_worth_parallelising(len(rest), seconds_per_frame):
        return serially()

    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(calib, roi_mode, config, gate),
        ) as pool:
            results = [first]
            for result in pool.map(_assess_worker, rest, chunksize=4):
                results.append(result)
                report(len(results))
            return results
    except (OSError, RuntimeError) as exc:        # BrokenProcessPool, no spawn, sandbox, ...
        print(
            f"  NOTE: the worker pool did not start ({type(exc).__name__}); finishing in one\n"
            f"        process, which is slower but identical. If this ran from a script, it\n"
            f"        needs an `if __name__ == \"__main__\":` guard."
        )
        return serially()


def _frames_of(session_dir: Path) -> list[tuple[int, Path]]:
    """`[(freq_hz, path)]` for the whole session, in one fixed order."""
    frames: list[tuple[int, Path]] = []
    for freq_dir in sorted(session_dir.iterdir(), key=lambda p: p.name):
        if not freq_dir.is_dir():
            continue
        freq_hz = _freq_of(freq_dir)
        frames += [(freq_hz, path) for path in sorted(_captures(freq_dir))]
    return frames


class Assessment(NamedTuple):
    """What one frame produced. `stages` and `decoded` are `None` unless asked for.

    `keep_stages` and `keep_input` exist for `/dades`'s live gate, which has to draw the
    rejection it is reporting. Both default to off, so ingest's own path costs exactly what it
    cost before: a rejected frame is not preprocessed, and a 48 MB RGBA array is never carried
    back out of a worker process.
    """

    result: Any | None
    stages: Any | None
    decoded: Any | None


def assess_frame(
    image_path: Path,
    calib,
    roi_mode: str,
    config,
    gate,
    *,
    keep_stages: bool = False,
    keep_input: bool = False,
) -> Assessment:
    """Decode, count markers, gate.

    The stages are computed once and handed back so an accepted image is not
    preprocessed a second time just to be written out.
    """
    try:
        decoded = decode(image_path)
    except UnsupportedImageError:
        return Assessment(None, None, None)

    n_markers = len(detect_markers(decoded.pixels)) if roi_mode == ROI_MODE_HOMOGRAPHY else 0
    result = assess(decoded.pixels, calib, n_markers=n_markers, config=config, gate=gate)

    # A `partial` frame has no usable ROI, so there are no stages to compute for it at all.
    wanted = result.accepted or (keep_stages and result.reason != "partial")
    stages = preprocess_stages(decoded.pixels, calib, config) if wanted else None
    return Assessment(result, stages, decoded if keep_input else None)


def gate_frame(image_path: Path, calib, roi_mode: str, config, gate):
    """The verdict `tasks.py ingest` will record for this frame, available before it runs.

    `/dades` calls this and nothing else. `docs/05` §9: "It is a different interface to one
    implementation, not a second one."
    """
    return verdict_of(assess_frame(image_path, calib, roi_mode, config, gate).result)


def verdict_of(result) -> tuple[str, str | None, dict[str, Any]]:
    """`(quality, reason, metrics)` for one frame.

    `reason` is `None` exactly when the frame was accepted. An undecodable file has no
    gate result at all and is recorded as `reject`/`unreadable`: it is a raw image that
    exists, so §4 still wants a row for it.
    """
    if result is None:
        return "reject", "unreadable", {}
    if not result.accepted:
        return result.quality, result.reason, _roundable(result.metrics)
    return result.quality, None, _roundable(result.metrics)


def _notes(conditions: dict[str, Any], reason: str | None) -> str:
    """Session notes, with the gate's reason appended on a rejected row.

    `quality` collapses `luminance` and `edge_density` to the same `reject`, so without
    this the committed file could not tell a clipped frame from a patternless one.
    """
    notes = str(conditions.get("notes", "") or "")
    if reason is None:
        return notes
    tag = f"gate:{reason}"
    return f"{notes}; {tag}" if notes else tag


def _captures(freq_dir: Path):
    return (p for p in freq_dir.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)


def _freq_of(freq_dir: Path) -> int:
    try:
        return int(freq_dir.name)
    except ValueError as exc:
        raise IngestError(
            f"{freq_dir}: subfolders under a session must be named by integer Hz "
            f"(docs/02 §4), got {freq_dir.name!r}"
        ) from exc


# --------------------------------------------------------------------------- #
# session inputs
# --------------------------------------------------------------------------- #

def read_session_log(session_dir: Path, session_id: str) -> dict[str, Any]:
    """Read and validate `data/raw/<session_id>/session.json`. Refuse, never warn.

    Five of these keys are columns of `metadata.csv` and cannot be recovered from the
    images; the rest are what lets someone else audit the session later. A warning here
    would be read as "carry on" and the session would be ingested without them — which
    is not recoverable once the rig has been dismantled. Losing ten seconds before the
    first shot beats losing the session after the last one.
    """
    path = session_dir / SESSION_LOG
    if not path.exists():
        raise IngestError(
            f"{path} not found. Every session needs one, written BEFORE the first "
            f"capture. Required keys: {', '.join(REQUIRED_SESSION_KEYS)}"
            + (", synthetic" if is_synthetic(session_id) else "")
            + ". Five of them are columns of metadata.csv (§4) and none of them can be "
            "reconstructed from the images four days after the lab."
        )

    try:
        log = json.loads(path.read_text("utf-8"))
    except json.JSONDecodeError as exc:
        raise IngestError(f"{path}: not valid JSON — {exc}") from exc
    if not isinstance(log, dict):
        raise IngestError(f"{path}: expected a JSON object, got {type(log).__name__}")

    absent = [k for k in REQUIRED_SESSION_KEYS if k not in log]
    empty = [
        k
        for k in REQUIRED_SESSION_KEYS
        if k in log and k not in OPTIONAL_VALUE_KEYS and log[k] in (None, "")
    ]
    if absent or empty:
        problems = []
        if absent:
            problems.append(f"missing key(s) {', '.join(absent)}")
        if empty:
            problems.append(f"empty value(s) for {', '.join(empty)}")
        raise IngestError(f"{path}: {'; '.join(problems)}")

    if log["session_id"] != session_id:
        raise IngestError(
            f"{path}: session_id is {log['session_id']!r} but the directory is "
            f"{session_id!r}. One of the two is a copy-paste from another session."
        )

    expected_date = session_date(session_id)
    if str(log["date"]).replace("-", "") != expected_date:
        raise IngestError(
            f"{path}: date {log['date']!r} does not match the {expected_date} in the "
            f"session id. `docs/02` §2 makes one session one day; if this really is a "
            f"second run on another day, it needs its own session id."
        )

    synthetic = bool(log.get("synthetic", False))
    if is_synthetic(session_id) and not synthetic:
        raise IngestError(
            f"{path}: a SYNTH-* session must carry \"synthetic\": true. That flag is what "
            f"keeps a rendered frame from being read later as a measurement."
        )
    if synthetic and not is_synthetic(session_id):
        raise IngestError(
            f"{path}: \"synthetic\": true on session {session_id!r}, whose id is a real "
            f"capture id. Rename the session SYNTH-{session_id} or drop the flag."
        )
    return log


def _session_calibration(root: Path, session_id: str, session_dir: Path | None = None):
    """`(calib, roi_mode, flags)`.

    A session with no calibration file is ingested as `center_crop` and flagged — the
    ingest skill allows this explicitly. It is *not* the same as the silent fallback §6
    forbids: that one is about an individual frame whose markers were not found in a
    session that *is* calibrated, and it stays a `partial` rejection.

    A `calibration.json` sitting *inside* the session directory wins over `data/calib/`. That
    is how a session still being staged by `/dades` carries its own geometry:
    `data/calib/<session_id>.json` is a committed file, and writing one for a session whose
    ingest then failed would leave an orphan in the repository describing nothing.
    """
    staged = None if session_dir is None else Path(session_dir) / "calibration.json"
    path = staged if staged is not None and staged.exists() else root / "calib" / f"{session_id}.json"
    if not path.exists():
        return (
            {"roi_mode": ROI_MODE_CENTER_CROP},
            ROI_MODE_CENTER_CROP,
            [
                (
                    f"no calibration at data/calib/{session_id}.json — ingested with "
                    f"roi_mode=center_crop; the plate is not geometrically normalised "
                    f"and this session is not comparable with calibrated ones"
                )
            ],
        )

    calib = read_calibration(path)
    roi_mode = calib.get("roi_mode", ROI_MODE_HOMOGRAPHY)
    if roi_mode == ROI_MODE_HOMOGRAPHY and not calib.get("corners"):
        raise IngestError(
            f"{path} declares roi_mode=homography but carries no corners — recalibrate "
            f"the session rather than ingesting it uncalibrated"
        )
    # `src/calibrate.py` compares the geometrically chosen corner against the nominal
    # table in `docs/02` §1.1 and records any divergence. Surfacing it here is the whole
    # point: it reaches the console the day the session is ingested and stays in
    # `data/ingest_report.json` afterwards.
    return calib, roi_mode, list(calib.get("flags") or [])


def _clear_session_processed(root: Path, session_id: str) -> None:
    """Drop this session's processed images before rewriting them.

    Without this, a re-ingest after some raw frames were deleted or started failing the
    gate would leave orphans on disk that no `metadata.csv` row points at — invisible to
    the curator and silently loadable by a training script that globs the directory.
    """
    processed = root / "processed"
    if not processed.exists():
        return
    for p in processed.rglob(f"{session_id}_*.png"):
        p.unlink()


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #

def _write_report(
    root: Path,
    report: SessionReport,
    rejected_images: list[dict[str, Any]],
    conditions: dict[str, Any],
    gate: dict[str, Any],
    config: dict[str, Any],
) -> Path:
    path = root / "ingest_report.json"
    existing = json.loads(path.read_text("utf-8")) if path.exists() else {}
    sessions = existing.get("sessions", {})

    sessions[report.session_id] = {
        "total": report.total,
        "accepted": report.accepted,
        "rejected": report.by_reason,
        "rejection_rate": round(report.rejection_rate, 4),
        "roi_mode": report.roi_mode,
        "flags": report.flags,
        "conditions": {k: conditions.get(k) for k in REQUIRED_CONDITIONS},
        "session_log": {k: conditions.get(k) for k in REQUIRED_SESSION_KEYS},
        "by_class": report.by_class,
        "rejected_images": rejected_images,
    }

    totals = Counter()
    by_reason: Counter[str] = Counter()
    for s in sessions.values():
        totals["total"] += s["total"]
        totals["accepted"] += s["accepted"]
        by_reason.update(s["rejected"])

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # `config/quality_gate.json`, verbatim. `docs/02` §6: a yield figure without the
        # thresholds that produced it is not reproducible, and the TDR reports the yield.
        "gate": gate,
        "preprocess_version": config.get("version"),
        "sessions": dict(sorted(sessions.items())),
        "totals": {
            "total": totals["total"],
            "accepted": totals["accepted"],
            "rejected": dict(sorted(by_reason.items())),
            "rejection_rate": round(
                1 - totals["accepted"] / totals["total"] if totals["total"] else 0.0, 4
            ),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _roundable(metrics: dict[str, Any]) -> dict[str, Any]:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}


__all__ = ["IngestError", "MixedProvenanceError", "SessionReport", "ingest_session"]
