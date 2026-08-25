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
"""

from __future__ import annotations

import json
from collections import Counter
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
    read_metadata,
    replace_session_rows,
    validate_session_id,
    write_metadata,
)
from preprocess import load_config, preprocess_stages
from quality import assess, load_gate

SESSION_LOG = "session.json"

#: §4 requires these on every row; they describe the capture conditions and cannot be
#: reconstructed after the fact.
REQUIRED_CONDITIONS = ("material", "grain_mass_g", "distance_mm", "lighting", "dwell_s")


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


def ingest_session(session_id: str, data_root: Path | str) -> SessionReport:
    """Ingest one session. Idempotent: re-running replaces its rows, never duplicates."""
    try:
        validate_session_id(session_id)
    except MetadataError as exc:
        raise IngestError(str(exc)) from exc

    root = Path(data_root)
    session_dir = root / "raw" / session_id
    if not session_dir.is_dir():
        raise IngestError(f"{session_dir} not found")

    conditions = _read_session_log(session_dir)
    calib, roi_mode, flags = _session_calibration(root, session_id)

    existing = read_metadata(root / "metadata.csv")
    # Checked before a single file is written, so a refusal leaves the dataset as it was.
    check_provenance(existing, session_id)

    config = load_config()
    gate = load_gate()

    rows: list[dict[str, Any]] = []
    rejected_images: list[dict[str, Any]] = []
    by_reason: Counter[str] = Counter()
    by_class: dict[str, dict[str, int]] = {}
    total = 0

    processed_root = root / "processed" / SPLIT_UNASSIGNED
    _clear_session_processed(root, session_id)

    for freq_dir in sorted(session_dir.iterdir(), key=lambda p: p.name):
        if not freq_dir.is_dir():
            continue
        freq_hz = _freq_of(freq_dir)
        tally = by_class.setdefault(str(freq_hz), {"total": 0, "accepted": 0})

        for image_path in sorted(_captures(freq_dir)):
            total += 1
            tally["total"] += 1
            index = image_path.stem.split("_")[-1]
            image_id = f"{session_id}_{freq_hz}_{index}"

            result, stages = _assess_one(image_path, calib, roi_mode, config, gate)
            quality, reason, metrics = _verdict(result)

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
                cv2.imwrite(str(out), stages.resized)
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

def _assess_one(image_path: Path, calib, roi_mode: str, config, gate):
    """Decode, count markers, gate. Returns `(GateResult | None, Stages | None)`.

    The stages are computed once and handed back so an accepted image is not
    preprocessed a second time just to be written out.
    """
    try:
        decoded = decode(image_path)
    except UnsupportedImageError:
        return None, None

    n_markers = len(detect_markers(decoded.pixels)) if roi_mode == ROI_MODE_HOMOGRAPHY else 0
    result = assess(decoded.pixels, calib, n_markers=n_markers, config=config, gate=gate)
    if not result.accepted:
        return result, None
    return result, preprocess_stages(decoded.pixels, calib, config)


def _verdict(result) -> tuple[str, str | None, dict[str, Any]]:
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

def _read_session_log(session_dir: Path) -> dict[str, Any]:
    path = session_dir / SESSION_LOG
    if not path.exists():
        raise IngestError(
            f"{path} not found — the session log records material, grain mass, distance, "
            f"lighting and dwell, and §4 requires all five on every row. They cannot be "
            f"reconstructed from the images."
        )
    log = json.loads(path.read_text("utf-8"))
    missing = [k for k in REQUIRED_CONDITIONS if log.get(k) in (None, "")]
    if missing:
        raise IngestError(f"{path}: session log is missing {', '.join(missing)}")
    return log


def _session_calibration(root: Path, session_id: str):
    """`(calib, roi_mode, flags)`.

    A session with no calibration file is ingested as `center_crop` and flagged — the
    ingest skill allows this explicitly. It is *not* the same as the silent fallback §6
    forbids: that one is about an individual frame whose markers were not found in a
    session that *is* calibrated, and it stays a `partial` rejection.
    """
    path = root / "calib" / f"{session_id}.json"
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
    return calib, roi_mode, []


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
