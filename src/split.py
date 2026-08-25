"""Assign `train` / `val` / `test` by session — docs/02 §2, docs/adr/0002.

Grouped by `session_id`, never random, never per image. Within a session, consecutive
frames of one frequency are near-duplicates; a random split puts near-identical frames
on both sides of the boundary, the model memorises the session rather than the physics,
and test accuracy comes out meaninglessly high. `CLAUDE.md` rule 3 records that the gap
between the naive random-split number and the session-wise number is a reported finding
of this project.

`sklearn.model_selection.GroupShuffleSplit` does the work, at seed 42, exactly as §2
prescribes. Its `test_size` counts *groups*, not samples, which is what makes the 2/1/1
session layout come out right regardless of how many frames each session contributed —
a session with 200 accepted frames and one with 40 are still one group each, and
neither is ever carved up to balance the proportions.

Two consequences worth stating plainly:

* Adding a session **re-partitions everything**. The assignment is a deterministic
  function of the set of session ids and the seed, not an incremental allocation, so a
  fifth session can move an existing one from train to test. Any accuracy figure must
  therefore name the sessions it was measured on, not just the split.
* This is the only thing that may write the `split` column. `CLAUDE.md` forbids
  hand-editing it, and `tasks.py ingest` deliberately leaves it `unassigned`.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from metadata import (
    SPLIT_NONE,
    SPLIT_UNASSIGNED,
    SPLIT_VALUES,
    is_accepted,
    read_metadata,
    sessions_by_split,
    write_metadata,
)

#: CLAUDE.md rule 5 — seed 42 everywhere, reported in every artefact.
SEED = 42

#: docs/02 §2: "Target split 2 sessions train / 1 val / 1 test."
TEST_FRACTION = 0.25
VAL_FRACTION = 0.25

MIN_SESSIONS = 3


class SplitError(ValueError):
    """The dataset cannot be partitioned by session as it stands."""


def assign_splits(rows: list[dict[str, Any]], seed: int = SEED) -> list[dict[str, Any]]:
    """Return `rows` with the `split` column filled in. Pure — the input is not mutated.

    Only rows the gate accepted are partitioned. A rejected frame keeps an empty split:
    it is not training data, not validation data and not test data, and giving it one
    would let a `glob` over `metadata.csv` pull a blurred frame into a fold.

    Grouping is over the sessions that actually contributed an accepted frame, so a
    session that was entirely rejected does not consume one of the three partitions.
    """
    if not rows:
        raise SplitError("no rows to split — run `python tasks.py ingest` first")

    usable = [r for r in rows if is_accepted(r)]
    if not usable:
        raise SplitError(
            "every ingested frame was rejected by the quality gate — nothing to split. "
            "See data/ingest_report.json for the reasons."
        )

    sessions = sorted({r["session_id"] for r in usable})
    if len(sessions) < MIN_SESSIONS:
        raise SplitError(
            f"{len(sessions)} session(s) — need at least {MIN_SESSIONS} to make a "
            f"train/val/test partition, and docs/02 §5 wants at least 4 for a defensible "
            f"result. Capture more sessions rather than splitting one."
        )

    mapping = _session_to_split(sessions, seed)
    return [
        {**r, "split": mapping[r["session_id"]] if is_accepted(r) else SPLIT_NONE}
        for r in rows
    ]


def split_dataset(data_root: Path | str, seed: int = SEED) -> dict[str, Any]:
    """Assign the splits, write `metadata.csv` back, and move the processed images."""
    root = Path(data_root)
    rows = read_metadata(root / "metadata.csv")
    assigned = assign_splits(rows, seed)
    write_metadata(assigned, root / "metadata.csv")

    moved = _relocate(root, assigned)
    by_split = sessions_by_split(assigned)
    return {
        "seed": seed,
        "moved": moved,
        "sessions": {s: sorted(by_split.get(s, set())) for s in SPLIT_VALUES},
        "images": {s: sum(1 for r in assigned if r["split"] == s) for s in SPLIT_VALUES},
        "rejected": sum(1 for r in assigned if not is_accepted(r)),
    }


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #

def _session_to_split(sessions: list[str], seed: int) -> dict[str, str]:
    """`{session_id: split}`.

    Two passes: hold out the test sessions, then carve val out of what is left. One
    `GroupShuffleSplit` cannot produce three parts, and doing it in two passes keeps the
    grouping guarantee intact at each step.

    The groups are the sessions themselves, and `sessions` arrives sorted, so the result
    depends only on the *set* of session ids and the seed — not on the order rows
    happened to be read in.
    """
    index = np.arange(len(sessions))
    groups = np.array(sessions)

    holdout = GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=seed)
    rest_idx, test_idx = next(holdout.split(index, groups=groups))

    # Val is taken as a fraction of what remains, so it stays ~25 % of the whole.
    val_fraction = min(VAL_FRACTION * len(sessions) / max(len(rest_idx), 1), 0.9)
    inner = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_pos, val_pos = next(inner.split(rest_idx, groups=groups[rest_idx]))

    mapping = {sessions[i]: "test" for i in test_idx}
    mapping.update({sessions[rest_idx[i]]: "val" for i in val_pos})
    mapping.update({sessions[rest_idx[i]]: "train" for i in train_pos})

    for split in SPLIT_VALUES:
        if not any(v == split for v in mapping.values()):
            raise SplitError(
                f"the {split!r} split came out empty with {len(sessions)} sessions at "
                f"seed {seed} — capture more sessions"
            )
    return mapping


def _relocate(root: Path, rows: list[dict[str, Any]]) -> int:
    """Move each processed image under `data/processed/<split>/<freq_hz>/`.

    Ingest cannot do this: it does not know the split yet. The final layout is the one
    §4 specifies; `unassigned/` only ever exists between an ingest and a split.
    """
    processed = root / "processed"
    if not processed.exists():
        return 0

    moved = 0
    for row in rows:
        if not is_accepted(row):
            continue                      # a rejected frame has no processed image
        name = _processed_name(row)
        target = processed / row["split"] / str(row["freq_hz"]) / name
        if target.exists():
            continue
        source = _find_processed(processed, str(row["freq_hz"]), name)
        if source is None:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        moved += 1

    _prune_empty(processed)
    return moved


def _processed_name(row: dict[str, Any]) -> str:
    """`<session_id>_<n>.png` — the §4 name, rebuilt from the image_id primary key."""
    index = row["image_id"].rsplit("_", 1)[-1]
    return f"{row['session_id']}_{index}.png"


def _find_processed(processed: Path, freq: str, name: str) -> Path | None:
    for split in (SPLIT_UNASSIGNED, *SPLIT_VALUES):
        candidate = processed / split / freq / name
        if candidate.exists():
            return candidate
    return None


def _prune_empty(processed: Path) -> None:
    for d in sorted(processed.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
