"""Preparation blocks — the independent unit of ADR 0009 when sessions cannot be one.

A block is the run of frames between two sand replenishments. The capture station records
the cadence as `station.reseed_every` in `data/raw/<session>/session.json`, counted in
*accepted* rounds, so the k-th accepted frame of a class (capture order) belongs to block
`k // reseed_every`. A final block with fewer than 3 frames is merged into the previous one:
a group of one or two frames cannot carry a class through a fold.

One implementation, used by `tasks.py train-hoglr` and by `experiments/protocol_v1/`.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

MIN_FINAL_BLOCK = 3


class GroupsError(RuntimeError):
    """A session's block structure cannot be read."""


def reseed_every(data_root: Path | str, session_id: str) -> int:
    path = Path(data_root) / "raw" / session_id / "session.json"
    try:
        station = json.loads(path.read_text("utf-8")).get("station") or {}
    except (OSError, ValueError) as exc:
        raise GroupsError(f"{path} cannot be read ({exc})") from exc
    k = int(station.get("reseed_every") or 0)
    if k <= 0:
        raise GroupsError(
            f"{path} has no station.reseed_every: this session has no recorded preparation "
            "blocks, so ADR 0009's fallback unit is undefined for it"
        )
    return k


def frame_number(image_id: str) -> int:
    return int(image_id.rsplit("_", 1)[-1])


def preparation_groups(rows: list[dict[str, Any]], data_root: Path | str) -> dict[str, str]:
    """`image_id -> "<session>_bNN"` for accepted rows (`session_id`, `freq_hz`, `image_id`)."""
    cells: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        cells[(r["session_id"], int(r["freq_hz"]))].append(r)
    cadence: dict[str, int] = {}
    out: dict[str, str] = {}
    for (session, _hz), cell in sorted(cells.items()):
        k = cadence.setdefault(session, reseed_every(data_root, session))
        cell = sorted(cell, key=lambda r: frame_number(r["image_id"]))
        n_blocks = -(-len(cell) // k)
        tail = len(cell) % k
        for rank, r in enumerate(cell):
            block = rank // k
            if n_blocks > 1 and tail and tail < MIN_FINAL_BLOCK and block == n_blocks - 1:
                block -= 1
            out[r["image_id"]] = f"{session}_b{block:02d}"
    return out
