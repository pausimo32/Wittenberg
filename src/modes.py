"""`data/modes.csv` -> `config/classes.json`, the file that fixes what the model predicts.

`docs/02` §3 determines the classes empirically: sweep the plate, mark every frequency
whose figure is stationary, symmetric and reforms after re-seeding, then "keep only
modes whose neighbours are >= 40 Hz away, so classes are physically separable". This
module is step 4 of that list — the only part of it that is code.

**The 40 Hz rule is symmetric.** If two modes sit 25 Hz apart, both are discarded. The
tempting alternative — keep the first, drop the second — leaves a surviving class whose
nearest neighbour is still 25 Hz away, which is precisely the situation the rule exists
to prevent. A rejected mode is not deleted from `modes.csv`; it stays there with its
measurements, and `classes.json` records why it was not used.

**The class index is frozen.** Classes are written in ascending frequency order and the
index is the position in that list. That integer is what a trained checkpoint stores in
place of a frequency, so it cannot be renumbered after training: inserting a newly
discovered lower mode would shift every index above it, and every existing checkpoint
would then answer with the wrong frequency while reporting perfect confidence. If the
mode list changes, the classes change, and the models trained on the old list are void.
`config/classes.json` says this in its own text so a reader of the artefact does not
have to find this docstring to learn it.

The 40 Hz threshold lives here rather than in a config file on purpose: it is a
physical separability criterion from the spec, not a knob to tune per run. It is copied
into every `classes.json` so the artefact stays self-describing.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

#: docs/02-dataset-spec.md §3 step 4.
MIN_SEPARATION_HZ = 40

#: docs/00-master-plan.md F1: "Expect 10-18 usable modes. Fewer than 10 fails the phase
#: gate". It is a gate on the *rig*, not on this code — see `build_classes`.
EXPECTED_MIN_MODES = 10

#: Why F1 sets the floor where it does, quoted so the failure says it at the bench.
THIN_SWEEP_CAUSE = (
    "it means the drive amplitude is too low or the plate is over-constrained at the "
    "centre. Fix the rig and sweep again before continuing — this is not something the "
    "40 Hz separation rule or any threshold in this file can be adjusted to fix."
)

REQUIRED_COLUMNS = ("freq_hz", "width_hz", "n_nodal_lines", "symmetry", "quality", "notes")

CLASSES_VERSION = "1.0.0"


class ModesError(ValueError):
    """`data/modes.csv` is missing, malformed, or describes no usable modes."""


def read_modes(path: Path | str) -> list[dict[str, Any]]:
    """Read and validate `data/modes.csv`. Raises `ModesError` rather than guessing."""
    p = Path(path)
    if not p.exists():
        raise ModesError(f"{p} not found — F1 produces it (docs/00-master-plan.md)")

    with p.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ModesError(f"{p}: missing column(s) {', '.join(missing)}")
        rows = list(reader)

    if not rows:
        raise ModesError(f"{p}: no modes — the sweep in docs/02 §3 has not been recorded")

    modes: list[dict[str, Any]] = []
    seen: set[int] = set()
    for i, row in enumerate(rows, start=2):        # 2: line 1 is the header
        freq = _int_field(row, "freq_hz", p, i)
        if freq in seen:
            raise ModesError(f"{p} line {i}: duplicate freq_hz {freq}")
        seen.add(freq)
        modes.append({**row, "freq_hz": freq})

    return sorted(modes, key=lambda m: m["freq_hz"])


def build_classes(
    modes: list[dict[str, Any]], provenance: str = "capture"
) -> dict[str, Any]:
    """Apply the separation rule and assign the frozen class indices.

    **A thin mode list is fatal for real captures and a warning for synthetic ones.**
    `docs/00-master-plan.md` F1 states the gate — "Fewer than 10 fails the phase gate" —
    and states its cause: too little drive amplitude, or a plate clamped too hard at the
    centre. That is a fact about the rig, and the only correct response is to go back to
    the rig, so this raises rather than printing something a person can walk past.

    Six synthetic classes are enough to exercise the pipeline and are not a mode sweep at
    all, so `provenance == "synthetic"` downgrades it to a warning. There is deliberately
    no `--force`: if a real sweep ever yields nine good modes, the gate moves in a commit
    with the reason written down, not at the command line at 11 p.m.
    """
    freqs = [m["freq_hz"] for m in modes]
    kept, rejected = [], []

    for i, m in enumerate(modes):
        left = freqs[i] - freqs[i - 1] if i else None
        right = freqs[i + 1] - freqs[i] if i + 1 < len(freqs) else None
        gaps = [g for g in (left, right) if g is not None]
        nearest = min(gaps) if gaps else None

        if nearest is not None and nearest < MIN_SEPARATION_HZ:
            rejected.append(
                {
                    "freq_hz": m["freq_hz"],
                    "nearest_neighbour_hz": nearest,
                    "reason": (
                        f"nearest neighbour is {nearest} Hz away; docs/02 §3 requires "
                        f">= {MIN_SEPARATION_HZ} Hz for the classes to be separable"
                    ),
                }
            )
        else:
            kept.append(m)

    classes = [
        {
            "index": i,
            "freq_hz": m["freq_hz"],
            "width_hz": m.get("width_hz", ""),
            "n_nodal_lines": m.get("n_nodal_lines", ""),
            "symmetry": m.get("symmetry", ""),
        }
        for i, m in enumerate(kept)
    ]

    warnings = []
    synthetic = provenance == "synthetic"

    if len(classes) < EXPECTED_MIN_MODES:
        thin = (
            f"{len(classes)} usable mode(s); docs/00-master-plan.md F1 requires at least "
            f"{EXPECTED_MIN_MODES}"
        )
        if synthetic:
            warnings.append(
                f"{thin}. Passed anyway because these are synthetic classes: six are "
                f"enough to exercise the pipeline, and no gate on a real sweep applies "
                f"to a renderer."
            )
        else:
            raise ModesError(f"{thin} — {THIN_SWEEP_CAUSE}")

    if synthetic:
        warnings.append(
            "these classes come from tools/synth_capture.py, not from a mode sweep. "
            "They are for exercising the pipeline only; regenerate config/classes.json "
            "from a real data/modes.csv before training anything that will be reported."
        )

    return {
        "version": CLASSES_VERSION,
        "provenance": provenance,
        "min_separation_hz": MIN_SEPARATION_HZ,
        "n_classes": len(classes),
        "classes": classes,
        "rejected": rejected,
        "warnings": warnings,
    }


def write_classes(result: dict[str, Any], path: Path | str) -> Path:
    """Write `config/classes.json`, index freeze stated in the file itself."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": result["version"],
        "provenance": result.get("provenance", "capture"),
        "_note": (
            "Generated from data/modes.csv by `python tasks.py modes`. The class index is "
            "FROZEN: it is the position in ascending frequency order and it is what a "
            "trained checkpoint stores instead of a frequency. Re-generating this file "
            "after adding or removing a mode renumbers the classes and voids every "
            "checkpoint trained against the previous version. Do not hand-edit it."
        ),
        "min_separation_hz": result["min_separation_hz"],
        "n_classes": result["n_classes"],
        "classes": result["classes"],
        "rejected": result["rejected"],
    }
    p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return p


def load_classes(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ModesError(f"{p} not found — run `python tasks.py modes` first")
    return json.loads(p.read_text("utf-8"))


def _int_field(row: dict[str, Any], key: str, path: Path, line: int) -> int:
    raw = (row.get(key) or "").strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise ModesError(f"{path} line {line}: {key} must be an integer, got {raw!r}") from exc


def write_site_classes(result: dict[str, Any], path: Path | str) -> Path:
    """The class list as a browser global, for the static site.

    Written by the same command that writes `config/classes.json`, so the two cannot
    drift. Mechanical fields only: everything editorial — the description, the
    complexity, the (m,n) indices — lives in `site/js/frequencies.js`, keyed by
    frequency and written by people.

    This exists because the old site hard-coded thirteen frequencies in three places,
    one of them indexing a parallel array *by position*: choosing fourteen real modes
    would have read past its end and stopped the page rendering. The classes are not
    chosen yet, so the list has to follow whatever `data/modes.csv` eventually says
    without anyone editing HTML.
    """
    entries = [
        {
            "hz": c["freq_hz"],
            "label": f"{c['freq_hz']} Hz",
            "n_nodal_lines": c.get("n_nodal_lines"),
            "symmetry": c.get("symmetry"),
            "width_hz": c.get("width_hz"),
        }
        for c in result["classes"]
    ]
    text = (
        "/* Generated from config/classes.json by `python tasks.py modes`.\n"
        " * Do not hand-edit: the class index is frozen and this file is regenerated.\n"
        " * Editorial metadata (description, complexity, mode indices) belongs in\n"
        " * frequencies.js, keyed by frequency. */\n"
        f"window.WITTENBERG_CLASSES = {json.dumps(entries, indent=2, ensure_ascii=False)};\n"
    )
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p
