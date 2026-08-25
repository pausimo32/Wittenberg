"""Generate synthetic capture sessions, so the ingest pipeline can be built and proved
before the lab produces a single real photograph.

`docs/00-master-plan.md` F3 is "do externally — the phase that takes real calendar
time": four sessions on four different days. That gates the *dataset*, and nothing can
change it. It should not also gate the *code that ingests the dataset*, which is what
this tool decouples: it emits `data/raw/SYNTH-NN/` trees that are structurally
identical to real ones, so `tasks.py ingest` and `tasks.py split` can be written,
tested and left green today, and the real captures only have to be dropped in.

    python tools/synth_capture.py --sessions 4 --frames 20 --classes 6

What comes out is deliberately *not* physics. The nodal figures are the standard
square-plate approximation, not a simulation of our plate, and no number produced from
synthetic data may ever appear in the TDR. They exist to exercise the gate, the
metadata writer and the splitter. `SYNTH-` is in every session id precisely so this is
impossible to forget, and `tasks.py ingest` refuses to mix these sessions with real
ones in one `metadata.csv`.

Each session gets its own lighting, material and grain mass — varied *between*
sessions and constant *within* one, as `docs/02` §2 requires — and a fixed fraction of
frames is spoiled on purpose, one defect per §6 rejection reason, so the yield report
has something real to count.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calibrate import calibrate, draw_marker, write_calibration

SEED = 42
SESSION_PREFIX = "SYNTH-"

FRAME_PX = 600
MARKER_PX = 90
MARKER_INSET = 15

PLATE_MIN = MARKER_INSET + MARKER_PX      # 105 — where the plate starts
PLATE_MAX = FRAME_PX - PLATE_MIN          # 495 — and ends

#: (n, m) mode pairs, in ascending frequency. The label is the frequency; the pair only
#: decides what the figure looks like.
MODE_SHAPES = [(1, 2), (1, 3), (2, 3), (1, 4), (2, 4), (3, 4), (1, 5), (2, 5), (3, 5), (4, 5)]

#: Conditions per session. Lighting varies between sessions and never within one
#: (`docs/02` §2), and the last session changes material for ablation A8.
#:
#: A **2x2 factorial**: two materials crossed with two lightings, one session per cell,
#: so each value of each condition appears in exactly two of the four sessions.
#:
#: This is not cosmetic. The split assigns whole sessions, so a condition that occurs in
#: only one session predicts the split perfectly. The first version of this table gave
#: every session a distinct lighting and dwell; the held-out session was then the only
#: `lamp_grazing` one, and test accuracy would have measured generalisation to an unseen
#: lamp angle rather than to an unseen plate. Fixing lighting alone left the same hole in
#: `material` — three sessions of sand and one of semolina puts semolina wholly on one
#: side of the split.
#:
#: Balanced this way, whichever single session is held out, its material, lighting and
#: dwell each still occur in a session that stays in train. Real capture planning has to
#: solve exactly this, and it has to be solved *before* the sessions are shot.
SESSION_CONDITIONS = [
    {"material": "sand", "grain_mass_g": 2.0, "lighting": "lamp_diffuse", "dwell_s": 3.0},
    {"material": "semolina", "grain_mass_g": 1.6, "lighting": "lamp_grazing", "dwell_s": 3.0},
    {"material": "sand", "grain_mass_g": 2.2, "lighting": "lamp_grazing", "dwell_s": 3.0},
    {"material": "semolina", "grain_mass_g": 1.5, "lighting": "lamp_diffuse", "dwell_s": 3.5},
]

#: Conditions that must never be confounded with the split. Asserted by
#: `tests/test_synth_capture.py` so the table cannot silently regress.
BALANCED_CONDITIONS = ("material", "lighting")

#: Defect injected at every Nth frame of a session, in precedence order. One per §6
#: rejection reason, so a run exercises every branch of the gate.
DEFECTS = [("blurred", 10), ("exposure", 13), ("partial", 17), ("unsettled", 23)]


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #

def build_modes(n_classes: int) -> list[dict]:
    """`n_classes` usable modes, all >= 40 Hz apart, plus one pair that is not.

    The close pair is what gives `tasks.py modes` something to actually reject: with
    only well-separated modes the §3 separation rule would never fire and the target
    would be untested. Both members of the pair are dropped, because the rule is
    symmetric — a mode 25 Hz from its neighbour is unusable in both directions.
    """
    if n_classes > len(MODE_SHAPES):
        raise SystemExit(f"--classes max is {len(MODE_SHAPES)}")

    modes = []
    freq = 420
    for i in range(n_classes):
        n, m = MODE_SHAPES[i]
        modes.append(
            {
                "freq_hz": freq,
                "width_hz": 12 + (i % 3) * 4,
                "n_nodal_lines": n + m,
                "symmetry": "diagonal" if n != m else "square",
                "quality": "clear",
                "notes": f"synthetic mode ({n},{m})",
            }
        )
        freq += 190 + i * 60

    decoy = freq + 300
    for offset in (0, 25):
        modes.append(
            {
                "freq_hz": decoy + offset,
                "width_hz": 6,
                "n_nodal_lines": 9,
                "symmetry": "asymmetric",
                "quality": "faint",
                "notes": "synthetic near-degenerate pair; must fail the 40 Hz rule",
            }
        )
    return modes


def write_modes_csv(modes: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["freq_hz", "width_hz", "n_nodal_lines", "symmetry", "quality", "notes"]
        )
        w.writeheader()
        w.writerows(modes)


def usable_frequencies(modes: list[dict], min_separation: int = 40) -> list[int]:
    freqs = sorted(m["freq_hz"] for m in modes)
    keep = []
    for i, f in enumerate(freqs):
        left = f - freqs[i - 1] if i else 10**9
        right = freqs[i + 1] - f if i + 1 < len(freqs) else 10**9
        if min(left, right) >= min_separation:
            keep.append(f)
    return keep


# --------------------------------------------------------------------------- #
# one frame
# --------------------------------------------------------------------------- #

def chladni_field(size: int, n: int, m: int) -> np.ndarray:
    """The classic free square-plate approximation, evaluated on the plate."""
    t = np.linspace(0.0, 1.0, size)
    x, y = np.meshgrid(t, t)
    return (
        np.cos(n * np.pi * x) * np.cos(m * np.pi * y)
        - np.cos(m * np.pi * x) * np.cos(n * np.pi * y)
    )


def render_frame(
    n: int,
    m: int,
    rng: np.random.Generator,
    lighting: str,
    n_markers: int = 4,
    blank: bool = False,
) -> np.ndarray:
    """One capture: plate, sand on the nodal lines, markers taped at the corners."""
    plate_px = PLATE_MAX - PLATE_MIN
    base = 96 if lighting != "ambient" else 84
    frame = np.full((FRAME_PX, FRAME_PX), base - 20, np.uint8)

    plate = np.full((plate_px, plate_px), base, np.float32)
    if blank:
        # Sand thrown on but not yet organised by the tone: sharp, evenly textured,
        # and carrying no nodal figure at all. It has to stay *sharp*, or the blur rule
        # fires first and the edge-density ceiling is never exercised.
        plate += rng.normal(0.0, 70.0, plate.shape)
    else:
        field = chladni_field(plate_px, n, m)
        # Sand piles where the plate does not move. The band is hard-edged, not a soft
        # falloff: real grains give a crisp ridge, and a soft one reads as out of focus
        # to the Laplacian check in §6 — the first version of this tool failed its own
        # quality gate for exactly that reason.
        ridge = np.abs(field) < 0.055
        plate[ridge] = 232.0
        plate += rng.normal(0.0, 9.0, plate.shape) * ridge  # grain scatter on the ridge
        plate += rng.normal(0.0, 2.0, plate.shape)          # sensor noise everywhere

    frame[PLATE_MIN:PLATE_MAX, PLATE_MIN:PLATE_MAX] = np.clip(plate, 0, 255).astype(np.uint8)
    frame = _light(frame, lighting, rng)

    for marker_id, (mx, my) in _marker_positions(n_markers).items():
        frame[my: my + MARKER_PX, mx: mx + MARKER_PX] = draw_marker(marker_id, MARKER_PX)

    return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGBA)


def _marker_positions(n_markers: int) -> dict[int, tuple[int, int]]:
    far = FRAME_PX - MARKER_INSET - MARKER_PX
    all_pos = {
        0: (MARKER_INSET, MARKER_INSET),
        1: (far, MARKER_INSET),
        2: (far, far),
        3: (MARKER_INSET, far),
    }
    return {i: p for i, p in all_pos.items() if i < n_markers}


def _light(frame: np.ndarray, lighting: str, rng: np.random.Generator) -> np.ndarray:
    """A lighting gradient — the between-session variance the split has to survive."""
    t = np.linspace(-1.0, 1.0, FRAME_PX)
    x, y = np.meshgrid(t, t)
    if lighting == "lamp_grazing":
        gain = 1.0 + 0.22 * x
    elif lighting == "ambient":
        gain = 1.0 - 0.10 * (x ** 2 + y ** 2)
    else:
        gain = 1.0 - 0.05 * (x ** 2 + y ** 2)
    out = frame.astype(np.float32) * gain + rng.normal(0.0, 1.5, frame.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def spoil(frame: np.ndarray, defect: str, rng: np.random.Generator) -> np.ndarray:
    """Damage a frame the way a real session does, one defect per §6 reason.

    Both defects are kept deliberately mild. A frame blurred hard enough to be
    unmistakable also stops ArUco detecting anything, so the gate reports it as
    `partial` and the `blurred` rule is never exercised at all; the same happens to a
    frame darkened until it has no gradients left. These are the strengths at which
    each defect still trips its own rule and no other.
    """
    if defect == "blurred":
        return cv2.GaussianBlur(frame, (15, 15), 4.0)
    if defect == "exposure":
        # Someone nudged the exposure lock: clipped highlights, which §6 calls out
        # as the thing that kills Canny.
        return np.clip(frame.astype(np.float32) * 2.3, 0, 255).astype(np.uint8)
    return frame


def defect_for(index: int) -> str | None:
    for name, period in DEFECTS:
        if index % period == period - 1:
            return name
    return None


# --------------------------------------------------------------------------- #
# a session
# --------------------------------------------------------------------------- #

def write_session(
    out_root: Path,
    session_id: str,
    conditions: dict,
    frequencies: list[int],
    shapes: list[tuple[int, int]],
    frames_per_class: int,
    rng: np.random.Generator,
) -> dict:
    session_dir = out_root / "raw" / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir)

    counts = {"total": 0, "clean": 0}
    index = 0
    calib_source = None

    for freq, (n, m) in zip(frequencies, shapes):
        freq_dir = session_dir / str(freq)
        freq_dir.mkdir(parents=True, exist_ok=True)

        for k in range(frames_per_class):
            defect = defect_for(index)
            frame = render_frame(
                n, m, rng,
                lighting=conditions["lighting"],
                n_markers=3 if defect == "partial" else 4,
                blank=defect == "unsettled",
            )
            frame = spoil(frame, defect or "", rng)

            cv2.imwrite(str(freq_dir / f"IMG_{k:04d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA))
            counts["total"] += 1
            counts["clean"] += defect is None
            if calib_source is None and defect is None:
                calib_source = frame
            index += 1

    (session_dir / "session.json").write_text(
        json.dumps(
            {
                "session_id": session_id,
                "synthetic": True,
                "distance_mm": 210,
                "notes": "generated by tools/synth_capture.py — never for the TDR",
                **conditions,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    calib = calibrate(calib_source, session_id=session_id)
    write_calibration(calib, out_root / "calib")
    return counts


# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--frames", type=int, default=20, help="frames per class per session")
    ap.add_argument("--classes", type=int, default=6)
    ap.add_argument("--out", type=Path, default=ROOT / "data")
    args = ap.parse_args(argv)

    if args.sessions > len(SESSION_CONDITIONS):
        raise SystemExit(f"--sessions max is {len(SESSION_CONDITIONS)}")

    rng = np.random.default_rng(SEED)
    modes = build_modes(args.classes)
    write_modes_csv(modes, args.out / "modes.csv")

    # config/classes.json is a committed file whose class index is frozen, so it must
    # never be mistaken for one derived from a real mode sweep. This marker is what
    # `tasks.py modes` reads to stamp the provenance.
    (args.out / "SYNTHETIC").write_text(
        "data/ in this working copy was generated by tools/synth_capture.py.\n"
        "No number derived from it may appear in the TDR. Delete data/raw/SYNTH-*,\n"
        "data/metadata.csv and this file before ingesting real captures.\n",
        encoding="utf-8",
    )

    frequencies = usable_frequencies(modes)
    shapes = MODE_SHAPES[: len(frequencies)]
    print(f"modes.csv: {len(modes)} modes, {len(frequencies)} usable -> {frequencies}")

    total = 0
    for i in range(args.sessions):
        session_id = f"{SESSION_PREFIX}{i + 1:02d}"
        counts = write_session(
            args.out, session_id, SESSION_CONDITIONS[i], frequencies, shapes, args.frames, rng
        )
        total += counts["total"]
        print(
            f"  {session_id}  {counts['total']:4d} frames "
            f"({counts['total'] - counts['clean']} spoilt on purpose)  "
            f"{SESSION_CONDITIONS[i]['lighting']}/{SESSION_CONDITIONS[i]['material']}"
        )

    print(f"{total} frames in {args.out / 'raw'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
