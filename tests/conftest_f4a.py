"""A miniature but structurally complete dataset, for the training-bench tests.

Three sessions so `tasks.py split` can produce a train/val/test partition at all, and every
class in `config/classes.json` in every session so no class is missing from a fold — which
`src/dataset.py` refuses, correctly, since a class with no training images cannot be learned.

Built with the same renderer `tools/synth_capture.py` uses, so what these tests train on is
the same kind of image the dry run produces.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from synth_capture import render_frame

from calibrate import calibrate, write_calibration
from ingest import ingest_session
from metadata import session_date
from model import classes_from_config
from split import split_dataset

#: One (n, m) mode shape per class, so the classes are at least visually distinguishable.
SHAPES = [(1, 2), (1, 3), (2, 3), (1, 4), (2, 4), (3, 4), (1, 5), (2, 5), (3, 5), (4, 5)]

SESSION_IDS = ("SYNTH-20260101-01", "SYNTH-20260102-01", "SYNTH-20260103-01")


def build_dataset(root: Path, frames_per_class: int = 3) -> Path:
    """Write, ingest and split a complete miniature dataset under `root`."""
    frequencies = [int(c["freq_hz"]) for c in classes_from_config()]
    rng = np.random.default_rng(11)

    for session_id in SESSION_IDS:
        session_dir = root / "raw" / session_id
        good = None
        for freq, (n, m) in zip(frequencies, SHAPES):
            freq_dir = session_dir / str(freq)
            freq_dir.mkdir(parents=True, exist_ok=True)
            for k in range(frames_per_class):
                frame = render_frame(n, m, rng, "lamp_diffuse", n_markers=4)
                good = good if good is not None else frame
                cv2.imwrite(
                    str(freq_dir / f"IMG_{k:04d}.png"),
                    cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA),
                )

        (session_dir / "session.json").write_text(
            json.dumps(
                {
                    "session_id": session_id,
                    "date": session_date(session_id),
                    "operator": "test",
                    "plate": "test-200x200",
                    "synthetic": True,
                    "material": "sand",
                    "grain_mass_g": 2.0,
                    "distance_mm": 210,
                    "lighting": "lamp_diffuse",
                    "dwell_s": 3.0,
                    "notes": "",
                }
            ),
            encoding="utf-8",
        )
        write_calibration(calibrate(good, session_id=session_id), root / "calib")
        ingest_session(session_id, root)

    split_dataset(root)
    return root
