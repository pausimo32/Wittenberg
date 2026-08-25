"""Contract tests for `tools/check_capture.py` — the bench check run on the first photo.

`docs/02` §6 puts this tool at the start of every session precisely because Canny's fixed
50/150 thresholds assume a sand-to-plate contrast that a low-contrast capture does not
have. The point of running it on photo one is that it says the same thing ingest will say
on day four.

So the property pinned here is **agreement**: for the same frame, the tool's verdict and
`src/quality.assess`'s verdict are the same verdict, because they are the same code. A
tool with its own copy of the gate would eventually tell the operator to keep shooting a
session that ingest was about to reject, which is the failure it exists to prevent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from check_capture import main  # noqa: E402
from synth_capture import render_frame, spoil  # noqa: E402

from calibrate import calibrate  # noqa: E402
from preprocess import load_config  # noqa: E402
from quality import assess, load_gate  # noqa: E402


def frame(defect: str = "", n_markers: int = 4, blank: bool = False) -> np.ndarray:
    rng = np.random.default_rng(7)
    return spoil(render_frame(1, 2, rng, "lamp_diffuse", n_markers=n_markers, blank=blank), defect, rng)


def written(tmp_path: Path, image: np.ndarray, name: str = "IMG_0001.png") -> Path:
    path = tmp_path / name
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA))
    return path


def verdict_of(image: np.ndarray):
    """What `tasks.py ingest` would decide about this frame."""
    calib = calibrate(image, session_id="check")
    n_markers = calib.n_markers
    return assess(
        image,
        {"roi_mode": calib.roi_mode, "corners": calib.corners},
        n_markers=n_markers,
        config=load_config(),
        gate=load_gate(),
    )


@pytest.mark.parametrize(
    "defect,n_markers,blank",
    [("", 4, False), ("blurred", 4, False), ("exposure", 4, False), ("", 4, True)],
)
def test_the_tool_agrees_with_the_ingest_gate(tmp_path, defect, n_markers, blank):
    image = frame(defect, n_markers, blank)
    path = written(tmp_path, image)
    exit_code = main([str(path), "--out", str(tmp_path / "strip.png")])
    assert (exit_code == 0) == verdict_of(image).accepted


def test_a_three_marker_frame_fails_before_anything_is_measured(tmp_path):
    """No four markers, no homography, so there is no ROI to measure blur on."""
    path = written(tmp_path, frame(n_markers=3))
    assert main([str(path), "--out", str(tmp_path / "strip.png")]) == 1


def test_it_writes_the_side_by_side_strip(tmp_path):
    """The number is the verdict; the strip is what tells the operator *why*."""
    path = written(tmp_path, frame())
    out = tmp_path / "strip.png"
    main([str(path), "--out", str(out)])
    strip = cv2.imread(str(out))
    assert strip is not None
    size = load_config()["roi"]["output_size"]
    assert strip.shape[:2] == (size, size * 3)


def test_the_thresholds_come_from_the_config_not_the_tool(tmp_path, monkeypatch, capsys):
    """Move the floor in `config/quality_gate.json` and the tool has to move with it."""
    gate = load_gate()
    gate["laplacian_variance_min"] = 10_000_000.0
    monkeypatch.setattr("check_capture.load_gate", lambda: gate)

    path = written(tmp_path, frame())
    assert main([str(path), "--out", str(tmp_path / "strip.png")]) == 1
    laplacian = next(l for l in capsys.readouterr().out.splitlines() if l.startswith("laplacian"))
    assert "FAIL" in laplacian
    assert "1e+07" in laplacian


def test_an_unreadable_file_is_refused_not_guessed(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("this is not a capture", encoding="utf-8")
    assert main([str(path), "--out", str(tmp_path / "strip.png")]) == 1


def test_it_carries_no_thresholds_of_its_own():
    """`docs/02` §6 puts the gate in config/quality_gate.json; a second copy here would
    drift, and the drift would show up as the tool disagreeing with ingest."""
    source = (REPO_ROOT / "tools" / "check_capture.py").read_text(encoding="utf-8")
    for literal in ("100.0", "215", "0.005", "0.25", "0.02"):
        assert literal not in source
