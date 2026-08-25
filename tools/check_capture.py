#!/usr/bin/env python3
"""Lab sanity check — run this on the FIRST photo of every capture session.

    python tools/check_capture.py path/to/IMG_0001.HEIC

Answers, in about a second, the question that otherwise takes four days to surface:
*does Canny actually see this pattern?* It detects the markers, warps the plate, runs the
pipeline and reports the **interior** edge density — excluding the ROI border, whose
edges are an artefact of the warp and would otherwise flatter the number
(`docs/02-dataset-spec.md` §6).

Exit code 0 = good to keep shooting. 1 = fix the lighting or the material now.

**This tool owns no thresholds and no transform.** It calls `src/imageio.py`,
`src/calibrate.py`, `src/preprocess.py` and `src/quality.py` — the same four modules
`tasks.py ingest` calls — and reads its limits from `config/quality_gate.json`. A
bench tool that carried its own copy of the gate would be a second implementation of
the very thing it is checking, and the day the two disagreed the operator would be
told the capture was fine when ingest was about to reject it. That is the failure this
tool exists to prevent, not one to reproduce.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calibrate import calibrate  # noqa: E402
from imageio import UnsupportedImageError, decode  # noqa: E402
from preprocess import load_config, preprocess_stages  # noqa: E402
from quality import assess, load_gate  # noqa: E402

#: What to try, in order, when Canny sees nothing. Ordered by how much they cost.
REMEDIES = (
    "more grazing light — it is contrast, not brightness, that Canny needs",
    "a lighter granular material against the plate, or a darker plate",
    "more sand, and a longer dwell so the lines settle",
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("image", type=Path)
    ap.add_argument("--out", type=Path, default=Path("capture_check.png"))
    a = ap.parse_args(argv)

    config = load_config()
    gate = load_gate()

    try:
        decoded = decode(a.image)
    except (UnsupportedImageError, FileNotFoundError) as exc:
        print(f"FAIL  cannot read {a.image}: {exc}")
        return 1

    # One frame, so the calibration is this frame's own. In a session `tasks.py ingest`
    # uses the session calibration instead — here there is not one yet, which is the
    # whole point of running this before the session starts.
    calib = calibrate(decoded.pixels, session_id="check")
    print(f"markers      {calib.marker_ids}  ({calib.n_markers}/{gate['markers_required']})")
    if calib.corners is None:
        print("FAIL  need markers 0, 1, 2, 3. Check the quiet-zone border and the lighting.")
        return 1
    for flag in calib.flags:
        print(f"WARNING      {flag}")
    if calib.px_per_mm:
        print(f"scale        {calib.px_per_mm:.2f} px/mm")

    result = assess(
        decoded.pixels,
        {"roi_mode": calib.roi_mode, "corners": calib.corners},
        n_markers=calib.n_markers,
        config=config,
        gate=gate,
    )
    metrics = result.metrics

    _report("laplacian", metrics["laplacian_variance"], gate["laplacian_variance_min"], None, "{:.0f}")
    _report("luminance", metrics["mean_luminance"], gate["mean_luminance"]["min"],
            gate["mean_luminance"]["max"], "{:.0f}")
    _report("edges", metrics["edge_density"], gate["edge_density"]["min"],
            gate["edge_density"]["max"], "{:.4f}")

    full = metrics["edge_density_full_frame"]
    if full > 0:
        share = (1 - metrics["edge_density"] / full) * 100
        print(f"             (full frame {full:.4f}; border is {share:.0f} % of it)")

    _write_strip(decoded.pixels, calib, config, a.out)
    print(f"\nwrote {a.out}  (capture | ROI | Canny)")

    if result.accepted:
        print("\nGood. Keep shooting.")
        return 0

    print(f"\nThis frame would be rejected at ingest as {result.quality!r} ({result.reason}).")
    if result.reason == "edge_density" and metrics["edge_density"] < gate["edge_density"]["min"]:
        print("The interior is essentially empty: Canny is not seeing the pattern.")
        print("Before shooting anything else, try in this order:")
        for i, remedy in enumerate(REMEDIES, start=1):
            print(f"  {i}. {remedy}")
        print("Only if none of that works, discuss lowering the Canny thresholds — that is a")
        print("preprocessing change: version bump, spec update, and it invalidates checkpoints.")
    return 1


def _report(label: str, value: float, low: float, high: float | None, fmt: str) -> None:
    ok = low <= value and (high is None or value <= high)
    limits = f"[{low:g}, {high:g}]" if high is not None else f">= {low:g}"
    print(f"{label:<12} {fmt.format(value)}  {'ok' if ok else 'FAIL'}   {limits}")


def _write_strip(pixels: np.ndarray, calib, config, out: Path) -> None:
    """Capture, warped ROI and Canny side by side — the three things worth eyeballing."""
    stages = preprocess_stages(pixels, {"roi_mode": calib.roi_mode, "corners": calib.corners}, config)
    size = stages.roi.shape[0]
    strip = np.hstack(
        [
            cv2.cvtColor(cv2.resize(pixels, (size, size)), cv2.COLOR_RGBA2BGR),
            cv2.cvtColor(stages.roi, cv2.COLOR_RGBA2BGR),
            cv2.cvtColor(stages.edges, cv2.COLOR_GRAY2BGR),
        ]
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), strip)


if __name__ == "__main__":
    raise SystemExit(main())
