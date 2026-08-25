#!/usr/bin/env python3
"""Measure what changing the OpenCV version does to the pipeline's pixels.

This is the tool that produced the tables in `docs/adr/0006-opencv-version-baseline.md`.
It exists because "align the pin" (`docs/03-preprocessing-spec.md` §5.3a) is a decision
whose cost has to be a number, not an assertion: preprocessing output *is* the dataset,
so an OpenCV upgrade is a dataset migration or it is nothing.

Two passes, because they answer different questions.

**Accumulated** (`--mode chain`) runs the whole pipeline in each environment. This is what
actually happens to a capture, and it is what the dataset would look like.

**Isolated** (`--mode isolated`) feeds *every* stage the same reference input — the pinned
environment's output of the previous stage — so each number is that one operation's own
difference rather than the sum of everything upstream. Without this pass a version change
looks like six operations drifting; with it, one of them is responsible and the other five
are amplifiers.

Usage — run `dump` once per environment, then `compare`:

    <pinned-python>  tools/opencv_baseline_diff.py dump  out/pinned  --mode chain
    <other-python>   tools/opencv_baseline_diff.py dump  out/other   --mode chain
    <any-python>     tools/opencv_baseline_diff.py compare out/pinned out/other

    <other-python>   tools/opencv_baseline_diff.py dump  out/iso  --mode isolated \\
                                                  --reference out/pinned

The second environment needs nothing from this repository but `cv2` and `numpy`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: Every array `preprocess_stages` produces after step 2. `source` is the caller's own
#: input and cannot differ between environments, so it is not compared.
STAGES = ("roi", "gray", "clahe", "blur", "edges", "resized", "tensor")


def frames_of(session_id: str) -> list[Path]:
    session = ROOT / "data" / "raw" / session_id
    if not session.is_dir():
        raise SystemExit(f"{session} not found — run tools/synth_capture.py first")
    out: list[Path] = []
    for freq_dir in sorted(p for p in session.iterdir() if p.is_dir()):
        out += sorted(freq_dir.glob("IMG_*.png"))
    return out


def dump(args) -> int:
    import cv2

    from calibrate import read_calibration
    import preprocess as P

    config = P.load_config(ROOT / "config" / "preprocess.json")
    calib = read_calibration(ROOT / "data" / "calib" / f"{args.session}.json")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.mode == "isolated" and args.reference is None:
        raise SystemExit("--mode isolated needs --reference <the pinned dump>")

    frames = frames_of(args.session)
    for path in frames:
        key = f"{path.parent.name}_{path.stem}"
        rgba = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)

        if args.mode == "chain":
            s = P.preprocess_stages(rgba, calib, config)
            arrays = {name: getattr(s, name) for name in STAGES}
        else:
            ref = np.load(Path(args.reference) / f"{key}.npz")
            arrays = {
                "roi": P._region_of_interest(rgba, calib, config),
                "gray": P._grayscale(ref["roi"], config["grayscale"]),
                "clahe": P._clahe(ref["gray"], config["clahe"]),
                "blur": P._blur(ref["clahe"], config["blur"]),
                "edges": P._canny(ref["blur"], config["canny"]),
                "resized": P._resize(ref["edges"], config["resize"]),
                "tensor": P._scale(ref["resized"], config["output"]),
            }
        np.savez_compressed(out / f"{key}.npz", **arrays)

    (out / "environment.json").write_text(
        json.dumps(
            {
                "mode": args.mode,
                "session": args.session,
                "frames": len(frames),
                "python": sys.version.split()[0],
                "opencv": cv2.__version__,
                "numpy": np.__version__,
                "preprocess_version": config.get("version"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"{len(frames)} frames -> {out}  (cv2 {cv2.__version__}, python {sys.version.split()[0]})")
    return 0


def compare(args) -> int:
    a_dir, b_dir = Path(args.a), Path(args.b)
    for d in (a_dir, b_dir):
        print(f"  {d.name}: {json.loads((d / 'environment.json').read_text('utf-8'))}")

    names = sorted(p.name for p in a_dir.glob("*.npz"))
    per_frame_max = {s: [] for s in STAGES}
    per_frame_frac = {s: [] for s in STAGES}

    for name in names:
        A, B = np.load(a_dir / name), np.load(b_dir / name)
        for s in STAGES:
            d = np.abs(A[s].astype(np.float64) - B[s].astype(np.float64))
            per_frame_max[s].append(float(d.max()))
            per_frame_frac[s].append(float((d > 0).mean()))

    print(f"\n{len(names)} frames — per-frame max |delta|, and share of pixels changed\n")
    header = f"{'stage':<9}{'median':>9}{'p95':>9}{'max':>9}  |{'median':>9}{'p95':>9}{'max':>9}"
    print(header)
    for s in STAGES:
        m, f = np.array(per_frame_max[s]), np.array(per_frame_frac[s]) * 100
        print(
            f"{s:<9}{np.median(m):>9.4g}{np.percentile(m, 95):>9.4g}{m.max():>9.4g}  |"
            f"{np.median(f):>8.3f}%{np.percentile(f, 95):>8.3f}%{f.max():>8.3f}%"
        )
        differing = int((m > 0).sum())
        print(f"{'':<9}{differing}/{len(names)} frames differ at this stage")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="command", required=True)

    d = sub.add_parser("dump", help="write this environment's stage arrays")
    d.add_argument("out", type=Path)
    d.add_argument("--mode", choices=("chain", "isolated"), default="chain")
    d.add_argument("--reference", type=Path, default=None, help="pinned dump, for isolated mode")
    d.add_argument("--session", default="SYNTH-20260101-01")
    d.set_defaults(func=dump)

    c = sub.add_parser("compare", help="compare two dumps")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    c.set_defaults(func=compare)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
