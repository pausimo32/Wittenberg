#!/usr/bin/env python3
"""Wittenberg task runner.

Replaces `make` — works identically on Windows, macOS and Linux with no extra
install. Every phase target that is not implemented yet exits 1 with the phase
that will implement it, so a green run always means something real happened.

    python tasks.py <target> [args...]
    python tasks.py --list
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config"
DATA = ROOT / "data"
RUNS = ROOT / "runs"
VENV = ROOT / ".venv"

# The F1/F3 targets import the pipeline modules directly, the same way tests/conftest.py
# does. Imports stay inside the target functions so `--list`, `setup` and `clean` keep
# working on a machine where the dependencies are not installed yet.
sys.path.insert(0, str(ROOT / "src"))

IS_WINDOWS = os.name == "nt"
PY = VENV / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")

TARGETS: dict[str, tuple] = {}


def target(name: str, help_text: str):
    def deco(fn):
        TARGETS[name] = (fn, help_text)
        return fn

    return deco


def npm(*args: str):
    exe = shutil.which("npm")
    if exe is None:
        die("npm not found on PATH — install Node 20")
    return run([exe, *args])


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def die(msg: str, code: int = 1):
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


def not_implemented(phase: str):
    die(f"NOT IMPLEMENTED ({phase})", 1)


def run(cmd: list[str] | str, **kw) -> subprocess.CompletedProcess:
    shell = isinstance(cmd, str)
    printable = cmd if shell else " ".join(cmd)
    print(f"  $ {printable}")
    return subprocess.run(cmd, shell=shell, cwd=ROOT, check=True, **kw)


def venv_python() -> str:
    if not PY.exists():
        die(".venv missing — run `python tasks.py setup` first")
    return str(PY)


def latest_run() -> Path:
    """Windows-safe replacement for a `runs/latest` symlink."""
    pointer = RUNS / "latest.txt"
    if not pointer.exists():
        die("no runs yet — run `python tasks.py train`")
    p = RUNS / pointer.read_text(encoding="utf-8").strip()
    if not p.is_dir():
        die(f"runs/latest.txt points at {p}, which does not exist")
    return p


def write_latest(run_dir: Path):
    RUNS.mkdir(exist_ok=True)
    (RUNS / "latest.txt").write_text(run_dir.name, encoding="utf-8")


# --------------------------------------------------------------------------- #
# F0 — setup and environment checks
# --------------------------------------------------------------------------- #

@target("setup", "create .venv, install Python and Node deps, verify versions")
def setup(argv: list[str]):
    if not PY.exists():
        print("Creating .venv")
        run([sys.executable, "-m", "venv", str(VENV)])
    run([str(PY), "-m", "pip", "install", "--upgrade", "pip", "--quiet"])
    run([str(PY), "-m", "pip", "install", "-r", "requirements.txt"])

    npm("install")

    check_versions([])
    check_gpu([])
    print("\nsetup OK")


@target("check-versions", "assert opencv-python matches the pinned version")
def check_versions(argv: list[str]):
    vfile = CONFIG / "versions.json"
    if not vfile.exists():
        die("config/versions.json missing")
    v = json.loads(vfile.read_text(encoding="utf-8"))
    want_py = v["opencv_python"]

    got = subprocess.run(
        [venv_python(), "-c", "import cv2; print(cv2.__version__)"],
        capture_output=True, text=True, cwd=ROOT,
    )
    if got.returncode != 0:
        die(f"cannot import cv2:\n{got.stderr.strip()}")
    installed = got.stdout.strip()

    def minor(s: str) -> str:
        return ".".join(s.split(".")[:2])

    if minor(installed) != minor(want_py):
        die(
            f"opencv-python is {installed}, config/versions.json pins {want_py}. "
            "Reproducibility across machines and across time depends on this — CLAHE "
            "tile interpolation has changed between minor versions. "
            "See docs/03-preprocessing-spec.md §5.3."
        )
    print(f"  opencv-python {installed}  OK (pinned {want_py})")


@target("check-gpu", "report whether PyTorch sees the GPU")
def check_gpu(argv: list[str]):
    code = (
        "import torch;"
        "print('  torch', torch.__version__);"
        "print('  cuda available:', torch.cuda.is_available());"
        "print('  device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
    )
    r = subprocess.run([venv_python(), "-c", code], capture_output=True, text=True, cwd=ROOT)
    print(r.stdout.rstrip() or r.stderr.rstrip())
    if "cuda available: False" in r.stdout:
        print(
            "  NOTE: training will run on CPU. On native Windows this usually means the\n"
            "  CPU-only torch wheel was installed. Reinstall from the CUDA index URL —\n"
            "  see SETUP.md §5."
        )


# --------------------------------------------------------------------------- #
# F1–F7 — implemented as each phase lands
# --------------------------------------------------------------------------- #

@target("modes", "validate data/modes.csv, write config/classes.json  [F1]")
def modes(argv):
    from modes import build_classes, read_modes, write_classes

    provenance = "synthetic" if (DATA / "SYNTHETIC").exists() else "capture"
    result = build_classes(read_modes(DATA / "modes.csv"), provenance)
    for warning in result["warnings"]:
        print(f"  WARNING: {warning}")
    for r in result["rejected"]:
        print(f"  rejected {r['freq_hz']} Hz — {r['reason']}")

    path = write_classes(result, CONFIG / "classes.json")
    print(f"  {result['n_classes']} classes -> {path.relative_to(ROOT)}")
    print("  the class index is frozen; retrain from scratch if this file changes")
    if not result["classes"]:
        die("no usable modes — nothing to train on")


@target("ingest", "raw captures -> data/processed + metadata.csv  [F3]")
def ingest(argv):
    """`python tasks.py ingest [--session ID | --all]`."""
    from ingest import ingest_session
    from metadata import MetadataError

    sessions = _sessions_to_ingest(argv)
    for session_id in sessions:
        try:
            r = ingest_session(session_id, DATA)
        except MetadataError as exc:
            die(str(exc))

        rejected = ", ".join(f"{k} {v}" for k, v in sorted(r.by_reason.items())) or "none"
        print(
            f"  {r.session_id}  {r.accepted}/{r.total} accepted "
            f"({r.rejection_rate:.1%} rejected: {rejected})  roi={r.roi_mode}"
        )
        for flag in r.flags:
            print(f"    FLAG: {flag}")

    print(f"  report -> {(DATA / 'ingest_report.json').relative_to(ROOT)}")
    print("  rejected frames are still in data/raw/ — they are counted, never deleted")
    print("  next: `python tasks.py split`")


@target("split", "assign train/val/test by session_id  [F3]")
def split(argv):
    from split import split_dataset

    try:
        result = split_dataset(DATA)
    except (ValueError, FileNotFoundError) as exc:
        die(str(exc))

    print(f"  GroupShuffleSplit on session_id, seed {result['seed']}")
    for name in ("train", "val", "test"):
        sessions = ", ".join(result["sessions"][name]) or "-"
        print(f"    {name:<5} {result['images'][name]:5d} images   sessions: {sessions}")
    print(f"  moved {result['moved']} processed images into their split directories")


def _sessions_to_ingest(argv: list[str]) -> list[str]:
    raw = DATA / "raw"
    if not raw.is_dir():
        die(f"{raw} not found — capture a session, or run tools/synth_capture.py")

    available = sorted(p.name for p in raw.iterdir() if p.is_dir())
    if "--all" in argv:
        if not available:
            die(f"no sessions in {raw}")
        return available

    if "--session" in argv:
        i = argv.index("--session")
        if i + 1 >= len(argv):
            die("--session needs a session id, e.g. --session 20260916-01")
        return [argv[i + 1]]

    die(
        "ingest needs a session: `--session <YYYYMMDD-NN>` or `--all`.\n"
        f"       available: {', '.join(available) or '(none)'}"
    )
    return []


@target("train", "train the CNN, write runs/<ts>/  [F4]")
def train(argv): not_implemented("F4")


@target("eval", "confusion matrix, per-class report, Grad-CAM  [F4]")
def eval_(argv): not_implemented("F4")


@target("ablate", "run one ablation from docs/04-model-spec.md §5  [F4]")
def ablate(argv): not_implemented("F4")


@target("check-model", "GATE: checkpoint preprocess hash == current config  [F5]")
def check_model(argv): not_implemented("F5")


@target("api", "start the FastAPI dev server with reload  [F5]")
def api(argv): not_implemented("F5")


@target("app", "build if stale, serve the local app, open the browser  [F6]")
def app(argv): not_implemented("F6")


@target("annexes", "generate the TDR annexes into annexes/  [F7]")
def annexes(argv): not_implemented("F7")


@target("clean", "remove build artefacts (never touches data/raw)")
def clean(argv):
    for p in ["web/dist", "web/node_modules/.vite", "__pycache__", ".pytest_cache"]:
        t = ROOT / p
        if t.exists():
            shutil.rmtree(t, ignore_errors=True)
            print(f"  removed {p}")
    print("clean OK  (data/raw and runs/ untouched — delete those by hand, on purpose)")


# --------------------------------------------------------------------------- #

def main(argv: list[str]):
    if not argv or argv[0] in ("--list", "-l", "--help", "-h"):
        print(__doc__)
        print("Targets:\n")
        for name, (_fn, help_text) in TARGETS.items():
            print(f"  {name:<16} {help_text}")
        return 0
    name = argv[0]
    if name not in TARGETS:
        die(f"unknown target {name!r} — run `python tasks.py --list`", 2)
    fn = TARGETS[name][0]
    try:
        fn(argv[1:])
    except subprocess.CalledProcessError as e:
        die(f"command failed with exit code {e.returncode}", e.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
