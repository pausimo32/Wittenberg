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
    check_device([])
    print("\nsetup OK")


@target("check-versions", "assert the pinned OpenCV BUILD is the one installed")
def check_versions(argv: list[str]):
    """`config/versions.json` names a package, not just a version — check the package.

    `docs/03-preprocessing-spec.md` §5.3 requires `opencv-contrib-python`, because
    `cv2.aruco` ships only in the contrib build and every session calibration depends on
    it. Plain `opencv-python` imports fine, reports the same `cv2.__version__`, and then
    raises `AttributeError: module 'cv2' has no attribute 'aruco'` the first time a real
    session is calibrated — i.e. after the lab work, not before it. Comparing the version
    string alone measures the wrong thing and passes on that machine.

    Three separate facts are asserted: the pinned distribution is installed, no
    conflicting OpenCV distribution is installed alongside it (§5.3 — they provide the
    same `cv2` and which one wins depends on install order), and `cv2.aruco` answers.
    """
    vfile = CONFIG / "versions.json"
    if not vfile.exists():
        die("config/versions.json missing")
    v = json.loads(vfile.read_text(encoding="utf-8"))
    want_package = v["opencv_package"]
    want_version = v["opencv_python"]

    probe = _probe_opencv()
    if probe.get("import_error"):
        die(f"cannot import cv2:\n  {probe['import_error']}")

    installed = probe["distributions"]
    if want_package not in installed:
        die(
            f"config/versions.json pins {want_package}, which is not installed "
            f"(found: {', '.join(installed) or 'no OpenCV distribution at all'}). "
            f"cv2.aruco ships only in the contrib build and src/calibrate.py needs it for "
            f"every session calibration — see docs/03-preprocessing-spec.md §5.3."
        )

    conflicting = sorted(set(installed) - {want_package})
    if conflicting:
        die(
            f"{want_package} and {', '.join(conflicting)} are both installed. They provide "
            f"the same `cv2` module and which one wins depends on install order, so the "
            f"pipeline is not reproducible on this machine. Uninstall "
            f"{', '.join(conflicting)} — docs/03-preprocessing-spec.md §5.3."
        )

    if probe.get("aruco_error"):
        die(
            f"{want_package} is installed but cv2.aruco does not work:\n"
            f"  {probe['aruco_error']}\n"
            f"src/calibrate.py cannot detect a single marker in this environment."
        )

    def minor(s: str) -> str:
        return ".".join(s.split(".")[:2])

    if minor(probe["cv2_version"]) != minor(want_version):
        die(
            f"{want_package} is {probe['cv2_version']}, config/versions.json pins "
            f"{want_version}. Reproducibility across machines and across time depends on "
            f"this: the ROI warp changes between major versions, and the measurement is in "
            f"docs/adr/0006-opencv-version-baseline.md. "
            f"See docs/03-preprocessing-spec.md §5.3a for the two acceptable resolutions."
        )

    print(f"  {want_package} {probe['distribution_version']}  OK (pinned {want_version})")
    print(f"  cv2 {probe['cv2_version']}, cv2.aruco answers  OK")


#: Every distribution that installs a module called `cv2`. Exactly one may be present.
OPENCV_DISTRIBUTIONS = (
    "opencv-python",
    "opencv-contrib-python",
    "opencv-python-headless",
    "opencv-contrib-python-headless",
)


def _probe_opencv() -> dict:
    """Ask the venv what it actually has installed and whether cv2.aruco works."""
    code = (
        "import json\n"
        "from importlib import metadata\n"
        f"names = {OPENCV_DISTRIBUTIONS!r}\n"
        "out = {'distributions': {}}\n"
        "for n in names:\n"
        "    try:\n"
        "        out['distributions'][n] = metadata.version(n)\n"
        "    except metadata.PackageNotFoundError:\n"
        "        pass\n"
        "try:\n"
        "    import cv2\n"
        "    out['cv2_version'] = cv2.__version__\n"
        "except Exception as exc:\n"
        "    out['import_error'] = f'{type(exc).__name__}: {exc}'\n"
        "    print(json.dumps(out))\n"
        "    raise SystemExit(0)\n"
        "try:\n"
        "    import cv2.aruco\n"
        "    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)\n"
        "except Exception as exc:\n"
        "    out['aruco_error'] = f'{type(exc).__name__}: {exc}'\n"
        "print(json.dumps(out))\n"
    )
    r = subprocess.run([venv_python(), "-c", code], capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0 or not r.stdout.strip():
        die(f"could not inspect the OpenCV installation:\n{r.stderr.strip()}")
    probe = json.loads(r.stdout.strip().splitlines()[-1])
    probe["distribution_version"] = next(iter(probe["distributions"].values()), "?")
    probe["distributions"] = sorted(probe["distributions"])
    return probe


@target("check-device", "report the inference device; CPU is a PASS (ADR 0005)")
def check_device(argv: list[str]):
    """`torch.cuda.is_available()` is not the question. Whether a kernel runs is.

    A `cu121`/`cu124` wheel on a Blackwell card (RTX 50-series, sm_120) installs cleanly,
    reports `is_available() == True`, prints the device name — and then dies on the first
    real operation with `no kernel image is available for execution on the device`. A
    check that only calls `is_available()` passes on exactly the machine it exists to
    protect, so this one runs a real matmul and synchronises.

    CPU is a **pass**, not a warning: ADR 0005 makes CPU the supported configuration of
    the app, ≈ 12 ms per inference. Only a GPU that is present and broken fails here.
    """
    code = r"""
import torch
print("  torch    ", torch.__version__)
print("  threads  ", torch.get_num_threads())
if not torch.cuda.is_available():
    print("  device    CPU")
    raise SystemExit(0)
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print(f"  device    {name}  (sm_{cap[0]}{cap[1]})")
# torch.cuda.is_available() is NOT sufficient: a build without kernels for this
# architecture reports True and then dies on the first real op. See ADR 0005.
try:
    (torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda")).sum().item()
    torch.cuda.synchronize()
    print("  gpu op    OK")
except RuntimeError as exc:
    print(f"  gpu op    FAILED: {exc}")
    print(f"  This build has no kernels for sm_{cap[0]}{cap[1]}. Blackwell (RTX 50xx) needs")
    print("  a cu128 or newer build. See requirements-train.txt.")
    raise SystemExit(1)
"""
    r = subprocess.run([venv_python(), "-c", code], capture_output=True, text=True, cwd=ROOT)
    print(r.stdout.rstrip() or r.stderr.rstrip())
    if r.returncode != 0:
        die("GPU present but unusable — see above", 1)
    if "device    CPU" in r.stdout:
        print(
            "  CPU is the supported configuration for the APP (ADR 0005), not a fallback.\n"
            "  Expected: ~24 ms preprocessing + ~12 ms inference per image.\n"
            "  Training wants the GPU machine; the app never needs one."
        )


# --------------------------------------------------------------------------- #
# F1–F7 — implemented as each phase lands
# --------------------------------------------------------------------------- #

@target("modes", "validate data/modes.csv, write config/classes.json  [F1]")
def modes(argv):
    from modes import ModesError, build_classes, read_modes, write_classes

    provenance = "synthetic" if (DATA / "SYNTHETIC").exists() else "capture"
    try:
        result = build_classes(read_modes(DATA / "modes.csv"), provenance)
    except ModesError as exc:
        die(str(exc))
    for warning in result["warnings"]:
        print(f"  WARNING: {warning}")
    for r in result["rejected"]:
        print(f"  rejected {r['freq_hz']} Hz — {r['reason']}")

    path = write_classes(result, CONFIG / "classes.json")
    print(f"  {result['n_classes']} classes -> {path.relative_to(ROOT)}")
    print("  the class index is frozen; retrain from scratch if this file changes")
    if not result["classes"]:
        die("no usable modes — nothing to train on")


@target("ingest", "raw captures -> data/processed + metadata.csv  [F3a]")
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


@target("split", "assign train/val/test by session_id  [F3a]")
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
