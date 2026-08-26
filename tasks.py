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
WEB = ROOT / "web"
DIST = WEB / "dist"

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
    """npm, always against `web/` — the only Node project in the repo.

    `--prefix` rather than a `cwd` change: `run` prints the command it executes, and a
    printed npm line that does not say which directory it ran in is the kind of thing
    someone debugs for ten minutes.
    """
    exe = shutil.which("npm")
    if exe is None:
        die("npm not found on PATH — install Node 20 or newer")
    return run([exe, "--prefix", str(WEB), *args])


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
    """Windows-safe replacement for a `runs/latest` symlink — CLAUDE.md points here.

    The pointer file itself is `src/runs.py`'s; this wrapper exists so the CLI reports a
    missing run the way every other target reports a problem, instead of a traceback.
    """
    import runs as runs_module

    try:
        return runs_module.latest_run(RUNS)
    except runs_module.RunsError as exc:
        die(str(exc))
        raise


def write_latest(run_dir: Path):
    import runs as runs_module

    runs_module.write_latest(run_dir, RUNS)


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


def _probe_opencv() -> dict:
    """Ask the venv what it actually has installed and whether cv2.aruco works.

    Deliberately a subprocess against `venv_python()`: this target has to be able to say
    "cv2 is not installed", which it cannot do by importing cv2 itself.
    """
    from environment import OPENCV_DISTRIBUTIONS

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
    """`python tasks.py ingest [--session ID | --all] [--jobs N]`.

    `--jobs 1` puts every frame back in this process. Ingest is decode-bound roughly 20
    to 1, so the default pool takes ~4 000 real frames from about 40 minutes to under
    ten; the verdicts are identical either way (`src/ingest.py`).
    """
    from ingest import ingest_session
    from metadata import MetadataError

    sessions = _sessions_to_ingest(argv)
    jobs = _option(argv, "--jobs", None)
    for session_id in sessions:
        try:
            r = ingest_session(session_id, DATA, jobs=int(jobs) if jobs else None)
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


@target("train", "train the CNN, write runs/<ts>/  [F4a]")
def train(argv):
    """`python tasks.py train [--model wittenberg_v1|baseline_paper] [--epochs N]`."""
    from dataset import DatasetError
    from model import ModelError
    from runs import RunsError
    from train import TrainingError
    from train import train as run_training

    model_name = _option(argv, "--model", "wittenberg_v1")
    epochs = _option(argv, "--epochs", None)

    try:
        run_dir = run_training(
            DATA, model_name=model_name, epochs=int(epochs) if epochs else None
        )
    except (TrainingError, DatasetError, ModelError, RunsError) as exc:
        die(str(exc))

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    print(f"  {metrics['model']}  {metrics['parameters']['trainable']} trainable parameters")
    print(
        f"  {metrics['epochs_run']} epochs on {metrics['device']}, "
        f"best at {metrics['best_epoch']}  "
        f"val_loss {metrics['best_val_loss']:.4f}  val_acc {metrics['best_val_accuracy']:.4f}"
    )
    print(f"  run -> {run_dir.relative_to(ROOT)}  ({metrics['train_seconds']} s)")
    if metrics.get("synthetic"):
        print("  SYNTHETIC DATA — no number in this run may be reported (docs/00 F2.5)")
    print("  next: `python tasks.py eval`")


@target("eval", "confusion matrix, per-class report, Grad-CAM  [F4a]")
def eval_(argv):
    """`python tasks.py eval [--run <name>] [--split test]`."""
    from dataset import DatasetError
    from evaluate import EvaluationError, evaluate_run
    from runs import RunsError

    try:
        run_dir = _run_from(argv)
        summary = evaluate_run(run_dir, DATA, split_name=_option(argv, "--split", "test"))
    except (EvaluationError, DatasetError, RunsError) as exc:
        die(str(exc))

    accuracy = summary["accuracy"]
    low, high = accuracy["frame_level_wilson_95"]
    print(f"  {summary['split']} split: {summary['images']} images, "
          f"sessions {', '.join(summary['sessions'])}")
    print(f"  accuracy {accuracy['frame_level']:.4f}  "
          f"95 % Wilson [{low:.4f}, {high:.4f}]  (frame-level — see §4.1)")
    print(f"  macro F1 {summary['per_class']['macro_f1']:.4f}   worst class "
          f"{summary['per_class']['worst_class_hz']} Hz at recall "
          f"{summary['per_class']['worst_class_recall']:.4f}")

    distance = summary["frequency_distance"]
    if distance["errors"]:
        print(f"  {distance['errors']} errors, median |f_true - f_pred| "
              f"{distance['median_distance_hz']:.0f} Hz, "
              f"{distance['adjacent_class_share']:.1%} on an adjacent mode")
    else:
        print("  no misclassifications — on synthetic data this means nothing (docs/00 F2.5)")

    written = summary["gradcam"]["written"]
    print(f"  Grad-CAM for {len(written)} classes -> "
          f"{(run_dir / 'report' / 'gradcam').relative_to(ROOT)}")
    print(f"  report -> {(run_dir / 'report').relative_to(ROOT)}")


@target("ablate", "run one ablation from docs/04-model-spec.md §5  [F4b]")
def ablate(argv): not_implemented("F4b")


@target("check-model", "GATE: checkpoint preprocess hash == current config  [F4a]")
def check_model(argv):
    """`python tasks.py check-model [--run <name>]`. Hard failure, never a warning."""
    from preprocess import load_config
    from runs import RunsError, check_run, missing_artefacts

    try:
        run_dir = _run_from(argv)
        absent = missing_artefacts(run_dir)
        if absent:
            die(f"{run_dir.name} is missing {', '.join(absent)} — it is not a finished run")
        result = check_run(run_dir, load_config())
    except RunsError as exc:
        die(str(exc))

    print(f"  {result['run']}  preprocessing {result['preprocess_version']}")
    print(f"  sha256 {result['sha256']}  MATCHES config/preprocess.json")
    print(f"  {result['opencv_package']} cv2 {result['opencv_version']}  MATCHES the checkpoint")
    print("  this checkpoint may be served (docs/03-preprocessing-spec.md §4)")


def _run_from(argv: list[str]) -> Path:
    from runs import resolve_run

    return resolve_run(argv, RUNS)


def _option(argv: list[str], flag: str, default):
    if flag not in argv:
        return default
    i = argv.index(flag)
    if i + 1 >= len(argv):
        die(f"{flag} needs a value")
    return argv[i + 1]


@target("api", "start the FastAPI dev server with reload  [F5]")
def api(argv):
    """`python tasks.py api [--run <name>] [--port N] [--no-reload]`.

    Development only — `docs/05` §2 runs field sessions from `tasks.py app`, never from a
    reloading server. Binds `127.0.0.1` and one worker in both branches.

    The gate runs here first, in this process, so a checkpoint that disagrees with
    `config/preprocess.json` produces the same message `check-model` prints instead of a
    uvicorn traceback from inside a reloader subprocess.

    Two branches because uvicorn's `--reload` re-imports the app in a spawned process,
    which on Windows does not inherit this file's `sys.path`: the reloading branch hands
    uvicorn an import string and `PYTHONPATH`, and `--run` — which pins a run and has
    nothing to gain from hot reload — serves the already-loaded app in this process.
    """
    from api import HOST, PORT, StartupRefused, create_app, load_serving

    port = int(_option(argv, "--port", PORT))
    pinned = "--run" in argv
    reload_ = not pinned and "--no-reload" not in argv

    try:
        serving = load_serving(argv)
    except StartupRefused as exc:
        die(str(exc))

    print(f"  run        {serving.run_dir.name}")
    print(f"  preprocess {serving.config.get('version')}  sha256 {serving.preprocess_sha256}")
    print(f"  {serving.opencv['package']} cv2 {serving.opencv['version']}")
    print(f"  device     {serving.device.type}  ({len(serving.classes)} classes)")
    print(f"  http://{HOST}:{port}  — 127.0.0.1 only, one worker, no outbound call")

    if reload_:
        src = str(ROOT / "src")
        env = {**os.environ}
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [src, env.get("PYTHONPATH")]))
        run(
            [
                venv_python(), "-m", "uvicorn", "api:create_app", "--factory",
                "--host", HOST, "--port", str(port), "--workers", "1",
                "--reload", "--reload-dir", src,
            ],
            env=env,
        )
        return

    import uvicorn

    uvicorn.run(create_app(argv), host=HOST, port=port, workers=1)


@target("app", "build if stale, serve the local app, open the browser  [F6]")
def app(argv):
    """`python tasks.py app [--run <name>] [--port N] [--no-open] [--force-build]`.

    What a field session runs (`docs/05` §2). Not the Vite dev server: hot-reload artefacts
    and dev overlays in front of a participant invalidate the session, so this serves the
    built bundle out of `web/dist` and never starts Vite.

    No reload, one worker, `127.0.0.1`. The gate runs first, in this process, so a
    checkpoint that disagrees with `config/preprocess.json` fails with the same message
    `check-model` prints rather than after the browser has already opened.
    """
    import webbrowser

    from api import HOST, PORT, StartupRefused, create_app

    port = int(_option(argv, "--port", PORT))
    build_frontend(force="--force-build" in argv)

    try:
        application = create_app(argv, web_dist=DIST)
    except StartupRefused as exc:
        die(str(exc))

    url = f"http://{HOST}:{port}"
    serving = application.state.serving
    print(f"  run        {serving.run_dir.name}")
    print(f"  device     {serving.device.type}  ({len(serving.classes)} classes)")
    if serving.do_not_report:
        # The app shows this as a banner on every view; the console says it too, because
        # whoever starts the server is the person who can still swap the checkpoint.
        print(f"  DO NOT REPORT  {serving.do_not_report}")
    print(f"  {url}  — 127.0.0.1 only, one worker, no outbound call")

    if "--no-open" not in argv:
        webbrowser.open(url)

    import uvicorn

    uvicorn.run(application, host=HOST, port=port, workers=1)


def build_frontend(force: bool = False):
    """`npm run build` when `web/dist` is older than anything that goes into it.

    Staleness by mtime rather than a hash: this exists so nobody demonstrates last week's
    bundle, not to be a build system. `node_modules` and `dist` are excluded from the scan
    or the check would always trip on itself.
    """
    if not WEB.is_dir():
        die("web/ not found — F6 builds the frontend")

    index = DIST / "index.html"
    if force or not index.exists():
        reason = "forced" if force else "no build yet"
    else:
        built = index.stat().st_mtime
        newest = max(
            (p.stat().st_mtime for p in _frontend_sources()),
            default=0.0,
        )
        if newest <= built:
            print(f"  web/dist is current ({index.parent.name}/)")
            return
        reason = "sources newer than the build"

    print(f"  building the frontend ({reason})")
    if not (WEB / "node_modules").is_dir():
        npm("install")
    npm("run", "build")


def _frontend_sources():
    skip = {"node_modules", "dist"}
    for path in WEB.rglob("*"):
        if path.is_file() and not any(part in skip for part in path.relative_to(WEB).parts):
            yield path


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
