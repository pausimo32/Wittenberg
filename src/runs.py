"""`runs/` — the directory a training run leaves behind, and the hash that guards it.

`docs/04-model-spec.md` §3 fixes the contents: `config.json`, `metrics.json`, `history.csv`,
`model.pt`, `report/`. This module owns creating that directory, the `runs/latest.txt`
pointer, and the one guard that stops a checkpoint being served under a preprocessing it
was not trained on.

**Why a pointer file and not a symlink.** Native Windows, no developer mode, no WSL
(`CLAUDE.md`). `runs/latest.txt` contains a directory name; nothing in the project calls
`os.symlink`.

**The hash is of the resolved config, canonically serialised.** `docs/03` §4 requires every
checkpoint to store `config/preprocess.json` and its SHA-256, and the API to refuse to start
on a mismatch. Hashing the file's *bytes* would make re-indenting the JSON — or Git checking
it out with different line endings, which on Windows it does — look exactly like a changed
Canny threshold. So the hash is taken over `json.dumps(config, sort_keys=True,
separators=(",", ":"))`: any changed value changes it, and nothing else does.

**The hash is only half the guard.** It pins the *parameters*, not the *implementation*.
Reinstall the venv with a different OpenCV and the hash still matches — while `docs/adr/0006`
measures `cv2.warpPerspective` differing by up to 7 grey levels in every frame, all 120 tensors
differing, and up to 5 298 of 262 144 edge pixels flipping state. So the OpenCV version and
package are the second key, checked exactly as hard as the first: a checkpoint that disagrees
with the running OpenCV is refused, not warned about. `check_run` does both.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
LATEST = "latest.txt"

#: `docs/04` §3. A run directory without these is not a run.
REQUIRED_ARTEFACTS = ("config.json", "metrics.json", "history.csv", "model.pt")


class RunsError(RuntimeError):
    """`runs/` is missing, malformed, or points at something that is not there."""


class PreprocessMismatch(RunsError):
    """The checkpoint was trained under a different `config/preprocess.json`."""


class OpenCVMismatch(RunsError):
    """The checkpoint was trained under a different OpenCV build.

    Separate from `PreprocessMismatch` because the remedy is different: that one is fixed by
    checking out the config the run was made with, this one by reinstalling the pinned wheel.
    Both are `RunsError`, so a caller that only wants "this checkpoint is not servable" can
    catch one thing.
    """


def preprocess_hash(config: dict[str, Any]) -> str:
    """SHA-256 of the resolved preprocessing config. See the module docstring."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def new_run_dir(runs_root: Path | str | None = None, stamp: str | None = None) -> Path:
    """`runs/<YYYYmmdd-HHMMSS>/`, created, with `report/` alongside."""
    root = Path(RUNS if runs_root is None else runs_root)
    name = stamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = root / name
    if run_dir.exists():
        raise RunsError(f"{run_dir} already exists — refusing to overwrite a run")
    (run_dir / "report").mkdir(parents=True)
    return run_dir


def write_latest(run_dir: Path, runs_root: Path | str | None = None) -> Path:
    """Point `runs/latest.txt` at `run_dir`. Windows-safe replacement for a symlink."""
    root = Path(RUNS if runs_root is None else runs_root)
    root.mkdir(parents=True, exist_ok=True)
    pointer = root / LATEST
    pointer.write_text(run_dir.name + "\n", encoding="utf-8")
    return pointer


def latest_run(runs_root: Path | str | None = None) -> Path:
    root = Path(RUNS if runs_root is None else runs_root)
    pointer = root / LATEST
    if not pointer.exists():
        raise RunsError("no runs yet — run `python tasks.py train`")
    run_dir = root / pointer.read_text(encoding="utf-8").strip()
    if not run_dir.is_dir():
        raise RunsError(f"runs/{LATEST} points at {run_dir}, which does not exist")
    return run_dir


def resolve_run(argv: list[str], runs_root: Path | str | None = None) -> Path:
    """`--run <name>` if given, otherwise whatever `runs/latest.txt` points at."""
    root = Path(RUNS if runs_root is None else runs_root)
    if "--run" in argv:
        i = argv.index("--run")
        if i + 1 >= len(argv):
            raise RunsError("--run needs a run directory name")
        run_dir = root / argv[i + 1]
        if not run_dir.is_dir():
            raise RunsError(f"{run_dir} does not exist")
        return run_dir
    return latest_run(root)


def read_run_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "config.json"
    if not path.exists():
        raise RunsError(f"{path} not found — {run_dir.name} is not a finished run")
    return json.loads(path.read_text("utf-8"))


def check_preprocess_hash(run_dir: Path, current: dict[str, Any]) -> dict[str, str]:
    """The gate. Raises `PreprocessMismatch`; never returns a warning.

    `docs/03` §4: "This is a hard failure, never a warning." A checkpoint trained on one
    edge map and served against another answers confidently and wrongly, and there is no
    symptom — the images look the same and the probabilities look the same.
    """
    stored = read_run_config(run_dir)
    want = stored.get("preprocess_sha256")
    if not want:
        raise RunsError(
            f"{run_dir.name}/config.json carries no preprocess_sha256, so it cannot be "
            f"checked. Retrain: a checkpoint whose preprocessing cannot be verified is "
            f"not servable (docs/03 §4)."
        )

    got = preprocess_hash(current)
    if got != want:
        raise PreprocessMismatch(
            f"{run_dir.name} was trained under config/preprocess.json version "
            f"{stored.get('preprocess_version')} (sha256 {want[:12]}...), but the working "
            f"copy is version {current.get('version')} (sha256 {got[:12]}...).\n"
            f"The checkpoint reads a different edge map from the one this config produces. "
            f"Retrain, or check out the config the run was made with. This is the guard "
            f"working, not an obstacle (docs/03-preprocessing-spec.md §4)."
        )
    return {
        "run": run_dir.name,
        "preprocess_version": stored.get("preprocess_version", "?"),
        "sha256": got,
    }


def check_opencv(run_dir: Path, current: dict[str, Any] | None = None) -> dict[str, str]:
    """The second key. Raises `OpenCVMismatch`; never returns a warning.

    `cv2.__version__` and the distribution name are compared exactly. A wheel bump that leaves
    `cv2.__version__` alone changes no pixel and does not fire; a move between OpenCV versions
    changes the ROI warp and does.

    A checkpoint written before this field existed carries no version at all. That is treated
    the same way a missing hash is treated: unverifiable is not the same as fine.
    """
    from environment import opencv_signature

    stored = read_run_config(run_dir)
    want_version = stored.get("opencv_version")
    want_package = stored.get("opencv_package")
    if not want_version:
        raise RunsError(
            f"{run_dir.name}/config.json records no opencv_version, so the build that "
            f"produced its pixels cannot be identified. Retrain: docs/adr/0006 measures the "
            f"ROI warp moving between OpenCV versions, and a checkpoint that cannot be "
            f"checked against one is not servable (docs/03 §4)."
        )

    signature = current if current is not None else opencv_signature()
    got_version = signature["opencv_version"]
    got_package = signature["opencv_package"]

    if got_version != want_version or got_package != want_package:
        raise OpenCVMismatch(
            f"{run_dir.name} was trained under {want_package} (cv2 {want_version}), but this "
            f"environment has {got_package} (cv2 {got_version}).\n"
            f"config/preprocess.json is unchanged, which is exactly why this check exists: the "
            f"parameters are the same and the pixels are not. docs/adr/0006 measures "
            f"cv2.warpPerspective differing by up to 7 grey levels in every frame between 4.12 "
            f"and 5.0, with every final tensor differing.\n"
            f"Reinstall the pinned wheel (`python tasks.py check-versions` names it), or "
            f"retrain and supersede docs/adr/0006."
        )
    return {"opencv_package": got_package, "opencv_version": got_version}


def check_run(run_dir: Path, config: dict[str, Any]) -> dict[str, str]:
    """Both keys of the guard, in one call. What `tasks.py check-model` and the API run."""
    result = check_preprocess_hash(run_dir, config)
    result.update(check_opencv(run_dir))
    return result


def missing_artefacts(run_dir: Path) -> list[str]:
    return [name for name in REQUIRED_ARTEFACTS if not (run_dir / name).exists()]
