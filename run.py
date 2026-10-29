#!/usr/bin/env python3
"""One command that starts Wittenberg: `python run.py`.

`tasks.py` is the developer's runner. This is the field session's: it prepares the
environment if it has to, then hands over to `tasks.py app`. It exists because the
documented path is two commands with different prerequisites, and each one has a way
of failing that reads as a traceback rather than as an instruction.

**It runs under the system Python, so it imports nothing but the standard library.**
Not at module level, not inside a function, not `torch`, not `cv2`. On this Mac
`python3` is 3.14 while the pinned wheels are cp311; on a fresh machine there is no
virtual environment at all. Anything imported here has to survive both. `tests/test_run.py`
asserts it by parsing this file's own AST.

Two phases, one file, told apart by which interpreter is running:

**A — bootstrap**, under whichever Python was typed. Finds a 3.11 (SETUP.md §0 pins it,
and a 3.14 venv would resolve different wheels than `requirements.txt` names), creates
`.venv`, installs `requirements.txt` when its hash has moved, then re-launches this file
under `.venv`'s Python.

**B — start**, under `.venv/bin/python`. Puts `src/` on `sys.path` — `src/` is not a
package and its modules import each other by flat name, so this is the precondition for
importing `api` at all — checks the two things that would otherwise be discovered after
the browser has already opened, and calls `tasks.app(argv)`.

**Phase B delegates; it does not reimplement.** The `site/` check, `create_app`, the
startup hash gate that refuses to serve a checkpoint trained under a different pipeline,
the printed header, the DO NOT REPORT banner and `uvicorn.run` all live in `tasks.app`
already. A second copy of that sequence here would be a second place for the gate to be
weakened, and `docs/05` §3 says of that gate: never make it a warning.

**npm is never called.** `tasks.py setup` runs `npm install` for `web/`, the Vite project
ADR 0007 replaced; the app serves `site/`, which is plain HTML, CSS and browser JS with
nothing to build. Node is a hard prerequisite of `setup` and a non-requirement of `app`,
so a launcher that ran `setup` would refuse to start the app on a machine that can run it
perfectly well.

Console output is English, like the rest of `tasks.py`. The French of CLAUDE.md rule 6 is
the browser UI, not the terminal.

    python run.py [--run <name>] [--port N] [--no-open]
"""

from __future__ import annotations

import hashlib
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"

IS_WINDOWS = os.name == "nt"

#: Same platform branch as `tasks.py`, which is the only one in the repository.
PY = VENV / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")

#: The sha256 of the `requirements.txt` the venv was last installed from. Inside `.venv`,
#: so it is git-ignored already and `clean`ing the venv cannot leave a stale stamp behind.
STAMP = VENV / ".wittenberg-deps"

#: SETUP.md §0. Not a range: `requirements.txt` pins exact wheels (torch 2.6.0,
#: numpy 2.1.3, opencv-contrib-python 4.12.0.88) and pip resolves those per Python version.
REQUIRED_PYTHON = (3, 11)

DEFAULT_PORT = 8000


class LaunchError(RuntimeError):
    """Something must be done by a person before the app can start."""


# --------------------------------------------------------------------------- #
# phase A — bootstrap
# --------------------------------------------------------------------------- #

def in_venv() -> bool:
    """True when this process is already the venv's interpreter.

    Compared by resolved path rather than by `sys.prefix`, because an activated venv and
    a re-launch have to look the same, and on macOS `.venv/bin/python` is a symlink whose
    unresolved form differs from `sys.executable`.
    """
    if not PY.exists():
        return False
    try:
        return Path(sys.executable).resolve() == PY.resolve()
    except OSError:
        return False


def python_version(executable: str) -> tuple[int, int] | None:
    """`(major, minor)` of `executable`, or None if it cannot be asked.

    Asked by running it, never inferred from the filename: `python3.11` on PATH is a name,
    not a guarantee, and a wrong guess here builds a venv whose wheels do not match the pins.
    """
    if executable == sys.executable:
        return sys.version_info[:2]
    try:
        probe = subprocess.run(
            [executable, "-c", "import sys; print('%d %d' % sys.version_info[:2])"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if probe.returncode != 0:
        return None
    parts = probe.stdout.split()
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def interpreter_for_venv() -> str:
    """A Python 3.11 to build `.venv` with, or `LaunchError` naming what to install."""
    candidates = [sys.executable, shutil.which("python3.11"), shutil.which("python3.11.exe")]
    if IS_WINDOWS:
        launcher = shutil.which("py")
        if launcher:
            # The py launcher is not itself an interpreter; ask it for the 3.11 it manages.
            found = subprocess.run(
                [launcher, "-3.11", "-c", "import sys; print(sys.executable)"],
                capture_output=True,
                text=True,
            )
            if found.returncode == 0:
                candidates.append(found.stdout.strip())

    for candidate in candidates:
        if candidate and python_version(candidate) == REQUIRED_PYTHON:
            return candidate

    wanted = "%d.%d" % REQUIRED_PYTHON
    running = "%d.%d" % sys.version_info[:2]
    raise LaunchError(
        "no Python {wanted} found (this one is {running})\n"
        "  Wittenberg pins its wheels for {wanted} — SETUP.md §0.\n"
        "  Install it from https://www.python.org/downloads/ and run this file again.".format(
            wanted=wanted, running=running
        )
    )


def requirements_digest() -> str:
    """sha256 of `requirements.txt`'s bytes.

    The bytes, not the parsed content: a launcher has no business knowing what a
    requirement line means, and any edit at all is a reason to install again.
    """
    return hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()


def dependencies_are_current() -> bool:
    """Whether `.venv` was installed from the `requirements.txt` on disk right now."""
    if not PY.exists() or not STAMP.exists():
        return False
    return STAMP.read_text(encoding="utf-8").strip() == requirements_digest()


def install_dependencies() -> None:
    """pip, twice, then the stamp. The one step that needs the network, and only once."""
    print("  installing dependencies (a few minutes, only the first time)", flush=True)
    try:
        subprocess.run(
            [str(PY), "-m", "pip", "install", "--upgrade", "pip", "--quiet"],
            cwd=str(ROOT),
            check=True,
        )
        subprocess.run(
            [str(PY), "-m", "pip", "install", "-r", str(REQUIREMENTS)],
            cwd=str(ROOT),
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise LaunchError(
            "installing dependencies failed (exit {code})\n"
            "  This step, and only this step, needs the internet. The app itself never\n"
            "  makes a network call. Connect once, then run this file again.".format(
                code=exc.returncode
            )
        ) from exc
    STAMP.write_text(requirements_digest() + "\n", encoding="utf-8")


def bootstrap() -> None:
    """Make `.venv` exist and match `requirements.txt`. No npm, ever — see the module docstring."""
    if not REQUIREMENTS.exists():
        raise LaunchError("{path} not found — is this the Wittenberg repository?".format(path=REQUIREMENTS))

    if not PY.exists():
        interpreter = interpreter_for_venv()
        print("  creating .venv with {interpreter}".format(interpreter=interpreter), flush=True)
        try:
            subprocess.run([interpreter, "-m", "venv", str(VENV)], cwd=str(ROOT), check=True)
        except subprocess.CalledProcessError as exc:
            raise LaunchError(
                "could not create .venv (exit {code})".format(code=exc.returncode)
            ) from exc

    if dependencies_are_current():
        return
    install_dependencies()


def relaunch(argv: list[str]) -> int:
    """Run this same file under `.venv`'s Python and hand back its exit code.

    `subprocess`, not `os.execv`: on Windows `execv` replaces the process in a way the
    console handles badly, and Ctrl-C on the server then has nothing to return to.
    """
    return subprocess.run([str(PY), str(Path(__file__).resolve()), *argv], cwd=str(ROOT)).returncode


# --------------------------------------------------------------------------- #
# phase B — start
# --------------------------------------------------------------------------- #

def normalise(argv: list[str]) -> list[str]:
    """`--port=9000` -> `--port 9000`.

    `tasks._option` reads the value as the next element and has no `=` form, so today
    `--port=9000` is not an error — it is silently the default port. Anyone typing it
    means the other thing.
    """
    out: list[str] = []
    for arg in argv:
        if arg.startswith("--") and "=" in arg:
            flag, _, value = arg.partition("=")
            out.extend([flag, value])
        else:
            out.append(arg)
    return out


def port_from(argv: list[str]) -> int:
    """The `--port` value, or the default. Wrong input is a message, not a traceback."""
    if "--port" not in argv:
        return DEFAULT_PORT
    index = argv.index("--port")
    if index + 1 >= len(argv):
        raise LaunchError("--port needs a number, for example: python run.py --port 8001")
    try:
        return int(argv[index + 1])
    except ValueError:
        raise LaunchError(
            "--port wants a number, got {value!r}".format(value=argv[index + 1])
        ) from None


def check_port(port: int) -> None:
    """Refuse a busy port here, before a browser tab is opened onto a dead page.

    `tasks.app` opens the browser and *then* binds, deliberately — the startup gate has
    already run by that point. A port collision is the one failure that arrives after it,
    so it is checked before rather than reported after.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", port))
    except OSError:
        raise LaunchError(
            "port {port} is already in use\n"
            "  Another Wittenberg is probably already running at "
            "http://127.0.0.1:{port}\n"
            "  Close it, or pick another port:  python run.py --port {other}".format(
                port=port, other=port + 1
            )
        ) from None
    finally:
        probe.close()


def check_checkpoint(argv: list[str]) -> None:
    """Say what is missing in words, since a fresh clone has no checkpoint at all.

    `.gitignore` excludes `*.pt` and `runs/*`, and `runs/latest.txt` matches `runs/*`, so
    a `git clone` brings the metrics and the config of a run and neither the weights nor
    the pointer to them. `resolve_run` is right to refuse; it is just not a sentence
    anyone can act on.

    `--run` is left to `resolve_run`, which already reports a named directory precisely.
    """
    if "--run" in argv:
        return

    import runs as runs_module

    try:
        runs_module.latest_run(ROOT / "runs")
    except runs_module.RunsError as exc:
        raise LaunchError(
            "{reason}\n"
            "  A trained checkpoint is not in the repository: model.pt and runs/latest.txt\n"
            "  are git-ignored (CLAUDE.md rule 4), so cloning does not bring them.\n"
            "  Copy a run directory from the training machine into runs/, write its name\n"
            "  into runs/latest.txt, and run this file again — or point at one directly:\n"
            "      python run.py --run <name>".format(reason=exc)
        ) from None


def start(argv: list[str]) -> int:
    """Preflight, then `tasks.app` — the same target `python tasks.py app` runs."""
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))

    check_port(port_from(argv))
    check_checkpoint(argv)

    import tasks

    tasks.app(argv)
    return 0


# --------------------------------------------------------------------------- #

def main(argv: list[str]) -> int:
    argv = normalise(argv)

    try:
        if in_venv():
            return start(argv)
        print("Wittenberg", flush=True)
        bootstrap()
        return relaunch(argv)
    except LaunchError as exc:
        print("\nERROR: {exc}".format(exc=exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print()
        return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
