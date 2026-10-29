"""Contract tests for `run.py`, the one-command launcher.

Four of its properties are load-bearing and every one of them fails silently rather than
loudly, so each is pinned by making it break.

**It must import nothing but the standard library.** It is run by whatever Python the user
has — on the current machine `python3` is 3.14, and the venv it is about to build is 3.11.
A single `import numpy` at module level turns the launcher into something that only works
after the thing it exists to install.

**It must not run npm.** `tasks.py setup` does, for the Vite project ADR 0007 replaced. A
launcher that inherited that step would refuse to start, on a machine without Node, an app
that has no build step and would have run there perfectly.

**It must not reimplement the startup sequence.** The hash gate lives in `tasks.app` and
`docs/05` §3 forbids softening it; a second copy of that sequence here is a second place to
soften it in.

**It must build the venv with 3.11.** `requirements.txt` pins exact wheels, and pip resolves
those per Python version — a 3.14 venv installs different code and nothing says so.
"""

from __future__ import annotations

import ast
import socket
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import run  # noqa: E402


# --------------------------------------------------------------------------- #
# stdlib only
# --------------------------------------------------------------------------- #

def _executable_source(path: Path) -> str:
    """The file's code with comments and docstrings removed.

    A source scan that reads the prose finds every word the prose uses to explain why the
    word is absent — including this one. Round-tripping through the AST leaves only what
    actually runs.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        first = node.body[0] if node.body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            node.body.pop(0)
            if not node.body:
                node.body.append(ast.Pass())
    return ast.unparse(tree)


def _top_level_imports(path: Path) -> set[str]:
    """The root package of every import in the file, at any nesting depth."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_the_launcher_imports_only_the_standard_library():
    """It runs before the dependencies exist. Anything else here is a bootstrap deadlock."""
    imported = _top_level_imports(REPO_ROOT / "run.py")
    # `tasks` and `runs` are repository modules, imported inside functions that only ever
    # run under `.venv` — they are the handover, not a dependency of the bootstrap.
    repository_modules = {"tasks", "runs"}
    third_party = imported - sys.stdlib_module_names - repository_modules - {"__future__"}
    assert not third_party, "run.py imports {names} — it must reach the venv first".format(
        names=sorted(third_party)
    )


def test_the_repository_imports_are_not_at_module_level():
    """`import tasks` at the top would run before `sys.path` carries `src/`."""
    tree = ast.parse((REPO_ROOT / "run.py").read_text(encoding="utf-8"))
    module_level: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            module_level.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            module_level.add(node.module.split(".")[0])
    assert "tasks" not in module_level
    assert "runs" not in module_level


# --------------------------------------------------------------------------- #
# interpreter selection
# --------------------------------------------------------------------------- #

@pytest.fixture
def interpreters(monkeypatch):
    """Describe which interpreters exist and what version each reports."""

    def install(versions: dict[str, tuple[int, int]]):
        monkeypatch.setattr(run, "IS_WINDOWS", False)
        monkeypatch.setattr(run.shutil, "which", lambda name: name if name in versions else None)
        monkeypatch.setattr(run, "python_version", lambda exe: versions.get(exe))

    return install


def test_a_python_311_on_the_path_is_chosen(interpreters):
    interpreters({sys.executable: (3, 14), "python3.11": (3, 11)})
    assert run.interpreter_for_venv() == "python3.11"


def test_the_running_interpreter_is_used_when_it_is_already_311(interpreters):
    interpreters({sys.executable: (3, 11), "python3.11": (3, 11)})
    assert run.interpreter_for_venv() == sys.executable


def test_a_newer_python_is_refused_rather_than_accepted(interpreters):
    """3.14 would build a venv resolving wheels `requirements.txt` never named."""
    interpreters({sys.executable: (3, 14)})
    with pytest.raises(run.LaunchError, match="3.11"):
        run.interpreter_for_venv()


def test_an_older_python_is_refused(interpreters):
    interpreters({sys.executable: (3, 10), "python3.11": (3, 10)})
    with pytest.raises(run.LaunchError):
        run.interpreter_for_venv()


def test_the_refusal_says_which_version_is_running(interpreters):
    """Otherwise the reader has no way to tell whether they already fixed it.

    The version quoted is this process's own, not the probe's: the sentence is about the
    Python that was typed, which is the one the reader has to change.
    """
    interpreters({sys.executable: (3, 14)})
    running = "%d.%d" % sys.version_info[:2]
    with pytest.raises(run.LaunchError) as caught:
        run.interpreter_for_venv()
    assert "this one is {running}".format(running=running) in str(caught.value)


def test_the_version_is_probed_not_read_off_the_filename(interpreters):
    """A `python3.11` on PATH is a name, not a promise."""
    interpreters({sys.executable: (3, 14), "python3.11": (3, 9)})
    with pytest.raises(run.LaunchError):
        run.interpreter_for_venv()


# --------------------------------------------------------------------------- #
# dependencies
# --------------------------------------------------------------------------- #

@pytest.fixture
def venv(monkeypatch, tmp_path):
    """A fake `.venv` whose python exists, with the stamp under this test's control."""
    root = tmp_path / "repo"
    (root / ".venv").mkdir(parents=True)
    requirements = root / "requirements.txt"
    requirements.write_text("torch==2.6.0\n", encoding="utf-8")
    python = root / ".venv" / "python"
    python.write_text("", encoding="utf-8")

    monkeypatch.setattr(run, "ROOT", root)
    monkeypatch.setattr(run, "VENV", root / ".venv")
    monkeypatch.setattr(run, "REQUIREMENTS", requirements)
    monkeypatch.setattr(run, "PY", python)
    monkeypatch.setattr(run, "STAMP", root / ".venv" / ".wittenberg-deps")
    return root


def test_an_unchanged_requirements_file_installs_nothing(venv, monkeypatch):
    """The second run must not spend four minutes proving pip is idempotent."""
    run.STAMP.write_text(run.requirements_digest() + "\n", encoding="utf-8")
    monkeypatch.setattr(run, "install_dependencies", _forbidden("pip"))
    run.bootstrap()


def test_an_edited_requirements_file_installs_again(venv, monkeypatch):
    run.STAMP.write_text("stale-digest\n", encoding="utf-8")
    calls: list[bool] = []
    monkeypatch.setattr(run, "install_dependencies", lambda: calls.append(True))
    run.bootstrap()
    assert calls == [True]


def test_a_venv_without_a_stamp_installs(venv, monkeypatch):
    """A `.venv` made by hand, or by `tasks.py setup`, has no stamp and may be anything."""
    calls: list[bool] = []
    monkeypatch.setattr(run, "install_dependencies", lambda: calls.append(True))
    run.bootstrap()
    assert calls == [True]


def test_the_stamp_is_written_only_after_pip_succeeds(venv, monkeypatch):
    """A stamp written first would make a failed install look complete on the next run."""
    def explode(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(run.subprocess, "run", explode)
    with pytest.raises(run.LaunchError):
        run.install_dependencies()
    assert not run.STAMP.exists()


def test_a_failed_install_says_the_network_is_needed_only_here(venv, monkeypatch, capsys):
    """ADR 0005 promises an offline app; the promise is about running it, not installing it."""
    def explode(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(run.subprocess, "run", explode)
    with pytest.raises(run.LaunchError, match="internet"):
        run.install_dependencies()


def test_the_launcher_never_shells_out_to_npm(venv, monkeypatch):
    """The app serves `site/`: plain HTML, CSS and browser JS, with nothing to build."""
    monkeypatch.setattr(run.shutil, "which", _forbidden("npm lookup"))
    monkeypatch.setattr(run.subprocess, "run", _forbidden("a subprocess"))
    run.STAMP.write_text(run.requirements_digest() + "\n", encoding="utf-8")
    run.bootstrap()

    assert "npm" not in _executable_source(REPO_ROOT / "run.py")


# --------------------------------------------------------------------------- #
# argument handling
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (["--port=9000"], ["--port", "9000"]),
        (["--port", "9000"], ["--port", "9000"]),
        (["--run=20260902-192239"], ["--run", "20260902-192239"]),
        (["--no-open"], ["--no-open"]),
        ([], []),
    ],
)
def test_equals_form_flags_are_normalised(given, expected):
    """`tasks._option` reads the next element, so `--port=9000` is silently the default."""
    assert run.normalise(given) == expected


def test_a_port_that_is_not_a_number_is_a_message():
    with pytest.raises(run.LaunchError, match="wants a number"):
        run.port_from(["--port", "huit-mille"])


def test_a_port_flag_without_a_value_is_a_message():
    with pytest.raises(run.LaunchError, match="--port"):
        run.port_from(["--port"])


def test_no_port_flag_means_the_default():
    assert run.port_from(["--no-open"]) == run.DEFAULT_PORT


# --------------------------------------------------------------------------- #
# preflight
# --------------------------------------------------------------------------- #

def test_a_busy_port_is_refused_before_the_browser_opens():
    """`tasks.app` opens the browser and then binds; a collision would land on a dead tab."""
    held = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    held.bind(("127.0.0.1", 0))
    held.listen(1)
    port = held.getsockname()[1]
    try:
        with pytest.raises(run.LaunchError, match="already in use"):
            run.check_port(port)
    finally:
        held.close()


def test_a_free_port_passes_and_is_released():
    """The probe must not keep the socket uvicorn is about to want."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    run.check_port(port)
    run.check_port(port)


def test_a_missing_checkpoint_names_model_pt_and_latest_txt(monkeypatch, tmp_path):
    """A fresh clone has neither: `.gitignore` excludes `*.pt` and `runs/*`."""
    monkeypatch.setattr(run, "ROOT", tmp_path)
    (tmp_path / "runs").mkdir()
    with pytest.raises(run.LaunchError) as caught:
        run.check_checkpoint([])
    message = str(caught.value)
    assert "model.pt" in message
    assert "runs/latest.txt" in message


def test_an_explicit_run_is_left_to_resolve_run(monkeypatch, tmp_path):
    """`resolve_run` already reports a named directory precisely; two messages is one too many."""
    monkeypatch.setattr(run, "ROOT", tmp_path)
    run.check_checkpoint(["--run", "20260902-192239"])


# --------------------------------------------------------------------------- #
# the handover
# --------------------------------------------------------------------------- #

def test_starting_delegates_to_the_app_target(monkeypatch):
    """The gate, the banner and `uvicorn.run` stay in one place — `docs/05` §3."""
    import tasks

    seen: list[list[str]] = []
    monkeypatch.setattr(tasks, "app", lambda argv: seen.append(argv))
    monkeypatch.setattr(run, "check_port", lambda port: None)
    monkeypatch.setattr(run, "check_checkpoint", lambda argv: None)

    assert run.start(["--no-open", "--port", "8001"]) == 0
    assert seen == [["--no-open", "--port", "8001"]]


def test_the_launcher_does_not_reimplement_the_startup_sequence():
    """`create_app`, the hash gate and `uvicorn.run` must appear nowhere in this file."""
    source = _executable_source(REPO_ROOT / "run.py")
    for forbidden in ("create_app(", "uvicorn.run", "load_serving(", "check_preprocess_hash"):
        assert forbidden not in source, "{f} belongs to tasks.app".format(f=forbidden)


def test_main_normalises_before_handing_over(monkeypatch):
    seen: list[list[str]] = []
    monkeypatch.setattr(run, "in_venv", lambda: True)
    monkeypatch.setattr(run, "start", lambda argv: seen.append(argv) or 0)

    assert run.main(["--port=9000"]) == 0
    assert seen == [["--port", "9000"]]


def test_a_launch_error_is_a_message_not_a_traceback(monkeypatch, capsys):
    def refuse():
        raise run.LaunchError("no Python 3.11 found")

    monkeypatch.setattr(run, "in_venv", lambda: False)
    monkeypatch.setattr(run, "bootstrap", refuse)

    assert run.main([]) == 1
    assert "no Python 3.11 found" in capsys.readouterr().err


def test_ctrl_c_is_a_clean_exit(monkeypatch):
    """Stopping the server is how a field session ends, not a crash."""
    monkeypatch.setattr(run, "in_venv", lambda: True)

    def interrupted(argv):
        raise KeyboardInterrupt

    monkeypatch.setattr(run, "start", interrupted)
    assert run.main([]) == 0


def _forbidden(what: str):
    def fail(*args: object, **kwargs: object):
        raise AssertionError("run.py must not use {what} here".format(what=what))

    return fail
