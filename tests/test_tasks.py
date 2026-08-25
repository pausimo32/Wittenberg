"""Contract tests for the two environment gates in `tasks.py`.

Both exist because the obvious version of the check passes on exactly the machine it is
supposed to protect, so both are tested by making them fail.

**`check-versions` checks the build, not the string.** `cv2.aruco` ships only in
`opencv-contrib-python` (`docs/03` §5.3). Plain `opencv-python` imports, reports the same
`cv2.__version__`, and raises `AttributeError` the first time `src/calibrate.py` is asked
for a dictionary — which is after the lab work, not before it.

**`check-device` runs a kernel, not `is_available()`.** A cu124 wheel on a Blackwell card
reports `True` and dies on the first operation (ADR 0005). That path cannot be tested here
without such a card, so what is pinned instead is that the target exists under the name the
documentation uses and that the old `check-gpu` is gone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import tasks  # noqa: E402


@pytest.fixture
def probe(monkeypatch):
    """Replace the subprocess probe, so a failure mode can be described rather than staged."""

    def install(**overrides):
        answer = {
            "distributions": ["opencv-contrib-python"],
            "distribution_version": "4.12.0.88",
            "cv2_version": "4.12.0",
        }
        answer.update(overrides)
        monkeypatch.setattr(tasks, "_probe_opencv", lambda: answer)

    return install


def test_the_pinned_build_passes(probe, capsys):
    probe()
    tasks.check_versions([])
    assert "OK" in capsys.readouterr().out


def test_plain_opencv_python_is_refused(probe):
    """The version string is identical, so only the distribution name catches this."""
    probe(distributions=["opencv-python"], distribution_version="4.12.0.88")
    with pytest.raises(SystemExit):
        tasks.check_versions([])


def test_the_refusal_names_aruco_and_the_calibration(probe, capsys):
    """The message has to say why, or someone will fix it by editing versions.json."""
    probe(distributions=["opencv-python"])
    with pytest.raises(SystemExit):
        tasks.check_versions([])
    message = capsys.readouterr().err
    assert "aruco" in message
    assert "calibrate.py" in message


def test_both_opencv_distributions_installed_is_refused(probe, capsys):
    """§5.3: they provide the same `cv2` and which one wins depends on install order."""
    probe(distributions=["opencv-contrib-python", "opencv-python"])
    with pytest.raises(SystemExit):
        tasks.check_versions([])
    assert "install order" in capsys.readouterr().err


def test_a_contrib_build_whose_aruco_does_not_work_is_refused(probe):
    probe(aruco_error="AttributeError: module 'cv2' has no attribute 'aruco'")
    with pytest.raises(SystemExit):
        tasks.check_versions([])


def test_a_different_minor_version_is_refused(probe, capsys):
    probe(cv2_version="5.0.0", distribution_version="5.0.0.93")
    with pytest.raises(SystemExit):
        tasks.check_versions([])
    assert "0006" in capsys.readouterr().err


def test_a_patch_release_of_the_pin_is_accepted(probe):
    """The pin is a minor version; `4.12.0.88` and `4.12.1` are the same baseline."""
    probe(cv2_version="4.12.1")
    tasks.check_versions([])


def test_an_unimportable_cv2_is_refused(probe):
    probe(import_error="ModuleNotFoundError: No module named 'cv2'", distributions=[])
    with pytest.raises(SystemExit):
        tasks.check_versions([])


# --------------------------------------------------------------------------- #
# the target registry
# --------------------------------------------------------------------------- #

def test_check_device_is_the_name_the_docs_use():
    assert "check-device" in tasks.TARGETS


def test_check_gpu_is_gone():
    """ADR 0005 renamed it, and the master plan's F0 definition of done says so."""
    assert "check-gpu" not in tasks.TARGETS


def test_setup_runs_both_gates():
    source = (REPO_ROOT / "tasks.py").read_text(encoding="utf-8")
    body = source[source.index("def setup("): source.index("@target(\"check-versions\"")]
    assert "check_versions([])" in body
    assert "check_device([])" in body


def test_no_target_still_claims_f3():
    """F3 split into F3a (code) and F3b (lab); a bare [F3] label names neither."""
    labels = [help_text for _fn, help_text in tasks.TARGETS.values()]
    assert not any("[F3]" in label for label in labels)
