"""What OpenCV build is actually installed — the half of the guard the config hash misses.

`docs/03-preprocessing-spec.md` §4 makes the SHA-256 of `config/preprocess.json` the thing that
stops a checkpoint being served under a preprocessing it was not trained on. That covers the
*parameters*. It does not cover the *implementation*: reinstall the venv with a different
OpenCV and the hash still matches, while `docs/adr/0006` measures `cv2.warpPerspective` moving
up to 7 grey levels in every frame, all 120 tensors differing, and up to 5 298 of 262 144 edge
pixels flipping state.

So the version is a second key of the same guard, and it lives here rather than being read
twice: `src/train.py` stamps it into every checkpoint, `src/runs.py` checks it, and
`tasks.py check-versions` asks the venv the same question about the same list of distributions.

`cv2.__version__` is what is compared, not the wheel version. A wheel bump from
`opencv-contrib-python` 4.12.0.88 to 4.12.0.90 leaves `cv2.__version__` at `4.12.0` and changes
no pixel; a move to 5.0.0 changes both. The wheel version is recorded anyway, for the audit
trail — it is what `pip install` takes as an argument.
"""

from __future__ import annotations

from typing import Any

#: Every distribution that installs a module called `cv2`. Exactly one may be present:
#: they provide the same module and which one wins depends on install order
#: (`docs/03-preprocessing-spec.md` §5.3).
OPENCV_DISTRIBUTIONS = (
    "opencv-python",
    "opencv-contrib-python",
    "opencv-python-headless",
    "opencv-contrib-python-headless",
)


def installed_opencv_distributions() -> dict[str, str]:
    """`{distribution: wheel version}` for whichever OpenCV wheels are installed."""
    from importlib import metadata

    found: dict[str, str] = {}
    for name in OPENCV_DISTRIBUTIONS:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return found


def opencv_signature() -> dict[str, Any]:
    """What a checkpoint records, and what `check-model` compares.

    `opencv_version` is the gate. `opencv_package` is the gate too — the contrib and
    non-contrib builds are different software, and only one of them has `cv2.aruco`. The wheel
    version is informational.
    """
    import cv2

    installed = installed_opencv_distributions()
    return {
        "opencv_version": cv2.__version__,
        "opencv_package": next(iter(installed), None),
        "opencv_wheel_version": next(iter(installed.values()), None),
        "opencv_distributions_installed": sorted(installed),
    }
