"""`data/metadata.csv` + `data/processed/` -> tensors, one split at a time.

The whole split is loaded into memory as `uint8` and kept there. `docs/04-model-spec.md` §3:
4 000 images as `uint8` is 65 MB, so a `DataLoader` re-reading files every epoch would leave
the device idling on Python for no reason. Indexing a resident tensor with a random
permutation removes that entirely.

**What is on disk is step 7, not step 8.** `tasks.py ingest` writes `Stages.resized` — the
128x128 `uint8` array after the resize. Step 8 (scale to `float32` in [0, 1]) happens here,
per batch, and it happens through `preprocess._scale`: the same function `src/preprocess.py`
uses, imported by its private name on purpose. The alternative is a second implementation of
step 8 in this file, and `CLAUDE.md` rule 1 forbids forking that pipeline — for a
multiplication as much as for a Canny threshold, because the failure mode is identical and
silent. `test_dataset.py` pins the batched path against it byte for byte.

**The NHWC -> NCHW permute lives here.** `docs/03` produces `(128, 128, 1)`; PyTorch wants
`(N, 1, 128, 128)`. §1 of the model spec puts the permute in the loader precisely so the
preprocessing contract stays framework-agnostic.

**Only `quality == ok` rows have an image.** A rejected frame has a `metadata.csv` row and
no file in `data/processed/`, by design (`docs/02` §6), so loading is driven by the rows
that carry a split, never by a glob over the directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import cv2
import numpy as np
import torch

from metadata import is_accepted, read_metadata
from model import classes_from_config

# Step 8 of docs/03 §2, by its private name. See the module docstring: importing it is
# deliberate, and it is the alternative to writing `* (1/255)` a second time.
from preprocess import _scale as scale_step

SPLITS = ("train", "val", "test")


class DatasetError(RuntimeError):
    """The split cannot be loaded as it stands."""


class Split(NamedTuple):
    """One partition, resident in memory.

    `images` is `(N, H, W)` `uint8` — step 7 exactly as written to `data/processed/`.
    `labels` is `(N,)` `int64`, the **frozen class index** from `config/classes.json`, not
    the frequency. `sessions` and `freqs` are kept per image so the evaluation can group by
    capture session (`docs/04` §4.1) and measure frequency distance (§4.5) without going
    back to the CSV.
    """

    name: str
    images: torch.Tensor
    labels: torch.Tensor
    sessions: list[str]
    freqs: list[int]
    image_ids: list[str]

    @property
    def size(self) -> int:
        """How many images. Deliberately not `__len__`: this is a `NamedTuple`, `len()` of
        one is its field count, and overriding that breaks `_replace` in ways that surface
        as a `TypeError` from inside the standard library."""
        return int(self.images.shape[0])

    @property
    def class_counts(self) -> dict[int, int]:
        return {int(c): int((self.labels == c).sum()) for c in self.labels.unique().tolist()}


def load_split(
    data_root: Path | str,
    split: str,
    classes: list[dict[str, Any]] | None = None,
) -> Split:
    """Load one split into memory. Raises rather than returning an empty partition."""
    if split not in SPLITS:
        raise DatasetError(f"unknown split {split!r} — one of {', '.join(SPLITS)}")

    root = Path(data_root)
    index = _class_index(classes if classes is not None else classes_from_config())
    rows = [r for r in read_metadata(root / "metadata.csv") if r["split"] == split]
    if not rows:
        raise DatasetError(
            f"no rows with split={split!r} in {root / 'metadata.csv'} — run "
            f"`python tasks.py ingest` and `python tasks.py split` first"
        )

    images, labels, sessions, freqs, image_ids = [], [], [], [], []
    missing: list[str] = []

    for row in sorted(rows, key=lambda r: r["image_id"]):
        if not is_accepted(row):
            # A row with a split that the gate rejected would be a splitter bug; the
            # splitter leaves those blank. Guard rather than silently skip.
            raise DatasetError(
                f"{row['image_id']} has split={split!r} but quality={row['quality']!r}; "
                f"only accepted frames belong to a partition (docs/02 §6)"
            )
        freq = int(row["freq_hz"])
        if freq not in index:
            raise DatasetError(
                f"{row['image_id']} is labelled {freq} Hz, which is not in "
                f"config/classes.json. The class list is frozen — re-ingest against the "
                f"current one rather than adding a class here."
            )
        path = root / "processed" / split / str(freq) / _processed_name(row)
        # Checked before `imread`, not after: `cv2.imread` prints its own `findDecoder`
        # warning per missing file, so a stale metadata.csv would bury the one message
        # that says what to do under one line of OpenCV noise per row.
        if not path.exists():
            missing.append(str(path.relative_to(root)).replace("\\", "/"))
            continue
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise DatasetError(f"{path} exists but is not a readable image")

        images.append(image)
        labels.append(index[freq])
        sessions.append(row["session_id"])
        freqs.append(freq)
        image_ids.append(row["image_id"])

    if missing:
        raise DatasetError(
            f"{len(missing)} row(s) of split={split!r} have no processed image, e.g. "
            f"{missing[0]}. metadata.csv and data/processed/ disagree — re-run "
            f"`python tasks.py ingest --all` then `python tasks.py split`."
        )

    return Split(
        name=split,
        images=torch.from_numpy(np.stack(images)),
        labels=torch.tensor(labels, dtype=torch.int64),
        sessions=sessions,
        freqs=freqs,
        image_ids=image_ids,
    )


def to_model_input(
    images: torch.Tensor, config: dict[str, Any], device: torch.device | str = "cpu"
) -> torch.Tensor:
    """`(N, H, W)` `uint8` -> `(N, 1, H, W)` `float32` in [0, 1], on `device`.

    Step 8 followed by the NHWC -> NCHW permute, in that order. `_scale` reshapes to
    `config["output"]["shape"]`, which describes **one** image, so the batch is scaled image
    by image and stacked rather than passed in whole. That costs a Python loop over 32
    arrays per step, which is nothing next to the convolutions, and it keeps step 8 in the
    one place `docs/03` defines it.
    """
    batch = np.stack([scale_step(image, config["output"]) for image in images.numpy()])
    return torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous().to(device)


def class_weights(split: Split, n_classes: int) -> torch.Tensor | None:
    """§2: class weights *if* any class has < 70 % of the median class count.

    Returns `None` when the set is balanced enough, so a balanced run carries no weights at
    all rather than a vector of ones — the two are numerically identical but only one of
    them says, in `metrics.json`, that reweighting was not needed. §2 also says to prefer
    collecting more images to reweighting, so this firing is a finding, not a fix.
    """
    counts = np.array([int((split.labels == c).sum()) for c in range(n_classes)], dtype=np.float64)
    if counts.min() == 0:
        raise DatasetError(
            f"split {split.name!r} has no images at all for class "
            f"{int(counts.argmin())} — it cannot be trained or scored on"
        )
    median = float(np.median(counts))
    if counts.min() >= 0.70 * median:
        return None
    return torch.tensor(median / counts, dtype=torch.float32)


def _class_index(classes: list[dict[str, Any]]) -> dict[int, int]:
    return {int(c["freq_hz"]): int(c["index"]) for c in classes}


def _processed_name(row: dict[str, Any]) -> str:
    """`<session_id>_<n>.png` — `docs/02` §4, rebuilt from the `image_id` primary key."""
    return f"{row['session_id']}_{row['image_id'].rsplit('_', 1)[-1]}.png"
