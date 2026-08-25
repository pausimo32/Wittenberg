"""Contract tests for `src/dataset.py` — `data/processed/` -> tensors.

The load-bearing property is that **step 8 is not re-implemented here.** What is on disk is
`Stages.resized` (step 7); the scale to `float32` in [0, 1] happens in the loader, and if it
drifted from `src/preprocess.py` by so much as a rounding rule the model would be trained on
one thing and served another, with nothing to say so. `CLAUDE.md` rule 1 forbids the fork;
this file pins the equality byte for byte.

After that: labels are the frozen class *index*, never the frequency; the NHWC -> NCHW
permute happens here and not in `preprocess.py`; and a class missing from a fold is refused
rather than trained around.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dataset import DatasetError, class_weights, load_split, to_model_input
from preprocess import _scale, load_config


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def train_split(tiny_dataset, tiny_classes):
    return load_split(tiny_dataset, "train", tiny_classes)


# --------------------------------------------------------------------------- #
# step 8 is preprocess.py's, not this module's
# --------------------------------------------------------------------------- #

def test_the_batched_scale_equals_preprocess_step_8(train_split, config):
    """Byte for byte. This is the assertion that keeps the pipeline unforked."""
    batch = to_model_input(train_split.images[:8], config)
    for i in range(8):
        reference = _scale(train_split.images[i].numpy(), config["output"])   # (H, W, 1)
        assert np.array_equal(batch[i, 0].numpy(), reference[..., 0])


def test_changing_the_configured_scale_changes_what_the_model_is_fed(train_split, config):
    """If it did not, `config/preprocess.json` would not be the single source of truth and
    `tasks.py check-model` would be guarding a value nothing reads."""
    halved = {**config, "output": {**config["output"], "scale": config["output"]["scale"] / 2}}
    assert not torch.equal(
        to_model_input(train_split.images[:4], config),
        to_model_input(train_split.images[:4], halved),
    )


def test_the_model_input_is_nchw_float32_in_zero_one(train_split, config):
    batch = to_model_input(train_split.images[:8], config)
    assert batch.shape == (8, 1, 128, 128)
    assert batch.dtype == torch.float32
    assert float(batch.min()) >= 0.0
    assert float(batch.max()) <= 1.0


def test_what_is_stored_is_uint8_step_7(train_split):
    """§3 of docs/04: the dataset stays resident as uint8 — 65 MB for 4 000 images."""
    assert train_split.images.dtype == torch.uint8
    assert train_split.images.shape[1:] == (128, 128)


# --------------------------------------------------------------------------- #
# labels are class indices, and they come from the frozen list
# --------------------------------------------------------------------------- #

def test_labels_are_class_indices_not_frequencies(train_split, tiny_classes):
    classes = tiny_classes
    assert train_split.labels.dtype == torch.int64
    assert int(train_split.labels.max()) < len(classes)
    index = {int(c["freq_hz"]): int(c["index"]) for c in classes}
    for label, freq in zip(train_split.labels.tolist(), train_split.freqs):
        assert label == index[freq]


def test_a_frequency_absent_from_the_class_list_is_refused(tiny_dataset, tiny_classes):
    """The class index is frozen. Inventing an index here would void every checkpoint."""
    classes = [c for c in tiny_classes if int(c["freq_hz"]) != 420]
    renumbered = [{**c, "index": i} for i, c in enumerate(classes)]
    with pytest.raises(DatasetError, match="classes.json"):
        load_split(tiny_dataset, "train", renumbered)


def test_every_split_is_disjoint_by_session(tiny_dataset, tiny_classes):
    """docs/adr/0002. If this ever fails, every accuracy figure in the TDR is inflated."""
    sessions = {
        name: set(load_split(tiny_dataset, name, tiny_classes).sessions)
        for name in ("train", "val", "test")
    }
    assert sessions["train"] & sessions["val"] == set()
    assert sessions["train"] & sessions["test"] == set()
    assert sessions["val"] & sessions["test"] == set()


def test_rows_are_loaded_from_metadata_not_from_a_glob(tiny_dataset, tiny_classes):
    """A rejected frame has a row and no file (docs/02 §6); globbing the directory would
    also pick up anything left behind by an older ingest."""
    split = load_split(tiny_dataset, "test", tiny_classes)
    assert split.size == len(split.image_ids) == len(split.sessions)
    assert len(set(split.image_ids)) == len(split.image_ids)


def test_an_unknown_split_name_is_refused(tiny_dataset, tiny_classes):
    with pytest.raises(DatasetError, match="unknown split"):
        load_split(tiny_dataset, "holdout", tiny_classes)


def test_an_empty_split_names_the_targets_that_fill_it(tmp_path):
    (tmp_path / "metadata.csv").write_text(
        "image_id,session_id,freq_hz,split,material,grain_mass_g,distance_mm,lighting,"
        "dwell_s,roi_mode,quality,notes\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="tasks.py ingest"):
        load_split(tmp_path, "train", [{"index": 0, "freq_hz": 420}])


# --------------------------------------------------------------------------- #
# class weights — §2
# --------------------------------------------------------------------------- #

def test_a_balanced_split_gets_no_weights(train_split, tiny_classes):
    """§2 weights only when a class drops below 70 % of the median. A vector of ones would
    be numerically identical and would hide, in metrics.json, that it was not needed."""
    assert class_weights(train_split, len(tiny_classes)) is None


def test_an_underrepresented_class_gets_weighted(train_split, tiny_classes):
    starved = train_split._replace(
        labels=torch.cat([train_split.labels[train_split.labels != 0], torch.tensor([0])])
    )
    weights = class_weights(starved, len(tiny_classes))
    assert weights is not None
    assert float(weights[0]) == max(float(w) for w in weights)


def test_a_class_with_no_images_at_all_is_refused(train_split, tiny_classes):
    """It cannot be trained on and it cannot be scored. Silently weighting it by infinity
    would produce a run whose worst-class recall is undefined rather than zero."""
    kept = train_split._replace(labels=train_split.labels[train_split.labels != 0])
    with pytest.raises(DatasetError, match="no images"):
        class_weights(kept, len(tiny_classes))
