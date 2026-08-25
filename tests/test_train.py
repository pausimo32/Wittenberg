"""Contract tests for `src/train.py` — `docs/04-model-spec.md` §3.

**Determinism is the one this project has actually promised.** `docs/06` §4: "Re-training
reproduces reported metrics within ±1 % (seed 42)". That is a claim in the written work, so
it is asserted here by training twice and comparing weights, not by trusting that the seeds
were set.

The rest are the properties that would let a run into `runs/` while being unreportable: a
missing preprocessing hash, a missing seed, or a synthetic run whose `metrics.json` does not
say so.

Runs are short on purpose — the gate for F4a is "it runs end to end and writes the
artefacts", never an accuracy figure.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from preprocess import load_config
from runs import check_preprocess_hash, latest_run, missing_artefacts
from train import HYPERPARAMETERS, SEED, TrainingError, configure_determinism, train


@pytest.fixture(scope="module")
def short_run(tiny_dataset, tmp_path_factory) -> Path:
    return train(tiny_dataset, epochs=2, runs_root=tmp_path_factory.mktemp("runs"))


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #

def test_the_same_seed_gives_the_same_weights(tiny_dataset, tmp_path):
    """The TDR claims this. Two runs, identical parameters, byte for byte."""
    first = train(tiny_dataset, epochs=2, runs_root=tmp_path / "a")
    second = train(tiny_dataset, epochs=2, runs_root=tmp_path / "b")

    a = torch.load(first / "model.pt", map_location="cpu", weights_only=False)["state_dict"]
    b = torch.load(second / "model.pt", map_location="cpu", weights_only=False)["state_dict"]
    assert a.keys() == b.keys()
    for key in a:
        assert torch.equal(a[key], b[key]), key


def test_a_different_seed_gives_different_weights(tiny_dataset, tmp_path):
    """Otherwise the test above would pass for the wrong reason."""
    first = train(tiny_dataset, epochs=2, runs_root=tmp_path / "a")
    other = train(tiny_dataset, epochs=2, runs_root=tmp_path / "b", seed=7)
    a = torch.load(first / "model.pt", map_location="cpu", weights_only=False)["state_dict"]
    b = torch.load(other / "model.pt", map_location="cpu", weights_only=False)["state_dict"]
    assert not all(torch.equal(a[k], b[k]) for k in a)


def test_determinism_is_configured_the_way_the_spec_says():
    settings = configure_determinism()
    assert settings["seed"] == SEED
    assert settings["deterministic_algorithms"] is True
    assert settings["cudnn_benchmark"] is False
    assert settings["cublas_workspace_config"] == ":4096:8"


def test_the_cublas_variable_is_set_in_the_environment():
    """cuBLAS reads it when the CUDA context is created, so setting it afterwards is
    setting it too late — and silently."""
    import os

    configure_determinism()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


# --------------------------------------------------------------------------- #
# the run directory
# --------------------------------------------------------------------------- #

def test_a_run_writes_every_artefact_the_spec_lists(short_run):
    """§3: config.json, metrics.json, history.csv, model.pt, report/."""
    assert missing_artefacts(short_run) == []
    assert (short_run / "report").is_dir()


def test_the_run_is_the_one_latest_points_at(short_run):
    assert latest_run(short_run.parent) == short_run


def test_the_checkpoint_passes_check_model(short_run):
    """The F4a gate. A run that cannot be verified against the current preprocessing is
    not servable and should never have been written."""
    assert check_preprocess_hash(short_run, load_config())["sha256"]


def test_the_checkpoint_carries_its_own_hash_too(short_run):
    """`model.pt` travels with the repository (ADR 0005); the API reads the hash from the
    checkpoint, not only from the run directory beside it."""
    checkpoint = torch.load(short_run / "model.pt", map_location="cpu", weights_only=False)
    assert checkpoint["preprocess_sha256"]
    config = json.loads((short_run / "config.json").read_text("utf-8"))
    assert checkpoint["preprocess_sha256"] == config["preprocess_sha256"]


def test_config_json_records_the_resolved_preprocessing(short_run):
    """§3: "full resolved config incl. preprocessing version". A hash with nothing to
    compare against is not auditable years later."""
    config = json.loads((short_run / "config.json").read_text("utf-8"))
    assert config["preprocess_config"] == load_config()
    assert config["preprocess_version"] == load_config()["version"]


def test_config_json_records_the_environment(short_run):
    config = json.loads((short_run / "config.json").read_text("utf-8"))
    for key in ("python", "torch", "opencv", "numpy", "platform"):
        assert config["environment"][key]


def test_metrics_json_reports_the_seed(short_run):
    """CLAUDE.md rule 5: "Report the seed in every metrics.json"."""
    assert json.loads((short_run / "metrics.json").read_text("utf-8"))["seed"] == SEED


def test_metrics_json_names_the_sessions_in_each_split(short_run):
    """docs/02 §2: an accuracy figure must name the sessions it was measured on, because
    adding a session re-partitions everything."""
    metrics = json.loads((short_run / "metrics.json").read_text("utf-8"))
    train_sessions = set(metrics["splits"]["train"]["sessions"])
    val_sessions = set(metrics["splits"]["val"]["sessions"])
    assert train_sessions and val_sessions
    assert train_sessions & val_sessions == set()


def test_a_synthetic_run_says_so_in_its_own_metrics(short_run):
    """The number in this file is ~100 % and means nothing. Anyone who opens it later must
    be told that without having to know where the data came from."""
    metrics = json.loads((short_run / "metrics.json").read_text("utf-8"))
    assert metrics["synthetic"] is True
    assert "TDR" in metrics["do_not_report"]


def test_history_csv_has_one_row_per_epoch(short_run):
    with (short_run / "history.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert set(rows[0]) == {"epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc"}


def test_the_hyperparameters_are_the_ones_the_spec_fixes():
    """§3, verbatim. Not knobs: an ablation moves these, in `docs/04` §5, not a hunch."""
    assert HYPERPARAMETERS["optimizer"] == "AdamW"
    assert HYPERPARAMETERS["lr"] == 1e-3
    assert HYPERPARAMETERS["lr_min"] == 1e-5
    assert HYPERPARAMETERS["batch_size"] == 32
    assert HYPERPARAMETERS["epochs"] == 80
    assert HYPERPARAMETERS["early_stopping_patience"] == 12
    assert HYPERPARAMETERS["label_smoothing"] == 0.05
    assert HYPERPARAMETERS["amp"] is False


def test_an_unknown_model_is_refused_before_anything_is_written(tiny_dataset, tmp_path):
    with pytest.raises(TrainingError, match="unknown model"):
        train(tiny_dataset, model_name="resnet50", runs_root=tmp_path)
    assert not tmp_path.exists() or not any(tmp_path.iterdir())


def test_the_baseline_trains_too(tiny_dataset, tmp_path):
    """§1 A exists to be compared against (A7), so it has to actually run."""
    run_dir = train(tiny_dataset, model_name="baseline_paper", epochs=1, runs_root=tmp_path)
    assert json.loads((run_dir / "metrics.json").read_text("utf-8"))["model"] == "baseline_paper"
