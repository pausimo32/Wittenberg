"""Contract tests for `src/evaluate.py` — `docs/04-model-spec.md` §4.

§4 lists six mandatory outputs and §4.6 says a run without Grad-CAM is not evaluated. So the
first thing pinned here is that all six exist after `evaluate_run`, by filename: a partial
evaluation that exits 0 is how a TDR ends up citing a figure nobody produced.

The Wilson interval gets its own tests because it is quoted in the written work and because
the reason it is Wilson rather than the normal approximation is a small-sample argument
`docs/02` §5 makes with a specific number.

Nothing here asserts an accuracy. The data is synthetic; the model reaches ~100 % and that
figure measures how well it inverted the formula that drew the images.
"""

from __future__ import annotations

import csv
import json

import pytest

from evaluate import GRADCAM_PER_CLASS, evaluate_run, wilson_interval
from model import classes_from_config
from runs import PreprocessMismatch
from train import train


@pytest.fixture(scope="module")
def evaluated(tiny_dataset, tmp_path_factory):
    run_dir = train(tiny_dataset, epochs=3, runs_root=tmp_path_factory.mktemp("runs"))
    return run_dir, evaluate_run(run_dir, tiny_dataset)


# --------------------------------------------------------------------------- #
# all six outputs of §4, by name
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "artefact",
    [
        "classification_report.csv",      # §4.2
        "confusion_matrix.csv",           # §4.3
        "confusion_matrix_normalised.csv",
        "confusion_matrix.png",
        "learning_curves.png",            # §4.4
        "frequency_distance.csv",         # §4.5
        "frequency_distance.png",
        "evaluation.json",
    ],
)
def test_every_mandatory_output_is_written(evaluated, artefact):
    run_dir, _ = evaluated
    assert (run_dir / "report" / artefact).exists()


def test_gradcam_is_produced_for_every_class(evaluated):
    """§4.6: "Non-negotiable — without it there is no evidence the model learned the
    physics"."""
    run_dir, summary = evaluated
    classes = classes_from_config()
    assert set(summary["gradcam"]["written"]) == {str(c["freq_hz"]) for c in classes}
    assert list((run_dir / "report" / "gradcam").glob("*.png"))


def test_gradcam_wants_three_of_each_kind_per_class(evaluated):
    """Three correct and three incorrect. A class the model gets entirely right cannot
    supply three incorrect ones, and the summary says so rather than quietly succeeding."""
    _run_dir, summary = evaluated
    assert summary["gradcam"]["per_class_target"] == GRADCAM_PER_CLASS == 3
    for counts in summary["gradcam"]["written"].values():
        assert counts["correct"] <= GRADCAM_PER_CLASS
        assert counts["incorrect"] <= GRADCAM_PER_CLASS
    if summary["gradcam"]["shortfall"]:
        assert "split being small" in summary["gradcam"]["shortfall_note"]


def test_the_headline_numbers_reach_metrics_json(evaluated):
    """`metrics.json` is one of the two files CLAUDE.md rule 4 commits, so a number in the
    TDR body has something committed to be traced back to (docs/06 §2)."""
    run_dir, _ = evaluated
    metrics = json.loads((run_dir / "metrics.json").read_text("utf-8"))
    for key in ("accuracy", "accuracy_wilson_95", "macro_f1", "worst_class_recall", "errors"):
        assert key in metrics["evaluation"]


def test_the_evaluation_names_the_sessions_it_measured(evaluated):
    _run_dir, summary = evaluated
    assert summary["sessions"]
    assert summary["accuracy"]["total"] == summary["images"]


# --------------------------------------------------------------------------- #
# §4.1 — the interval, and what it is an interval over
# --------------------------------------------------------------------------- #

def test_wilson_matches_the_figure_the_dataset_spec_quotes():
    """docs/02 §5: "4 correct out of 5 gives a 95 % Wilson interval of roughly 38-96 %"."""
    low, high = wilson_interval(4, 5)
    assert 0.36 < low < 0.40
    assert 0.94 < high < 0.98


def test_wilson_never_leaves_zero_to_one():
    """The normal approximation gives 45-115 % for 4 of 5, which is not a probability."""
    for successes, total in [(0, 5), (5, 5), (1, 3), (30, 30), (0, 1)]:
        low, high = wilson_interval(successes, total)
        assert 0.0 <= low <= high <= 1.0


def test_a_perfect_score_still_has_an_interval():
    """30 of 30 is not certainty. This is the row of §5 that argues for a bigger test set."""
    low, high = wilson_interval(30, 30)
    assert low < 0.9        # ~0.886 — 30 of 30 is not evidence of 100 %
    assert high == pytest.approx(1.0)


def test_more_samples_narrow_the_interval():
    narrow = wilson_interval(80, 100)
    wide = wilson_interval(8, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_an_empty_split_does_not_divide_by_zero():
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_the_frame_level_interval_is_labelled_optimistic(evaluated):
    """docs/02 §5: a burst with the rig untouched is one observation, not 60. Reporting the
    frame-level number unlabelled is how a 5-image test set comes to look precise."""
    _run_dir, summary = evaluated
    assert "one observation" in summary["accuracy"]["frame_level_note"]
    assert "session" in summary["accuracy"]["session_level_note"]


def test_accuracy_is_reported_per_session_as_well(evaluated):
    """§4.1: the independent unit is the session, so the per-session numbers are what a
    leave-one-session-out interval will eventually be built from."""
    _run_dir, summary = evaluated
    assert summary["accuracy"]["per_session"]
    assert summary["accuracy"]["independent_units"] == len(summary["accuracy"]["per_session"])


# --------------------------------------------------------------------------- #
# §4.3 and §4.5 — the files a TDR annex is made of
# --------------------------------------------------------------------------- #

def test_the_confusion_matrix_csv_is_square_and_labelled(evaluated):
    """Annex 9 is this CSV. It has to be readable without the code that made it."""
    run_dir, _ = evaluated
    with (run_dir / "report" / "confusion_matrix.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    frequencies = [str(c["freq_hz"]) for c in classes_from_config()]
    assert rows[0][1:] == frequencies
    assert [r[0] for r in rows[1:]] == frequencies


def test_the_confusion_counts_add_up_to_the_split(evaluated):
    run_dir, summary = evaluated
    with (run_dir / "report" / "confusion_matrix.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))[1:]
    assert sum(int(v) for row in rows for v in row[1:]) == summary["images"]


def test_the_per_class_report_names_the_worst_class(evaluated):
    """§6's acceptance gate is worst-class recall, so it is computed, not eyeballed."""
    _run_dir, summary = evaluated
    assert summary["per_class"]["worst_class_hz"] in [
        int(c["freq_hz"]) for c in classes_from_config()
    ]
    assert 0.0 <= summary["per_class"]["worst_class_recall"] <= 1.0


def test_the_frequency_distance_analysis_answers_the_question_it_exists_for(evaluated):
    """§4.5: whether confusions cluster among neighbouring modes. With no errors there is
    nothing to cluster, and the fields say so rather than reporting a made-up zero."""
    _run_dir, summary = evaluated
    distance = summary["frequency_distance"]
    if distance["errors"] == 0:
        assert distance["median_distance_hz"] is None
        assert distance["adjacent_class_share"] is None
    else:
        assert distance["median_distance_hz"] >= 0
        assert 0.0 <= distance["adjacent_class_share"] <= 1.0


# --------------------------------------------------------------------------- #
# the gate applies to evaluation too
# --------------------------------------------------------------------------- #

def test_a_checkpoint_from_another_preprocessing_is_not_evaluated(evaluated, tiny_dataset):
    """A number measured with a checkpoint that reads a different edge map describes
    nothing, so it must not be produced at all (docs/03 §4)."""
    run_dir, _ = evaluated
    config_path = run_dir / "config.json"
    original = config_path.read_text("utf-8")
    stored = json.loads(original)
    stored["preprocess_sha256"] = "0" * 64
    config_path.write_text(json.dumps(stored), encoding="utf-8")
    try:
        with pytest.raises(PreprocessMismatch):
            evaluate_run(run_dir, tiny_dataset)
    finally:
        config_path.write_text(original, encoding="utf-8")


# --------------------------------------------------------------------------- #
# both architectures, because they fail differently
# --------------------------------------------------------------------------- #

def test_the_baseline_is_evaluated_too(tiny_dataset, tmp_path):
    """`baseline_paper` puts its ReLU directly after the layer Grad-CAM hooks, so an
    in-place ReLU there raises "view+inplace" during the backward pass — a failure
    `wittenberg_v1` never sees, because a BatchNorm sits in between. §1 A exists to be
    compared against (A7), and a model that cannot be evaluated cannot be compared.
    """
    run_dir = train(tiny_dataset, model_name="baseline_paper", epochs=1, runs_root=tmp_path)
    summary = evaluate_run(run_dir, tiny_dataset)
    assert list((run_dir / "report" / "gradcam").glob("*.png"))
    assert summary["gradcam"]["written"]


def test_no_relu_is_inplace_in_either_model():
    """The bug above is silent until Grad-CAM runs, which is at the end of a training run."""
    from torch import nn

    from model import MODEL_NAMES, build_model

    for name in MODEL_NAMES:
        model = build_model(name, 6)
        relus = [m for m in model.modules() if isinstance(m, nn.ReLU)]
        assert relus
        assert not any(m.inplace for m in relus), name
