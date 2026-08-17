"""Interprétation des probabilités : ordre, seuils et vocabulaire."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.conftest import manifest_dict, write_manifest
from wittenberg.model.decision import decide, softmax, to_probabilities
from wittenberg.model.manifest import OutputActivation, load_manifest


@pytest.fixture
def manifest(tmp_path: Path):  # type: ignore[no-untyped-def]
    return load_manifest(write_manifest(tmp_path, manifest_dict()))


def test_softmax_sums_to_one() -> None:
    result = softmax(np.array([2.0, 1.0, 0.1, -3.0]))

    assert result.sum() == pytest.approx(1.0)
    assert np.argmax(result) == 0


def test_softmax_is_numerically_stable() -> None:
    result = softmax(np.array([1000.0, 1001.0, 999.0]))

    assert np.all(np.isfinite(result))
    assert result.sum() == pytest.approx(1.0)


def test_logits_are_converted() -> None:
    result = to_probabilities(np.array([5.0, 1.0, 0.0]), OutputActivation.LOGITS)

    assert result.sum() == pytest.approx(1.0)


def test_softmax_output_is_passed_through() -> None:
    values = np.array([0.7, 0.2, 0.1])
    result = to_probabilities(values, OutputActivation.SOFTMAX)

    np.testing.assert_allclose(result, values)


def test_unnormalised_softmax_claim_is_renormalised() -> None:
    """Le manifeste annonce softmax mais la somme vaut 2 : on renormalise."""
    result = to_probabilities(np.array([1.0, 0.6, 0.4]), OutputActivation.SOFTMAX)

    assert result.sum() == pytest.approx(1.0)
    assert result[0] == pytest.approx(0.5)


def test_top_k_is_ordered_and_limited(manifest) -> None:  # type: ignore[no-untyped-def]
    decision = decide(np.array([0.1, 0.6, 0.05, 0.25]), manifest, top_k=3)

    assert [entry.index for entry in decision.top_k] == [1, 3, 0]
    assert decision.best.index == 1
    assert decision.best.label == "222 Hz"
    assert decision.best.frequency_hz == 222.0


def test_labels_follow_manifest_index_not_probability_order(manifest) -> None:  # type: ignore[no-untyped-def]
    """La probabilité la plus forte à l'index 3 doit donner l'étiquette de l'index 3."""
    decision = decide(np.array([0.05, 0.05, 0.1, 0.8]), manifest)

    assert decision.best.label == "444 Hz"
    assert decision.best.frequency_hz == 444.0


def test_high_confidence_is_reliable(manifest) -> None:  # type: ignore[no-untyped-def]
    decision = decide(np.array([0.9, 0.04, 0.03, 0.03]), manifest)

    assert decision.reliable is True
    assert decision.reason is None


def test_low_confidence_is_flagged(manifest) -> None:  # type: ignore[no-untyped-def]
    decision = decide(np.array([0.3, 0.28, 0.22, 0.2]), manifest)

    assert decision.reliable is False
    assert decision.reason == "low_confidence"
    assert decision.best.probability == pytest.approx(0.3)


def test_margin_rule_detects_ambiguity(tmp_path: Path) -> None:
    manifest = load_manifest(
        write_manifest(
            tmp_path, manifest_dict(decision={"min_confidence": 0.4, "min_margin": 0.15})
        )
    )
    decision = decide(np.array([0.45, 0.44, 0.06, 0.05]), manifest)

    assert decision.reliable is False
    assert decision.reason == "ambiguous"
    assert decision.margin == pytest.approx(0.01, abs=1e-6)


def test_wrong_score_count_is_rejected(manifest) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="4 classes"):
        decide(np.array([0.5, 0.5]), manifest)


def test_top_k_cannot_exceed_class_count(manifest) -> None:  # type: ignore[no-untyped-def]
    decision = decide(np.array([0.4, 0.3, 0.2, 0.1]), manifest, top_k=50)

    assert len(decision.top_k) == 4


def test_ties_are_resolved_stably(manifest) -> None:  # type: ignore[no-untyped-def]
    """À probabilités égales, l'ordre du manifeste tranche — jamais le hasard."""
    first = decide(np.array([0.25, 0.25, 0.25, 0.25]), manifest)
    second = decide(np.array([0.25, 0.25, 0.25, 0.25]), manifest)

    assert first.best.index == second.best.index == 0
