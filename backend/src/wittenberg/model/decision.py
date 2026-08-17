"""Interprétation des sorties du modèle.

Vocabulaire — à respecter dans le code comme dans l'interface :

* ``confidence`` désigne la probabilité attribuée à UNE prédiction. Ce n'est
  ni une précision ni un taux de réussite.
* l'*accuracy*, le *macro-F1* et les métriques par classe qualifient le modèle
  dans son ensemble et se calculent sur un jeu de validation
  (``wittenberg.tools.evaluate``), jamais à partir d'une seule image.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wittenberg.model.manifest import DecisionSection, ModelManifest, OutputActivation

#: Tolérance admise sur la somme des probabilités d'une sortie déjà softmax.
_SUM_TOLERANCE = 1e-3


@dataclass(frozen=True)
class ScoredClass:
    """Une classe candidate et sa probabilité."""

    index: int
    label: str
    frequency_hz: float
    probability: float


@dataclass(frozen=True)
class Decision:
    """Décision complète tirée d'un vecteur de probabilités."""

    best: ScoredClass
    top_k: list[ScoredClass]
    reliable: bool
    #: écart entre la première et la deuxième probabilité
    margin: float
    #: code expliquant pourquoi la prédiction n'est pas jugée fiable
    reason: str | None


def softmax(scores: np.ndarray) -> np.ndarray:
    """Softmax numériquement stable."""
    shifted = scores - np.max(scores)
    exponentials = np.exp(shifted)
    total = float(np.sum(exponentials))
    if total <= 0 or not np.isfinite(total):
        # Ne peut se produire qu'avec des scores dégénérés ; on répartit
        # uniformément plutôt que de propager un NaN.
        return np.full_like(scores, 1.0 / scores.size)
    return exponentials / total


def to_probabilities(scores: np.ndarray, activation: OutputActivation) -> np.ndarray:
    """Convertit les scores bruts du modèle en probabilités."""
    values = np.asarray(scores, dtype=np.float64).ravel()
    if activation is OutputActivation.LOGITS:
        return softmax(values)

    total = float(np.sum(values))
    if abs(total - 1.0) <= _SUM_TOLERANCE and float(np.min(values)) >= -_SUM_TOLERANCE:
        return np.clip(values, 0.0, 1.0)
    # Le manifeste annonce softmax mais la sortie ne somme pas à 1 : on
    # renormalise si c'est possible, sinon on applique un softmax.
    if total > 0 and float(np.min(values)) >= 0.0:
        return values / total
    return softmax(values)


def decide(
    probabilities: np.ndarray,
    manifest: ModelManifest,
    *,
    top_k: int | None = None,
    decision_config: DecisionSection | None = None,
) -> Decision:
    """Classe les probabilités et détermine si la prédiction est exploitable."""
    values = np.asarray(probabilities, dtype=np.float64).ravel()
    if values.size != manifest.num_classes:
        raise ValueError(
            f"le modèle a produit {values.size} scores alors que le manifeste "
            f"déclare {manifest.num_classes} classes"
        )

    config = decision_config or manifest.decision
    limit = max(1, min(top_k or config.top_k, manifest.num_classes))

    order = np.argsort(-values, kind="stable")
    classes = manifest.ordered_classes
    ranked = [
        ScoredClass(
            index=int(i),
            label=classes[int(i)].label,
            frequency_hz=classes[int(i)].frequency_hz,
            probability=float(values[int(i)]),
        )
        for i in order[:limit]
    ]

    best = ranked[0]
    margin = float(values[order[0]] - values[order[1]]) if values.size > 1 else 1.0

    reason: str | None = None
    if best.probability < config.min_confidence:
        reason = "low_confidence"
    elif config.min_margin > 0 and margin < config.min_margin:
        reason = "ambiguous"

    return Decision(
        best=best,
        top_k=ranked,
        reliable=reason is None,
        margin=round(margin, 6),
        reason=reason,
    )
