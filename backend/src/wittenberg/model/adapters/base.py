"""Interface commune à tous les runtimes d'inférence."""

from __future__ import annotations

import abc

import numpy as np


class AdapterError(Exception):
    """Le runtime est absent, l'artefact illisible ou l'inférence a échoué."""


class ModelAdapter(abc.ABC):
    """Enveloppe un modèle chargé une seule fois, exécuté sur CPU.

    Les implémentations ne doivent jamais désérialiser un artefact provenant
    d'une requête : seul le fichier de confiance configuré côté serveur est
    chargé.
    """

    #: Faux si le runtime exige une sérialisation des appels.
    thread_safe: bool = False

    @property
    @abc.abstractmethod
    def input_shape(self) -> tuple[int | None, ...]:
        """Forme d'entrée déclarée par l'artefact (``None`` = dimension libre)."""

    @property
    @abc.abstractmethod
    def output_size(self) -> int | None:
        """Nombre de sorties déclaré par l'artefact, si connu."""

    @property
    @abc.abstractmethod
    def runtime_version(self) -> str:
        """Version du runtime, pour la traçabilité."""

    @abc.abstractmethod
    def infer(self, tensor: np.ndarray) -> np.ndarray:
        """Exécute une inférence et renvoie un vecteur 1-D de scores."""

    def close(self) -> None:  # noqa: B027 - crochet facultatif, volontairement vide
        """Libère les ressources. Sans effet par défaut."""


def as_score_vector(raw: np.ndarray) -> np.ndarray:
    """Réduit une sortie brute à un vecteur 1-D de scores en float64.

    Accepte ``(C,)`` et ``(1, C)`` ; refuse tout lot de taille supérieure à 1,
    qui signalerait une incohérence de forme plutôt qu'un résultat exploitable.
    """
    array = np.asarray(raw, dtype=np.float64)
    array = np.squeeze(array)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise AdapterError(
            f"sortie du modèle de forme inattendue {tuple(np.shape(raw))} ; "
            "un vecteur de scores par classe est attendu"
        )
    if not np.all(np.isfinite(array)):
        raise AdapterError("la sortie du modèle contient des valeurs non finies")
    return array
