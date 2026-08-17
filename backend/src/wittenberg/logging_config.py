"""Journalisation structurée en JSON.

Règle de confidentialité : aucun contenu binaire, aucune image, aucun nom de
fichier fourni par l'utilisateur et aucune métadonnée personnelle ne doit être
journalisé. Seules des mesures et des identifiants techniques le sont.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sys
from typing import Any

# Attributs standard de LogRecord : tout le reste est considéré comme un extra.
_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__.keys()) | {
    "message",
    "asctime",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """Formate chaque enregistrement sur une seule ligne JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = _safe(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def _safe(value: Any) -> Any:
    """Empêche l'écriture accidentelle de binaire dans les journaux."""
    if isinstance(value, bytes | bytearray | memoryview):
        return f"<{len(value)} octets omis>"
    return value


def configure_logging(level: str = "INFO") -> None:
    """Installe le formateur JSON sur la racine et les journaux uvicorn."""
    # Les messages sont en français : la console Windows utilise sinon une page
    # de code qui mutile les accents.
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        with contextlib.suppress(ValueError, OSError):
            reconfigure(encoding="utf-8")

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
