"""Shared pipeline utilities."""

from __future__ import annotations

from pathlib import Path

_SENTINEL = ".done"


def is_done(path: Path) -> bool:
    """Return True if the stage that writes to *path* completed successfully."""
    return (path / _SENTINEL).exists()


def mark_done(path: Path) -> None:
    """Write the sentinel file that signals a stage completed successfully."""
    path.mkdir(parents=True, exist_ok=True)
    (path / _SENTINEL).touch()
