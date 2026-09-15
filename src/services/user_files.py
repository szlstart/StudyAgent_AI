"""Safe resolution helpers for paths persisted in per-user records."""

from __future__ import annotations

from pathlib import Path


def resolve_user_file(path_value: str | Path, user_root: Path) -> Path | None:
    """Return a file only when it is contained by the current user's data root."""
    if not path_value:
        return None
    try:
        root = user_root.resolve()
        candidate = Path(path_value).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            return None
        return candidate
    except (OSError, RuntimeError, ValueError):
        return None

