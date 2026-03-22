"""Path management for the project."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
DOWNLOADED_DATA_DIR = DATA_DIR / "downloaded"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXPORTS_DIR = DATA_DIR / "exports"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"


def ensure_directories() -> None:
    """Ensure key project directories exist."""
    for path in (DOWNLOADED_DATA_DIR, PROCESSED_DATA_DIR, EXPORTS_DIR, NOTEBOOKS_DIR):
        path.mkdir(parents=True, exist_ok=True)
