"""Shared path rules for scripts that may run outside the project directory."""
from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_GROUPED_DATASET = DEFAULT_DATA_DIR / "face_test_by_person"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "face_eval.csv"


def user_path(value: str | Path) -> Path:
    """Resolve a user argument against the directory where the command was run."""
    return Path(value).expanduser().resolve()


def registry_reference(value: str | Path, registry_path: Path) -> Path:
    """Resolve current absolute and historical relative registry paths.

    New registrations use absolute paths. Older registries commonly stored paths
    such as ``data/people/person_001/embeddings.npy`` relative to the project.
    A portable registry may instead store a path relative to registry.json.
    """
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    project_candidate = (PROJECT_ROOT / path).resolve()
    registry_candidate = (registry_path.parent / path).resolve()
    if project_candidate.exists():
        return project_candidate
    if registry_candidate.exists():
        return registry_candidate
    return project_candidate if path.parts and path.parts[0] == "data" else registry_candidate
