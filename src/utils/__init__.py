from pathlib import Path


def get_project_root() -> Path:
    current_dir = Path(__file__).resolve().parent

    for parent in current_dir.parents:
        if (parent / "pyproject.toml").exists():
            return parent

    raise RuntimeError("Project root not found.")


def get_data_dir() -> Path:
    data_dir = get_project_root() / "data"

    if data_dir.exists():
        return data_dir

    raise RuntimeError("Data dir not found.")


def get_artifacts_dir() -> Path:
    artifacts_dir = get_project_root() / "artifacts"

    if artifacts_dir.exists():
        return artifacts_dir

    raise RuntimeError("Artifacts dir not found.")
