"""Gestion des dossiers de runs: un entrainement = un dossier.

Les runs sont numerotes comme chez Ultralytics: `train`, `train2`, `train3`...
Aucun run existant n'est ecrase, et l'inference peut retrouver le dernier.
"""

from pathlib import Path

WEIGHTS_DIR: str = "weights"
BEST_WEIGHTS: str = "best.pt"


def next_run_dir(project_dir: Path, name: str) -> Path:
    """Retourne un dossier de run encore inutilise (sans le creer).

    L'index suit toujours le plus grand deja present: un run supprime ne
    libere pas son numero, donc deux runs ne peuvent pas partager un dossier.
    """
    latest: Path | None = latest_run_dir(project_dir, name)
    index: int | None = None if latest is None else _run_index(latest, name)
    if index is None:
        return project_dir / name

    return project_dir / f"{name}{index + 1}"


def latest_run_dir(project_dir: Path, name: str) -> Path | None:
    """Retourne le dossier du run le plus recent, ou None si aucun."""
    runs: list[tuple[int, Path]] = [
        (index, directory)
        for directory in project_dir.glob(f"{name}*")
        if directory.is_dir() and (index := _run_index(directory, name)) is not None
    ]

    if not runs:
        return None

    return max(runs)[1]


def best_weights(run_dir: Path) -> Path:
    """Chemin des meilleurs poids a l'interieur d'un run."""
    return run_dir / WEIGHTS_DIR / BEST_WEIGHTS


def latest_weights(project_dir: Path, name: str) -> Path | None:
    """Meilleurs poids du dernier run, ou None si introuvables."""
    run_dir: Path | None = latest_run_dir(project_dir, name)
    if run_dir is None:
        return None

    weights: Path = best_weights(run_dir)

    return weights if weights.exists() else None


def _run_index(directory: Path, name: str) -> int | None:
    """1 pour `name`, n pour `name<n>`, None si le dossier n'est pas un run."""
    suffix: str = directory.name[len(name) :]
    if not suffix:
        return 1

    return int(suffix) if suffix.isdigit() else None
