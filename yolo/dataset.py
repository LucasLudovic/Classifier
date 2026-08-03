from pathlib import Path
from typing import TypeAlias

IMAGE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

# Roboflow exporte le split de validation sous le nom "valid",
# Ultralytics attend "val".
ROBOFLOW_SPLITS: dict[str, str] = {"valid": "val", "validation": "val"}

# {split: {classe: nombre d'images}}
Counts: TypeAlias = dict[str, dict[str, int]]


def normalize_roboflow_export(dataset_dir: Path) -> list[tuple[str, str]]:
    """Renomme les splits d'un export Roboflow vers les noms attendus par YOLO.

    Retourne les renommages effectues, sous forme de couples (avant, apres).
    """
    renamed: list[tuple[str, str]] = []

    for source_name, target_name in ROBOFLOW_SPLITS.items():
        source: Path = dataset_dir / source_name
        target: Path = dataset_dir / target_name

        if source.is_dir() and not target.exists():
            source.rename(target)
            renamed.append((source_name, target_name))

    return renamed


def count_images(dataset_dir: Path) -> Counts:
    """Compte les images par classe, pour chaque split present."""
    return {
        split_dir.name: {
            class_dir.name: sum(
                1
                for file in class_dir.iterdir()
                if file.suffix.lower() in IMAGE_SUFFIXES
            )
            for class_dir in sorted(split_dir.iterdir())
            if class_dir.is_dir()
        }
        for split_dir in sorted(dataset_dir.iterdir())
        if split_dir.is_dir()
    }


def check(dataset_dir: Path, expected_classes: list[str]) -> Counts:
    """Verifie les splits et la coherence des classes, retourne les comptes."""
    for split in ("train", "val"):
        if not (dataset_dir / split).is_dir():
            raise FileNotFoundError(f"Split manquant: {dataset_dir / split}")

    counts: Counts = count_images(dataset_dir)
    expected: list[str] = sorted(expected_classes)

    for split, classes in counts.items():
        found: list[str] = sorted(classes)
        if found != expected:
            raise ValueError(
                f"Classes inattendues dans {split}/: {found} != {expected}"
            )

    return counts
