import torch

from typing import TypeAlias

# Index de GPU ou nom de peripherique, au format attendu par Ultralytics.
Device: TypeAlias = int | str


def resolve_device() -> Device:
    return 0 if torch.cuda.is_available() else "cpu"
