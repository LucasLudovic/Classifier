import torch

from typing import Union

# Index de GPU ou nom de peripherique, au format attendu par Ultralytics.
Device = Union[int, str]


def resolve_device() -> Device:
    return 0 if torch.cuda.is_available() else "cpu"
