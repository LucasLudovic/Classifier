from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class HyperParameters:
    epochs: int = 50
    batch_size: int = 4
    learning_rate: float = 1e-3
    conv_kernel_size: int = 3
    stride_kernel_size: int = 2

    img_size: Tuple[int, int] = (800, 800)

    conv_in_channels: int = 3
    conv_out_channels: List[int] = field(default_factory=lambda: [32, 64, 128])

    output_classes: List[str] = field(default_factory=lambda: ["none", "electrode"])

    train_dir: Path = Path("data/train")
    val_dir: Path = Path("data/val")

    checkpoint: Path = Path("runs/classify/model.pt")


@dataclass
class InferenceParameters:
    weights: Path = HyperParameters().checkpoint
    source: Path = Path("data/test/none")
