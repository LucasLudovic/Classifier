from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from model.model import Model
from training.train import Trainer


@dataclass
class HyperParameters:
    batch_size: int = 4
    conv_kernel_size: int = 3
    stride_kernel_size: int = 2

    conv_in_channels: int = 3
    conv_out_channels: List[int] = field(default_factory=lambda: [32, 64, 128])

    output_classes: List[str] = field(default_factory=lambda: ["none", "electrod"])


def main():
    img_size = (800, 600)

    train_dir: Path = Path("data/train")
    val_dir: Path = Path("data/val")

    train_dataset = ImageFolder(root=train_dir)
    val_dataset = ImageFolder(root=val_dir)

    train_loader: DataLoader = DataLoader(
        dataset=train_dataset,
        batch_size=HyperParameters.batch_size,
        shuffle=True,
        num_workers=8,
    )
    val_loader: DataLoader = DataLoader(
        dataset=val_dataset,
        batch_size=HyperParameters.batch_size,
        shuffle=True,
        num_workers=8,
    )

    model: Model = Model(
        input_channels=HyperParameters.conv_in_channels,
        out_channels=HyperParameters.conv_out_channels,
        conv_kernel_size=HyperParameters.conv_kernel_size,
        output_classes=HyperParameters.output_classes,
        input_shape=img_size,
        stride_kernel_size=HyperParameters.stride_kernel_size,
    )


if __name__ == "__main__":
    exit(main())
