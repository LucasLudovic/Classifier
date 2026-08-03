import torch

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from model.model import Model
from training.train import Trainer


@dataclass
class HyperParameters:
    epochs: int = 50
    batch_size: int = 4
    learning_rate: float = 0.1
    conv_kernel_size: int = 3
    stride_kernel_size: int = 2

    img_size: Tuple[int, int] = (800, 800)

    conv_in_channels: int = 3
    conv_out_channels: List[int] = field(default_factory=lambda: [32, 64, 128])

    output_classes: List[str] = field(default_factory=lambda: ["none", "electrode"])


def main():
    device: torch.Device = "cuda" if torch.cuda.is_available() else "cpu"
    params: HyperParameters = HyperParameters()

    train_dir: Path = Path("data/train")
    val_dir: Path = Path("data/val")

    train_transform: transforms.Compose = transforms.Compose(
        [transforms.Resize(params.img_size), transforms.ToTensor()]
    )

    val_transform: transforms.Compose = transforms.Compose(
        [transforms.Resize(params.img_size), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = ImageFolder(root=val_dir, transform=val_transform)

    train_loader: DataLoader = DataLoader(
        dataset=train_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=8,
    )
    val_loader: DataLoader = DataLoader(
        dataset=val_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=8,
    )

    model: Model = Model(
        input_channels=params.conv_in_channels,
        out_channels=params.conv_out_channels,
        conv_kernel_size=params.conv_kernel_size,
        output_classes=params.output_classes,
        input_shape=params.img_size,
        stride_kernel_size=HyperParameters.stride_kernel_size,
    )
    model.to(device)

    trainer: Trainer = Trainer(
        model=model, learning_rate=params.learning_rate, device=device
    )

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        nb_epochs=params.epochs,
    )


if __name__ == "__main__":
    exit(main())
