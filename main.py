import torch

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from config import HyperParameters
from model.factory import build_model
from model.model import Model
from training.train import Trainer


def main():
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    params: HyperParameters = HyperParameters()

    train_transform: transforms.Compose = transforms.Compose(
        [transforms.Resize(params.img_size), transforms.ToTensor()]
    )

    val_transform: transforms.Compose = transforms.Compose(
        [transforms.Resize(params.img_size), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(root=params.train_dir, transform=train_transform)
    val_dataset = ImageFolder(root=params.val_dir, transform=val_transform)

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

    model: Model = build_model(params)
    model.to(device)

    trainer: Trainer = Trainer(
        model=model, learning_rate=params.learning_rate, device=device
    )

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        nb_epochs=params.epochs,
    )

    trainer.save(params.checkpoint)
    print(f"Poids sauvegardes: {params.checkpoint}")


if __name__ == "__main__":
    exit(main())
