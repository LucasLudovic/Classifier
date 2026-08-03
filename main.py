import torch

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from config import HyperParameters
from data_transforms import build_eval_transform, build_train_transform
from model.factory import build_model
from model.model import Model
from training.train import Trainer


def main():
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    params: HyperParameters = HyperParameters()

    train_transform: transforms.Compose = build_train_transform(params.img_size)
    val_transform: transforms.Compose = build_eval_transform(params.img_size)

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
        model=model,
        learning_rate=params.learning_rate,
        device=device,
        save_dir=params.checkpoint,
    )

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        nb_epochs=params.epochs,
    )


if __name__ == "__main__":
    exit(main())
