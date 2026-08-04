import torch

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from custom.config import HyperParameters
from custom.data_transforms import build_eval_transform, build_train_transform
from custom.model.factory import build_model
from custom.model.model import Model
from custom.training.train import Trainer
from run_directory import next_run_dir


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

    run_dir: Path = next_run_dir(params.project_dir, params.run_name)
    print(f"Run: {run_dir}")

    trainer: Trainer = Trainer(
        model=model,
        learning_rate=params.learning_rate,
        device=device,
        run_dir=run_dir,
    )

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        nb_epochs=params.epochs,
    )

    print(f"Poids: {trainer.weights}")


if __name__ == "__main__":
    exit(main())
