from pathlib import Path

from yolo import dataset
from yolo.config import HyperParameters
from yolo.device import Device, resolve_device
from yolo.model.classifier import Classifier, Metrics
from yolo.training.train import Trainer


def main() -> int:
    params: HyperParameters = HyperParameters()
    device: Device = resolve_device()

    for before, after in dataset.normalize_roboflow_export(params.dataset_dir):
        print(f"{before}/ -> {after}/")

    for split, classes in dataset.check(
        params.dataset_dir, params.output_classes
    ).items():
        print(f"{split}: {sum(classes.values())} images {classes}")

    model: Classifier = Classifier(weights=params.pretrained_weights)
    trainer: Trainer = Trainer(model=model, params=params, device=device)

    run_dir: Path = trainer.fit()
    print(f"Poids: {run_dir / 'weights' / 'best.pt'}")

    if (params.dataset_dir / "test").is_dir():
        metrics: Metrics = trainer.validate(split="test")
        print(f"Test --- top1: {metrics.top1:.4f}, top5: {metrics.top5:.4f}")

    return 0


if __name__ == "__main__":
    exit(main())
