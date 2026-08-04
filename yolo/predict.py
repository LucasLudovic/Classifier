from pathlib import Path

from yolo.config import HyperParameters, InferenceParameters
from yolo.device import Device, resolve_device
from yolo.model.classifier import Classifier, Prediction


def main() -> int:
    params: InferenceParameters = InferenceParameters()
    device: Device = resolve_device()

    weights: Path | None = params.resolve_weights()
    if weights is None or not weights.exists():
        project_dir: Path = HyperParameters().project_dir
        print(f"Poids introuvables dans {project_dir}/ (lancer yolo/train.py d'abord)")
        return 1

    print(f"Poids: {weights}")

    model: Classifier = Classifier(weights=weights)

    predictions: list[Prediction] = model.predict(
        source=params.source, img_size=params.img_size, device=device
    )

    for prediction in predictions:
        print(f"{prediction.label} ({prediction.confidence:.4f})")

    return 0


if __name__ == "__main__":
    exit(main())
