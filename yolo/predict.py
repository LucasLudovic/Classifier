from yolo.config import InferenceParameters
from yolo.device import Device, resolve_device
from yolo.model.classifier import Classifier, Prediction


def main() -> int:
    params: InferenceParameters = InferenceParameters()
    device: Device = resolve_device()

    model: Classifier = Classifier(weights=params.weights)

    predictions: list[Prediction] = model.predict(
        source=params.source, img_size=params.img_size, device=device
    )

    for prediction in predictions:
        print(f"{prediction.label} ({prediction.confidence:.4f})")

    return 0


if __name__ == "__main__":
    exit(main())
