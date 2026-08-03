import torch

from config import HyperParameters, InferenceParameters
from inference.predictor import Prediction, Predictor
from model.factory import build_model
from model.model import Model


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    params: HyperParameters = HyperParameters()
    inference: InferenceParameters = InferenceParameters()

    if not inference.weights.exists():
        print(f"Poids introuvables: {inference.weights} (lancer main.py d'abord)")
        return 1

    model: Model = build_model(params)
    model.load_state_dict(torch.load(inference.weights, map_location=device))

    predictor: Predictor = Predictor(
        model=model,
        output_classes=params.output_classes,
        img_size=params.img_size,
        device=device,
    )

    predictions: list[Prediction] = predictor.predict(source=inference.source)

    for prediction in predictions:
        print(prediction)

    return 0


if __name__ == "__main__":
    exit(main())
