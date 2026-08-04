import torch

from pathlib import Path

from custom.config import HyperParameters, InferenceParameters
from custom.inference.predictor import Prediction, Predictor
from custom.model.factory import build_model
from custom.model.model import Model


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    params: HyperParameters = HyperParameters()
    inference: InferenceParameters = InferenceParameters()

    weights: Path | None = inference.resolve_weights()
    if weights is None or not weights.exists():
        print(
            f"Poids introuvables dans {params.project_dir}/ "
            "(lancer custom/train.py d'abord)"
        )
        return 1

    print(f"Poids: {weights}")

    model: Model = build_model(params)
    model.load_state_dict(torch.load(weights, map_location=device))

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
