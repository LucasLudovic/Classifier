from dataclasses import dataclass, field
from pathlib import Path

from run_directory import latest_weights


@dataclass
class HyperParameters:
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.01
    img_size: int = 400
    seed: int = 42

    # n / s / m / l / x : du plus rapide au plus precis
    pretrained_weights: str = "yolo26n-cls.pt"
    optimizer: str = "SGD"

    dataset_dir: Path = Path("data")
    output_classes: list[str] = field(
        default_factory=lambda: ["none", "electrode", "both"]
    )

    project_dir: Path = Path("runs/classify")
    run_name: str = "electrode-cls"


@dataclass
class InferenceParameters:
    source: Path = Path("serie_x/valide/electrode")
    img_size: int = HyperParameters.img_size
    # None -> les poids du dernier entrainement
    weights: Path | None = None

    def resolve_weights(self) -> Path | None:
        """Poids demandes, sinon meilleurs poids du run le plus recent."""
        if self.weights is not None:
            return self.weights

        params: HyperParameters = HyperParameters()

        return latest_weights(params.project_dir, params.run_name)
