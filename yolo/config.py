from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class HyperParameters:
    epochs: int = 50
    batch_size: int = 4
    learning_rate: float = 0.01
    img_size: int = 800

    # n / s / m / l / x : du plus rapide au plus precis
    pretrained_weights: str = "yolo26n-cls.pt"
    optimizer: str = "SGD"

    dataset_dir: Path = Path("data")
    output_classes: list[str] = field(default_factory=lambda: ["none", "electrode"])

    project_dir: Path = Path("runs/classify/runs")
    run_name: str = "electrode-cls"

    @property
    def run_dir(self) -> Path:
        return self.project_dir / self.run_name

    @property
    def best_weights(self) -> Path:
        return self.run_dir / "weights" / "best.pt"


@dataclass
class InferenceParameters:
    weights: Path = HyperParameters().best_weights
    source: Path = Path("data/test/electrode")
    img_size: int = HyperParameters.img_size
