from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from run_directory import latest_weights


@dataclass
class HyperParameters:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    conv_kernel_size: int = 3
    stride_kernel_size: int = 2

    img_size: Tuple[int, int] = (400, 400)

    conv_in_channels: int = 3
    conv_out_channels: List[int] = field(default_factory=lambda: [32, 64, 128])

    output_classes: List[str] = field(
        default_factory=lambda: ["none", "electrode", "both"]
    )

    train_dir: Path = Path("data/train")
    val_dir: Path = Path("data/val")

    project_dir: Path = Path("runs/custom")
    run_name: str = "train"


@dataclass
class InferenceParameters:
    source: Path = Path("data/test/both")
    # None -> les poids du dernier entrainement
    weights: Path | None = None

    def resolve_weights(self) -> Path | None:
        """Poids demandes, sinon meilleurs poids du run le plus recent."""
        if self.weights is not None:
            return self.weights

        params: HyperParameters = HyperParameters()

        return latest_weights(params.project_dir, params.run_name)
