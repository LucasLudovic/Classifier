from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from yolo.device import Device


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float


@dataclass(frozen=True)
class Metrics:
    top1: float
    top5: float


class Classifier:
    """Cycle de vie d'un modele de classification YOLO26.

    La tete de classification est redimensionnee automatiquement par
    Ultralytics au nombre de dossiers de classes trouves dans le dataset.
    """

    def __init__(self, weights: str | Path = "yolo26n-cls.pt"):
        self._model: YOLO = YOLO(str(weights))

    @property
    def names(self) -> list[str]:
        return [self._model.names[index] for index in sorted(self._model.names)]

    def train(self, **arguments: Any) -> Path:
        """Entraine le modele et retourne le dossier du run."""
        results = self._model.train(**arguments)
        return Path(results.save_dir)

    def validate(self, **arguments: Any) -> Metrics:
        metrics = self._model.val(**arguments)
        return Metrics(top1=float(metrics.top1), top5=float(metrics.top5))

    def predict(
        self,
        source: str | Path,
        img_size: int,
        device: Device,
    ) -> list[Prediction]:
        """Retourne la classe top-1 de chaque image de `source`."""
        results = self._model.predict(
            source=source, imgsz=img_size, device=device, verbose=False
        )

        return [
            Prediction(
                label=result.names[int(result.probs.top1)],
                confidence=float(result.probs.top1conf),
            )
            for result in results
        ]

    def export(self, export_format: str = "onnx", img_size: int = 640) -> Path:
        return Path(self._model.export(format=export_format, imgsz=img_size))
