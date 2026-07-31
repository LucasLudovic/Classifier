from pathlib import Path

from yolo.config import HyperParameters
from yolo.device import Device
from yolo.model.classifier import Classifier, Metrics


class Trainer:
    """Traduit `HyperParameters` en arguments Ultralytics."""

    def __init__(self, model: Classifier, params: HyperParameters, device: Device):
        self._model: Classifier = model
        self._params: HyperParameters = params
        self._device: Device = device

    def fit(self) -> Path:
        """Entraine sur le dataset de la config, retourne le dossier du run."""
        return self._model.train(
            data=str(self._params.dataset_dir.resolve()),
            epochs=self._params.epochs,
            imgsz=self._params.img_size,
            batch=self._params.batch_size,
            lr0=self._params.learning_rate,
            optimizer=self._params.optimizer,
            device=self._device,
            project=str(self._params.project_dir),
            name=self._params.run_name,
            exist_ok=True,
        )

    def validate(self, split: str = "val") -> Metrics:
        return self._model.validate(split=split, device=self._device)
