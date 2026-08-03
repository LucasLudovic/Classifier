import torch
import torch.nn as nn

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from PIL import Image
from torchvision import transforms

from data_transforms import build_eval_transform


IMAGE_SUFFIXES: Tuple[str, ...] = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class Prediction:
    path: Path
    label: str
    confidence: float

    def __str__(self) -> str:
        return f"{self.path.name}: {self.label} ({self.confidence:.2%})"


class Predictor:
    def __init__(
        self,
        model: nn.Module,
        output_classes: List[str],
        img_size: Tuple[int, int],
        device: torch.device,
    ):
        self._model: nn.Module = model.to(device)
        self._model.eval()

        self._device: torch.device = device
        # ImageFolder attribue les indices par ordre alphabetique des dossiers,
        # on trie donc les classes pour retrouver le meme mapping.
        self._classes: List[str] = sorted(output_classes)
        self._transform: transforms.Compose = build_eval_transform(img_size)

    @torch.no_grad()
    def predict(self, source: Path) -> List[Prediction]:
        """Retourne la classe la plus probable de chaque image de `source`."""
        return [self._predict_image(path) for path in self._collect(source)]

    @torch.no_grad()
    def _predict_image(self, path: Path) -> Prediction:
        image: Image.Image = Image.open(path).convert("RGB")
        inputs: torch.Tensor = self._transform(image).unsqueeze(0).to(self._device)

        logits: torch.Tensor = self._model(inputs)
        probabilities: torch.Tensor = logits.softmax(dim=1)[0]
        index: int = int(probabilities.argmax().item())

        return Prediction(
            path=path,
            label=self._classes[index],
            confidence=float(probabilities[index].item()),
        )

    @staticmethod
    def _collect(source: Path) -> List[Path]:
        if source.is_file():
            return [source]

        return sorted(
            path
            for path in source.rglob("*")
            if path.suffix.lower() in IMAGE_SUFFIXES
        )
