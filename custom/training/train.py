import torch
import torch.nn as nn

from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import DataLoader

from run_directory import best_weights


@dataclass(frozen=True)
class Metrics:
    loss: float
    accuracy: float

    def __str__(self) -> str:
        return f"loss: {self.loss:.4f}, accuracy: {self.accuracy:.2%}"


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float,
        device: torch.device,
        run_dir: Path,
    ):
        self._model: nn.Module = model
        self._device: torch.device = device
        self._criterion: nn.Module = nn.CrossEntropyLoss()
        # self._optimizer: torch.optim.Optimizer = torch.optim.SGD(
        #     model.parameters(), lr=learning_rate, momentum=0.9
        # )
        self._optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            params=model.parameters(), lr=learning_rate
        )
        self.run_dir: Path = run_dir
        self.weights: Path = best_weights(run_dir)

    @torch.enable_grad()
    def fit(self, train_loader: DataLoader, val_loader: DataLoader, nb_epochs: int):
        best_val: float = float("inf")
        for epoch in range(nb_epochs):
            save: bool = False
            current_batch: int = 0
            running_loss: float = 0.0
            correct: int = 0
            total: int = 0

            self._model.train()
            for index, data in enumerate(train_loader):
                inputs, labels = data
                inputs = inputs.to(self._device)
                labels = labels.to(self._device)

                self._optimizer.zero_grad()

                logits: torch.Tensor = self._model(inputs)
                loss: torch.Tensor = self._criterion(logits, labels)

                loss.backward()
                self._optimizer.step()

                current_batch = index + 1
                running_loss += loss.item()
                correct += int((logits.argmax(dim=1) == labels).sum().item())
                total += labels.size(0)

            train_metrics: Metrics = Metrics(
                loss=running_loss / current_batch, accuracy=correct / total
            )
            val_metrics: Metrics = self._validate(val_loader)
            if val_metrics.loss < best_val:
                best_val = val_metrics.loss
                self.save()
                save = True

            print(
                f"Epoch {epoch}/{nb_epochs} --- train: {train_metrics} --- val: {val_metrics} {"<--" if save else ""}"
            )
            save = False

    def save(self) -> None:
        """Sauvegarde les poids du run courant pour l'inference."""
        self.weights.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), self.weights)

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> Metrics:
        current_batch: int = 0
        running_loss: float = 0.0
        correct: int = 0
        total: int = 0

        self._model.eval()
        for index, data in enumerate(val_loader):
            inputs, labels = data
            inputs = inputs.to(self._device)
            labels = labels.to(self._device)

            logits: torch.Tensor = self._model(inputs)
            loss: torch.Tensor = self._criterion(logits, labels)

            current_batch = index + 1
            running_loss += loss.item()
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += labels.size(0)

        return Metrics(loss=running_loss / current_batch, accuracy=correct / total)
