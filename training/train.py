import torch
import torch.nn as nn

from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, model: nn.Module, learning_rate: float, device: torch.Device):
        self._model: nn.Module = model
        self._device: torch.Device = device
        self._criterion: nn.Module = nn.CrossEntropyLoss()
        self._optimizer: torch.optim.Optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate
        )

    @torch.enable_grad()
    def fit(self, train_loader: DataLoader, val_loader: DataLoader, nb_epochs: int):
        self._model.train()
        for epoch in range(nb_epochs):
            current_batch: int = 0
            running_loss: float = 0.0

            for index, data in enumerate(train_loader):
                inputs, labels = data
                self._optimizer.zero_grad()

                logits: torch.Tensor = self._model(inputs)
                loss: torch.Tensor = self._criterion(logits, labels)

                loss.backward()
                self._optimizer.step()

                current_batch = index + 1
                running_loss += loss.item()

            running_loss = running_loss / current_batch
            val_loss = self._validate(val_loader)
            print(
                f"Epoch {epoch}/{nb_epochs} --- {self._criterion._get_name()}: train: {running_loss}, val: {val_loss}"
            )

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> float:
        current_batch: int = 0
        running_loss: float = 0.0

        for index, data in enumerate(val_loader):
            inputs, labels = data
            inputs = inputs.to(self._device)
            labels = labels.to(self._device)

            logits: torch.Tensor = self._model(inputs)
            loss: torch.Tensor = self._criterion(logits, labels)

            current_batch = index + 1
            running_loss += loss.item()

        return running_loss / current_batch
