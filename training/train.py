import torch
import torch.nn as nn

from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, model: nn.Module, learning_rate: int):
        self._model: nn.Module = model
        self._criterion: nn.Module = nn.CrossEntropyLoss()
        self._optimizer: torch.optim.Optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate
        )

    @torch.enable_grad()
    def fit(self, train_loader: DataLoader, val_loader: DataLoader, nb_epochs: int):
        for epoch in range(nb_epochs):
            running_loss: float = 0.0

            for _, data in enumerate(train_loader):
                inputs, labels = data
                self._optimizer.zero_grad()

                logits: torch.Tensor = self._model(inputs)
                loss: torch.Tensor = self._criterion(logits, labels)

                loss.backward()
                self._optimizer.step()

                running_loss += loss.item()

            val_loss = self._validate(val_loader)
            print(
                f"Epoch {epoch}/{nb_epochs} --- {self._criterion._get_name}: train: {running_loss}, val: {val_loss}"
            )

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> float:
        running_loss: float = 0.0
        for _, data in enumerate(val_loader):
            inputs, labels = data

            logits: torch.Tensor = self._model(inputs)
            loss: torch.Tensor = self._criterion(logits, labels)

            running_loss += loss.item()

        return running_loss
