import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self._conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size
        )
        self._relu = nn.ReLU()
        self._pooling = nn.MaxPool2d(kernel_size=kernel_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        conv: torch.Tensor = self._conv(input)
        rectified: torch.Tensor = self._relu(conv)
        activated: torch.Tensor = self._pooling(rectified)
        return activated
