import torch
import torch.nn as nn

from typing import List


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_kernel_size: int,
        pool_kernel_size: int,
        stride: int,
    ):
        super().__init__()

        self._conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=conv_kernel_size,
            stride=stride,
        )
        self._relu = nn.ReLU()
        self._pooling = nn.MaxPool2d(kernel_size=pool_kernel_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        conv: torch.Tensor = self._conv(input)
        rectified: torch.Tensor = self._relu(conv)
        activated: torch.Tensor = self._pooling(rectified)
        return activated
