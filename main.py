import torch
import torch.nn as nn
import cv2 as cv

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class HyperParameters:
    conv_kernel_size: int = 3
    stride_kernel_size: int = 2

    conv_in_channels: int = 3
    conv_out_channels: List[int] = [32, 64, 128]

    output_classes = ["none", "electrod"]


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


class Model(nn.Module):
    def __init__(
        self,
        input_channels: int,
        out_channels: List[int],
        conv_kernel_size: int,
        output_classes: List[str],
        input_shape: Tuple[int, int],
        stride_kernel_size: int = HyperParameters.stride_kernel_size,
    ):
        super().__init__()
        self._nb_classes = len(output_classes)
        self._kernel_size = conv_kernel_size

        in_channels: List[int] = [input_channels] + out_channels[:-1]
        fully_connected_size: int = out_channels[-1] * input_shape[0] * input_shape[1]

        self._convLayers: nn.ModuleList = nn.ModuleList(
            [
                ConvLayer(
                    in_channels=in_channels[index],
                    out_channels=out_channels[index],
                    kernel_size=conv_kernel_size,
                )
                for index in range(len(out_channels))
            ]
        )

        self._flatten = nn.Flatten()
        self._fully_connected = nn.Linear(
            in_features=fully_connected_size, out_features=self._nb_classes
        )
        self._softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        current: torch.Tensor = input

        for layer in self._convLayers:
            current = layer(current)

        flatten: torch.Tensor = self._flatten(current)
        logits: torch.Tensor = self._fully_connected(flatten)
        result: torch.Tensor = self._softmax(logits)

        return result


def main():
    img_size = (800, 600)
    model: Model = Model(
        input_channels=HyperParameters.conv_in_channels,
        out_channels=HyperParameters.conv_out_channels,
        conv_kernel_size=HyperParameters.conv_kernel_size,
        output_classes=HyperParameters.output_classes,
        input_shape=img_size,
        stride_kernel_size=HyperParameters.stride_kernel_size,
    )


if __name__ == "__main__":
    exit(main())
