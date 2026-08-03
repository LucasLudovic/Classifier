import torch
import torch.nn as nn

from typing import List, Tuple

from model.cnn import ConvLayer


class Model(nn.Module):
    def __init__(
        self,
        input_channels: int,
        out_channels: List[int],
        conv_kernel_size: int,
        output_classes: List[str],
        input_shape: Tuple[int, int],
        pool_kernel_size: int = 2,
        stride_kernel_size: int = 2,
    ):
        super().__init__()
        self._nb_classes = len(output_classes)
        self._kernel_size = conv_kernel_size

        in_channels: List[int] = [input_channels] + out_channels[:-1]

        self._convLayers: nn.ModuleList = nn.ModuleList(
            [
                ConvLayer(
                    in_channels=in_channels[index],
                    out_channels=out_channels[index],
                    conv_kernel_size=conv_kernel_size,
                    pool_kernel_size=pool_kernel_size,
                    stride=stride_kernel_size,
                )
                for index in range(len(out_channels))
            ]
        )

        self._pool = nn.AdaptiveMaxPool2d(output_size=1)
        self._flatten = nn.Flatten()
        self._fully_connected = nn.Linear(
            in_features=out_channels[-1], out_features=self._nb_classes
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        current: torch.Tensor = input

        for layer in self._convLayers:
            current = layer(current)

        pooled: torch.Tensor = self._pool(current)
        flatten: torch.Tensor = self._flatten(pooled)
        logits: torch.Tensor = self._fully_connected(flatten)

        return logits
