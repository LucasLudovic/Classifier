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
        stride_kernel_size: int = 2,
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

        return logits
