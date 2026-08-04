from custom.config import HyperParameters
from custom.model.model import Model


def build_model(params: HyperParameters) -> Model:
    """Construit le reseau a partir des hyperparametres.

    Partage entre l'entrainement et l'inference pour garantir que
    l'architecture chargee correspond a celle des poids sauvegardes.
    """
    return Model(
        input_channels=params.conv_in_channels,
        out_channels=params.conv_out_channels,
        conv_kernel_size=params.conv_kernel_size,
        output_classes=params.output_classes,
        input_shape=params.img_size,
        stride_kernel_size=params.stride_kernel_size,
    )
