import monai
import torch


class EfficientNet(monai.networks.nets.EfficientNetBN):

    def __init__(self, last_activation=None, dropout_rate=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override the original dropout_rate
        self._dropout = torch.nn.Dropout(dropout_rate)
        if last_activation is not None:
            self._swish = last_activation
