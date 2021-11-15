import monai
import torch


class EfficientNet(monai.networks.nets.EfficientNetBN):

    def __init__(self, last_activation=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if last_activation is not None:
            self._swish = last_activation
