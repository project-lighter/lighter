import monai
from torch import nn
from monai.networks.nets.resnet import ResNetBottleneck as Bottleneck
import torch
from loguru import logger

class Resnet3D(nn.Module):
    def __init__(self, num_classes, pretrained=None) -> None:
        super().__init__()
        self.model = monai.networks.nets.resnet.ResNet(
            block=Bottleneck,
            layers=(3, 4, 6, 3),
            block_inplanes=(64, 128, 256, 512),
            spatial_dims=3,
            n_input_channels=1,
            conv1_t_stride=2,
            conv1_t_size=7,
            widen_factor=2,
            num_classes=num_classes
        )

        if pretrained is not None:
            self.load(pretrained)

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def load(self, pretrained):
        pretrained_model = torch.load(pretrained)
        trained_trunk = pretrained_model['trunk_state_dict']
        msg = self.model.load_state_dict(trained_trunk, strict=False)
        logger.info(f'Loaded pretrained model weights \n' 
                    f'Missing keys: {msg[0]} and unexpected keys: {msg[1]}')