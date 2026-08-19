import torch
from monai.networks.nets import ResNetFeatures
from torch import nn

from representation_benchmark_project.models.transforms.custom_transforms import (
    IndividualStandardization,
)


class Resnet3D(nn.Module):
    def __init__(self, use_transforms=True) -> None:
        super().__init__()
        self.input_dim = 3
        self.use_transforms = use_transforms

        self.backbone = ResNetFeatures(
            model_name="resnet50", pretrained=True, spatial_dims=3, in_channels=1
        )
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.indiv_norm = IndividualStandardization()

        # transforms (only individual standardisation)
        self.transforms = self.indiv_norm

    def forward(self, x):
        if self.use_transforms:
            x = self.transforms(x)

        x = self.backbone(x)[-1]  # get last feature maps
        x = self.avgpool(x)
        return torch.flatten(x, start_dim=1)
