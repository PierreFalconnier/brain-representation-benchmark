import torch
from torch import nn
from torchvision.models import resnet50

from representation_benchmark_project.models.transforms.all_transforms import (
    get_radimagenet_transforms,
)


class RadImagenet(nn.Module):
    def __init__(self, weights_path, use_transforms=True):
        super().__init__()
        self.input_dim = 2
        self.use_transforms = use_transforms
        self.weights_path = weights_path
        # the weights are available on https://github.com/BMEII-AI/RadImageNet
        base_model = resnet50(pretrained=False)
        encoder_layers = list(base_model.children())
        self.backbone = nn.Sequential(*encoder_layers[:9])
        state_dict = torch.load(
            self.weights_path,
            map_location=torch.device("cpu"),
        )
        self.load_state_dict(state_dict)

        self.transforms = get_radimagenet_transforms()

    def forward(self, x):
        if self.use_transforms:
            x = self.transforms(x)

        return self.backbone(x).flatten(start_dim=1)
