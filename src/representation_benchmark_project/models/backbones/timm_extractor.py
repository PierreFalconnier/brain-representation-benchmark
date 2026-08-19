import timm
from torch import nn

from representation_benchmark_project.models.transforms.all_transforms import (
    replace_resize_crop,
)


class TimmExtractor(nn.Module):
    def __init__(
        self,
        model_name,
        pretrained=True,
        num_classes=0,
        features_only=False,
        use_transforms=True,
    ):
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.features_only = features_only
        self.use_transforms = use_transforms
        self.input_dim = 2

        self.backbone = timm.create_model(
            self.model_name,
            pretrained=self.pretrained,
            num_classes=self.num_classes,
            features_only=self.features_only,
        )

        # transforms
        data_config = timm.data.resolve_model_data_config(self.backbone)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)
        # remove resize+center crop, replace by rize of size of centercrop
        self.transforms = replace_resize_crop(self.transforms)

    def forward(self, x):
        if self.use_transforms:
            x = self.transforms(x)
        if self.features_only:
            return self.backbone(x)[-1]  # return last feature map
        return self.backbone(x)
