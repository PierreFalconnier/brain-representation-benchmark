from open_clip import create_model_from_pretrained
from torch import nn

from representation_benchmark_project.models.transforms.all_transforms import (
    get_biomedclip_transforms,
    replace_resize_crop,
)


class BiomedCLIP(nn.Module):
    def __init__(self, model_name, use_transforms=True) -> None:
        super().__init__()
        self.model_name = model_name
        self.input_dim = 2
        self.use_transforms = use_transforms

        model, self.transforms = create_model_from_pretrained(self.model_name)
        self.backbone = model.visual  # get the vision backbone
        self.backbone = self.backbone

        # the transforms except rgb conversion and ToTensor
        self.transforms = get_biomedclip_transforms()
        # remove crop, add repeated channels
        self.transforms = replace_resize_crop(self.transforms)

    def forward(self, x):
        if self.use_transforms:
            x = self.transforms(x)
        # return self.backbone.forward_features(x)
        return self.backbone.forward(x)
