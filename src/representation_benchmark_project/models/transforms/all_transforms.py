import timm
import torch
import torchvision.transforms as T
from monai.transforms import Lambda
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import nn
from torchvision import transforms
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize

from representation_benchmark_project.models.transforms.custom_transforms import (
    BatchCenterSpatialCrop,
    BatchResizeWithPadOrCrop,
    IndividualStandardization,
    PadToSquare,
    RepeatChannelTransform,
    Resize3DBatch,
    ResizeLongestEdge,
    ResizeLongestEdge3D,
    ResizeToDivisible,
)


# This function was necessary because ACSConverter fails to properly convert
# the batchnorm2d of a resnet50_clip to batchnorm3d
def replace_batchnorm2d_with_3d(module):
    """
    Recursively replaces BatchNorm2d layers with BatchNorm3d in a module.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(
                module,
                name,
                nn.BatchNorm3d(
                    num_features=child.num_features,
                    eps=child.eps,
                    momentum=child.momentum,
                    affine=child.affine,
                    track_running_stats=child.track_running_stats,
                ),
            )
        else:
            # recursively check child modules
            replace_batchnorm2d_with_3d(child)

    return module


def get_acs_transforms(device):
    mean = torch.tensor([0.4850, 0.4560, 0.4060]).view(1, 3, 1, 1, 1).to(device)
    std = torch.tensor([0.2290, 0.2240, 0.2250]).view(1, 3, 1, 1, 1).to(device)
    t = [
        Resize3DBatch(size=224),
        RepeatChannelTransform(batch_dim_exist=True),
        Lambda(lambda x: (x - mean) / std),
    ]

    return Compose(t)


def replace_resize_crop(compose):
    """
    Takes a transforms.Compose object, removes existing Resize and CenterCrop transforms,
    and replaces them with a Resize transform using the dimensions from CenterCrop.
    """
    new_transforms = []
    resize_dim = None

    for t in compose.transforms:
        if isinstance(t, timm.data.transforms_factory.MaybeToTensor):
            continue

        if isinstance(t, transforms.CenterCrop):
            resize_dim = t.size
        elif not isinstance(t, (transforms.Resize, transforms.CenterCrop)):
            new_transforms.append(t)

    if resize_dim is not None:
        # add the new resize transform
        new_transforms.insert(
            0,
            transforms.Resize(
                size=resize_dim, interpolation=transforms.InterpolationMode.BICUBIC
            ),
        )

    new_transforms.insert(0, PadToSquare())

    new_transforms.insert(
        len(new_transforms) - 1, RepeatChannelTransform(batch_dim_exist=True)
    )

    return transforms.Compose(new_transforms)


def get_biomedclip_transforms():
    return Compose(
        [
            Resize(size=224, interpolation="bicubic", antialias=True),
            CenterCrop(size=(224, 224)),
            Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )


def get_medsam_transforms():
    sam_transforms = T.Compose(
        [
            ResizeLongestEdge(1024),
            PadToSquare(),
            RepeatChannelTransform(batch_dim_exist=True),
            # IndividualStandadization(),
        ]
    )
    return sam_transforms


def get_sammed2d_transforms():
    t = T.Compose(
        [
            ResizeLongestEdge(256),
            PadToSquare(),
            RepeatChannelTransform(batch_dim_exist=True),
            Normalize(
                mean=[123.675 / 255, 116.28 / 255, 103.53 / 255],
                std=[58.395 / 255, 57.12 / 255, 57.375 / 255],
            ),  # the images from sammed 2d are in range [0,255], ours are in [0,1]
        ]
    )
    return t


def get_radimagenet_transforms():
    # from https://github.com/BMEII-AI/RadImageNet/blob/main/pytorch_example.ipynb
    t = T.Compose(
        [
            PadToSquare(),
            transforms.Resize(224),
            RepeatChannelTransform(batch_dim_exist=True),
            Lambda(lambda x: 2 * x - 1),  # [0,1] --> [-1,1]
        ]
    )
    return t


def get_sammed3d_transforms():
    t = T.Compose(
        [
            ResizeLongestEdge3D(256),
            BatchCenterSpatialCrop(roi_size=(128, 128, 128)),
            IndividualStandardization(positive_only=True),
        ]
    )
    return t


def get_vmamba_transforms():
    t = T.Compose(
        [
            T.Resize((224, 224), T.InterpolationMode.BICUBIC),
            RepeatChannelTransform(batch_dim_exist=True),
            T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )
    return t


def get_swinunetr_transforms():
    t = T.Compose(
        [
            ResizeToDivisible(divisible_by=32, batch_dim_exists=True),
            IndividualStandardization(positive_only=True),
        ]
    )
    return t


def get_brainagenext_transforms():
    t = T.Compose(
        [
            BatchResizeWithPadOrCrop(roi_size=(160, 192, 160)),
            IndividualStandardization(positive_only=False),
        ]
    )
    return t


def get_openmind_transforms():
    t = T.Compose(
        [
            BatchResizeWithPadOrCrop(roi_size=(160, 224, 160)),
            IndividualStandardization(positive_only=False),
        ]
    )
    return t
