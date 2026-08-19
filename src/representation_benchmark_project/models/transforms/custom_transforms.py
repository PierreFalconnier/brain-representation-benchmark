import math

import torch
import torchvision
from monai.data import MetaTensor
from monai.transforms import CenterSpatialCrop, Resize, ResizeWithPadOrCrop
from monai.transforms.transform import Transform
from torch import nn

# import torch.nn.functional as F
from torchvision.transforms import functional as F


class BatchCenterSpatialCrop:
    def __init__(self, roi_size, lazy=False):
        """
        Adapts MONAI's CenterSpatialCrop to work on a batch of 3D volumes.
        """
        self.cropper = CenterSpatialCrop(roi_size, lazy=lazy)
        self.lazy = lazy

    def __call__(self, batch):
        assert batch.dim() == 5, "Input batch must be a 5D tensor (N, C, H, W, D)"
        cropped_batch = []
        for volume in batch:
            cropped = self.cropper(volume)
            cropped_batch.append(cropped)

        return torch.stack(cropped_batch, dim=0)


class BatchResizeWithPadOrCrop:
    def __init__(self, roi_size, lazy=False):
        """
        Adapts MONAI's ResizeWithPadOrCrop to work on a batch of 3D volumes.
        """
        self.padder_or_cropper = ResizeWithPadOrCrop(roi_size, lazy=lazy)
        self.lazy = lazy

    def __call__(self, batch):
        assert batch.dim() == 5, "Input batch must be a 5D tensor (N, C, H, W, D)"
        cropped_batch = []
        for volume in batch:
            cropped = self.padder_or_cropper(volume)
            cropped_batch.append(cropped)

        return torch.stack(cropped_batch, dim=0)


#  the longest edge matches the given size while preserving aspect ratio.
class ResizeLongestEdge:
    def __init__(
        self, size, interpolation=torchvision.transforms.InterpolationMode.BICUBIC
    ):
        self.size = size
        self.interpolation = interpolation
        self.t = Resize(spatial_size=size, mode="trilinear", anti_aliasing=True)

    def __call__(self, img):
        h, w = img.shape[-2:]
        scale_factor = self.size / max(h, w)
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)

        return F.resize(img, size=(new_h, new_w), interpolation=self.interpolation)


class ResizeLongestEdge3D:
    def __init__(self, size):
        """
        Resize a batch of 3D volumes so that their longest spatial dimension matches `size`,
        while preserving aspect ratio using trilinear interpolation.
        """
        self.size = size

    def __call__(self, img):
        assert img.ndim == 5, "Input must be (N, C, D, H, W)"

        N, C, D, H, W = img.shape
        longest_dim = max(D, H, W)
        scale_factor = self.size / longest_dim  #  scale factor

        new_D = int(D * scale_factor)
        new_H = int(H * scale_factor)
        new_W = int(W * scale_factor)

        img_resized = torch.nn.Upsample(size=(new_D, new_H, new_W), mode="trilinear")(
            img
        )
        return img_resized


# pad 2D image to get a square
class PadToSquare(object):
    def __init__(self) -> None:
        pass

    def __call__(self, batch):
        H, W = batch.shape[-2], batch.shape[-1]
        max_side = max(H, W)

        # Calculate padding

        pad_h = max_side - H
        pad_w = max_side - W

        # padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        # /!\ for some reason , the previous commented order (left, right, top, bottom) leada to wrong padding
        # even though it follows the instructions from F.pad

        # the padding below works as expected : left, top, righ, bottom
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)

        # Apply padding
        padded_batch = F.pad(batch, padding, padding_mode="constant", fill=0)

        return padded_batch


# standarsize a 3D image with its own mean and std
class IndividualStandardization:
    def __init__(self, positive_only: bool = False) -> None:
        self.positive_only = positive_only

    def __call__(self, batch):
        if self.positive_only:
            mask = batch > 0
            masked_batch = torch.where(
                mask, batch, torch.tensor(0.0, device=batch.device)
            )

            sum_positive = torch.sum(masked_batch, dim=(-1, -2, -3), keepdim=True)
            count_positive = torch.sum(mask, dim=(-1, -2, -3), keepdim=True).clamp(
                min=1
            )
            mean = sum_positive / count_positive

            squared_diff = torch.where(
                mask, (batch - mean) ** 2, torch.tensor(0.0, device=batch.device)
            )
            std = torch.sqrt(
                torch.sum(squared_diff, dim=(-1, -2, -3), keepdim=True) / count_positive
                + 1e-8
            )
        else:
            mean = torch.mean(batch, dim=(-1, -2, -3), keepdim=True)
            std = torch.std(batch, dim=(-1, -2, -3), keepdim=True) + 1e-8
        return (batch - mean) / std


class Resize3DBatch(object):
    def __init__(self, size) -> None:
        if isinstance(size, int):
            size = (size,) * 3
        elif isinstance(size, tuple):
            if len(size) != 3:
                raise ValueError(f"Expected a tuple of length 3, but got {size}")
        else:
            raise TypeError(
                f"size must be an int or a tuple of length 3, but got {type(size)}"
            )

        self.t = Resize(spatial_size=size, mode="trilinear", anti_aliasing=True)

    def __call__(self, batch):
        return torch.stack([self.t(image) for image in batch])


class RepeatChannelTransform(object):
    def __init__(self, batch_dim_exist=False):
        self.batch_dim_exist = 1 * batch_dim_exist

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim - self.batch_dim_exist < 3:
            raise ValueError("Input tensor must have at least 3 dimensions")
        if x.size(0 + self.batch_dim_exist) == 3:
            return x
        if x.size(0 + self.batch_dim_exist) != 1:
            raise ValueError("Input tensor's channel dimension (C) must be 1.")

        # repeat the channel dim 3 times
        n_spatial_dim = x.ndim - 1 - self.batch_dim_exist
        dim_list = [3] + [1] * n_spatial_dim
        if self.batch_dim_exist:
            dim_list = [1] + dim_list

        return x.repeat(*dim_list)


class SqueezeTransform(Transform):
    def __init__(self, dim=0):
        self.dim = dim

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.squeeze(x, dim=self.dim)
        return x


class SqueezeTransformd(Transform):
    def __init__(self, keys, dim=0):
        self.dim = dim
        self.keys = [keys] if isinstance(keys, str) else keys

    def __call__(self, sample: dict) -> dict:
        for key in self.keys:
            sample[key] = torch.squeeze(sample[key], dim=self.dim)

        return sample


class MetaTensorToTorchTensor(Transform):
    def __init__(self):
        pass

    def __call__(self, x: MetaTensor) -> torch.Tensor:
        return MetaTensor.ensure_torch_and_prune_meta(im=x, meta=None)


class MetaTensorToTorchTensord(Transform):
    def __init__(self, keys):
        self.keys = [keys] if isinstance(keys, str) else keys

    def __call__(self, sample: dict) -> dict:
        for key in self.keys:
            sample[key] = MetaTensor.ensure_torch_and_prune_meta(
                im=sample[key], meta=None
            )
        return sample


class ResizeToDivisible:
    def __init__(
        self,
        divisible_by: int = 32,
        batch_dim_exists: bool = True,
        mode: str = "trilinear",
        align_corners: bool = False,
    ):
        """
        Args:
            divisible_by: the divisor for H, W, D (default 32).
            batch_dim_exists: if False, input shape is (C, H, W, D); otherwise (B, C, H, W, D).
            mode: interpolation mode passed to torch.nn.functional.interpolate (e.g. "trilinear").
            align_corners: align_corners forwarded for relevant modes.
        """
        self.divisible_by = int(divisible_by)
        self.batch_dim_exists = bool(batch_dim_exists)
        self.mode = mode
        self.align_corners = align_corners

    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Resize the spatial dims H, W, D up to the nearest multiple of divisible_by.
        Accepts tensors of shape (B, C, H, W, D) or (C, H, W, D) when batch_dim_exists=False.
        Returns a tensor with the same number of leading dims as the input.
        """
        if not isinstance(volume, torch.Tensor):
            raise TypeError("volume must be a torch.Tensor")

        # normalize to batched format (B, C, H, W, D)
        if self.batch_dim_exists:
            if volume.dim() != 5:
                raise ValueError(
                    "Expected input shape (B, C, H, W, D) when batch_dim_exists=True"
                )
            batched = volume
            squeeze_batch = False
        else:
            if volume.dim() != 4:
                raise ValueError(
                    "Expected input shape (C, H, W, D) when batch_dim_exists=False"
                )
            batched = volume.unsqueeze(0)  # (1, C, H, W, D)
            squeeze_batch = True

        B, C, H, W, D = batched.shape

        def _ceil_to_div(x):
            return int(math.ceil(x / self.divisible_by) * self.divisible_by)

        new_H, new_W, new_D = _ceil_to_div(H), _ceil_to_div(W), _ceil_to_div(D)

        # if already the correct size, return original (or squeezed) tensor
        if (new_H, new_W, new_D) == (H, W, D):
            return volume

        orig_dtype = batched.dtype
        need_cast_back = not torch.is_floating_point(batched)
        if need_cast_back:
            batched = batched.float()

        # permute to (B, C, D, H, W) for interpolate
        x = batched.permute(0, 1, 4, 2, 3)  # (B, C, D, H, W)
        size = (new_D, new_H, new_W)
        if self.mode in ("trilinear", "bilinear"):
            x = nn.functional.interpolate(
                x, size=size, mode=self.mode, align_corners=self.align_corners
            )
        else:
            x = nn.functional.interpolate(x, size=size, mode=self.mode)

        # permute back to (B, C, H, W, D)
        x = x.permute(0, 1, 3, 4, 2)

        if need_cast_back:
            x = x.to(orig_dtype)

        if squeeze_batch:
            x = x.squeeze(0)  # return (C, H, W, D)

        return x
