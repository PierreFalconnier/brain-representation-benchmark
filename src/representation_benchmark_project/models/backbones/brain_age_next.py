import warnings
from pathlib import Path

import torch
from torch import nn

from representation_benchmark_project.models.transforms.all_transforms import (
    get_brainagenext_transforms,
)

try:
    from nnunet_mednext.network_architecture.mednextv1.create_mednext_encoder_v1 import (
        create_mednext_encoder_v1,  # type: ignore; type: ignore
    )
except ImportError:
    warnings.warn("MedNeXt not found, BrainAgeNeXt backbone will not work")


class BrainAgeNeXt(nn.Module):
    def __init__(self, checkpoint_hf, use_transforms=True) -> None:
        super().__init__()
        self.input_dim = 3
        self.use_transforms = use_transforms

        checkpoint_hf = Path(checkpoint_hf) if checkpoint_hf else None

        self.backbone = create_mednext_encoder_v1(
            num_input_channels=1,
            num_classes=1,
            model_id="B",
            kernel_size=3,
            deep_supervision=True,
        )
        if checkpoint_hf is not None:
            from huggingface_hub import hf_hub_download

            local_checkpoint_hf = hf_hub_download(
                repo_id=str(checkpoint_hf.parents[0]), filename=checkpoint_hf.name
            )
            self.backbone.load_state_dict(
                torch.load(local_checkpoint_hf, map_location=torch.device("cpu")),
                strict=False,
            )

        # ====================

        # add global avg pooling for feature extraction
        self.pooling = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))

        self.transforms = get_brainagenext_transforms()

    @torch._dynamo.disable
    def _run_backbone(self, x):
        return self.backbone.forward(x)

    def forward(self, x):
        if self.use_transforms:
            x = self.transforms(x)

        # x = self.backbone.forward(x)
        x = self._run_backbone(x)
        return self.pooling(x).flatten(start_dim=1)
