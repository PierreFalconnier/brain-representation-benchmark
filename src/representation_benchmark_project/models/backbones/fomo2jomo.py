from pathlib import Path

import fomo_code  # type: ignore # bootstraps sys.path via fomo_code/__init__.py  # noqa: F401
import torch
import torch.nn.functional as F
from models.networks.mmunetvae import mmunetvae  # type: ignore
from torch import nn
from utils.load_weights import load_pretrained_checkpoint  # type: ignore


class JomoBackbone(nn.Module):
    """
    Feature-extraction wrapper around FOMO25's MultiModalUNetVAE encoder.

    loads pretrained weights, runs only the encoder +
    bottleneck (mu) projections, and returns a flattened global feature vector
    via adaptive average pooling.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        use_vae: bool = True,
    ) -> None:
        super().__init__()

        self.input_dim = 1  # single modality input, like BrainAgeNeXt
        self.use_vae = use_vae
        self.input_dim = 3

        # Build the model in "classification" mode just to instantiate the
        # encoder + latent projection heads; decoder_task is unused.
        self.model = mmunetvae(
            input_channels=1,
            output_channels=1,
            mode="classification",
            use_vae=use_vae,
            use_skip_connections=False,
        )

        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            if checkpoint_path.is_file():
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
            else:
                # Fall back to the default FOMO25 location/env var lookup
                checkpoint = load_pretrained_checkpoint(checkpoint_path.name)
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            # Use FOMO25's default checkpoint discovery
            checkpoint = load_pretrained_checkpoint()
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)

        self.encoder = self.model.encoder
        self.conv_mu_shared = self.model.conv_mu_shared
        self.conv_mu_modality = self.model.conv_mu_modality

        self.pooling = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))

    @torch._dynamo.disable
    def _run_encoder(self, x):
        return self.encoder(x)

    def forward(self, x):
        skips = self._run_encoder(x)

        # Multi-scale pooled features -> bottleneck latent (mu only, deterministic)
        features = [
            F.adaptive_avg_pool3d(f, output_size=skips[-1].shape[2:]) for f in skips
        ]
        pooled = torch.cat(features, dim=1)

        mu_s = self.conv_mu_shared(pooled)
        mu_m = self.conv_mu_modality(pooled)

        z = torch.cat([mu_s, mu_m], dim=1)  # [B, latent_channels, d, h, w]

        return self.pooling(z).flatten(start_dim=1)


if __name__ == "__main__":
    from pathlib import Path

    checkpoint_path = (
        Path(__file__).resolve().parents[4]
        / "submodules/fomo25_mmunetvae_pretrained.ckpt"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = JomoBackbone(
        checkpoint_path=checkpoint_path,
        use_vae=True,
    ).to(device)
    model.eval()

    batch_size = 4
    x = torch.randn(batch_size, 1, 160, 192, 160, device=device)

    with torch.no_grad():
        out = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
    print("Output dtype:", out.dtype)
    print("Output sample (first 10 values of first item):", out[0, :10])
