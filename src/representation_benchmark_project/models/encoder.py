import torch
from torch import nn


class ImageEncoder(nn.Module):
    def __init__(self, backbone, adapt_method=None, K=None, seed=None, device="cuda"):
        super().__init__()
        self.backbone = backbone
        self.device = device
        self.adapt_method = adapt_method
        self.seed = seed
        torch.manual_seed(self.seed)
        assert (
            self.adapt_method in ["1_axis", "3_axis", "raptor"]
            or backbone.input_dim == 3
        ), "unknown adapt_method provided"

        self.backbone.to(self.device)
        self.backbone.eval()

        if self.adapt_method == "raptor" and backbone.input_dim == 2:
            assert self.backbone.features_only, (
                "raptor method requires features_only=True for the backbone"
            )
            dummy_input = torch.randn((1, 1, 224, 224)).to(self.device)
            dummy_output = self.backbone(dummy_input)
            self.output_shape = dummy_output.shape  # B, C_out, H_out, W_out
            if K is None:
                # compute the int K so that 3*K*spatial_dim**2 is approximately 512
                K = int(512 / (3 * self.output_shape[-1] * self.output_shape[-2]))
                self.K = max(K, 1)
            # create a random matrix of shape (K, output_shape) to project the concatenated features
            self.proj_matrix = torch.randn((self.K, self.output_shape[-3])).to(
                self.device
            )
            print(f"Output shape of the backbone: {self.output_shape}")
            print(f"Projection matrix shape: {self.proj_matrix.shape}")
            print(
                f"Final feature dim for 3D image with raptor: {3 * self.K * self.output_shape[-1] * self.output_shape[-2]}"
            )

    # @torch.no_grad()
    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Automatic Mixed Precision
        with torch.autocast(device_type=self.device, dtype=torch.float16):
            B, C_in, H, W, D = x.shape  # should have no assumption on C_in

            # baskbone is 3D, no need to adapt
            if self.backbone.input_dim == 3:
                return self.backbone(x)

            # along axis D (axial slices)
            output_D = []
            for d in range(D):
                x_slice = x[:, :, :, :, d]  # B, C_in, H, W
                output_D.append(
                    self.backbone(x_slice)
                )  # B, C_out | if raptor, B, C_out, H_out, W_out
            output_D = torch.stack(output_D, dim=1).mean(
                dim=1
            )  # B, C_out | if raptor, B, C_out, H_out, W_out

            if self.adapt_method == "1_axis":
                return output_D

            if self.adapt_method in ["3_axis", "raptor"]:
                # along axis H
                output_H = []
                for h in range(H):
                    x_slice = x[:, :, h, :, :]  # B, C_in, W, D
                    output_H.append(self.backbone(x_slice))
                output_H = torch.stack(output_H, dim=1).mean(dim=1)

                # along axis W
                output_W = []
                for w in range(W):
                    x_slice = x[:, :, :, w, :]  # B, C_in, H, D
                    output_W.append(self.backbone(x_slice))
                output_W = torch.stack(output_W, dim=1).mean(dim=1)

                if self.adapt_method == "3_axis":
                    return torch.cat([output_D, output_H, output_W], dim=-1)

                if self.adapt_method == "raptor":
                    # the mean along each axis as been computed already
                    # stack the three outputs of shape (B, C_out, H_out, W_out)
                    # and obtain a tensor of shape (B, 3, C_out, H_out, W_out)
                    output = torch.stack([output_D, output_H, output_W], dim=1)
                    # multiply the output by the projection matrix of shape (K, C_out)
                    # to obtain a tensor of shape (B,3, K, H_out, W_out)
                    output = torch.einsum("bachw,kc->bakhw", output, self.proj_matrix)
                    # flatten the output to obtain a tensor of shape (B, K*H_out*W_out)
                    output = output.flatten(start_dim=1)
                    return output

    @torch.no_grad()
    def get_output_dim(self):
        x = torch.randn((1, 1, 32, 32, 32)).to(self.device)
        out = self.forward(x)
        return out.shape[-1]
