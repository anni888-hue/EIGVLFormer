import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_group_norm(
    channels: int,
    max_groups: int = 16,
) -> nn.GroupNorm:
    groups = min(max_groups, channels)

    while channels % groups != 0:
        groups -= 1

    return nn.GroupNorm(groups, channels)


class HCSER(nn.Module):
    def __init__(
        self,
        p2_channels: int = 64,
        p3_channels: int = 256,
        out_channels: int = 256,
        hidden_dim: int = 64,
    ):
        super().__init__()

        if hidden_dim < 8:
            raise ValueError(
                f"hidden_dim should be at least 8, got {hidden_dim}"
            )

        self.p2_channels = p2_channels
        self.p3_channels = p3_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim

        self.p3_proj = nn.Sequential(
            nn.Conv2d(
                p3_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            _make_group_norm(out_channels),
            nn.SiLU(inplace=True),
        )

        edge_input_channels = p2_channels * 8

        self.edge_encoder = nn.Sequential(
            nn.Conv2d(
                edge_input_channels,
                hidden_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            _make_group_norm(hidden_dim),
            nn.SiLU(inplace=True),

            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=hidden_dim,
                bias=False,
            ),
            _make_group_norm(hidden_dim),
            nn.SiLU(inplace=True),
        )

        self.semantic_proj = nn.Sequential(
            nn.Conv2d(
                out_channels,
                hidden_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            _make_group_norm(hidden_dim),
            nn.SiLU(inplace=True),
        )

        squeeze_dim = max(hidden_dim // 4, 8)

        self.channel_router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                hidden_dim,
                squeeze_dim,
                kernel_size=1,
                bias=True,
            ),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                squeeze_dim,
                hidden_dim,
                kernel_size=1,
                bias=True,
            ),
        )

        self.spatial_router = nn.Sequential(
            nn.Conv2d(
                hidden_dim * 2,
                hidden_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            _make_group_norm(hidden_dim),
            nn.SiLU(inplace=True),

            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=hidden_dim,
                bias=False,
            ),
            _make_group_norm(hidden_dim),
            nn.SiLU(inplace=True),

            nn.Conv2d(
                hidden_dim,
                1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )
        self.edge_out = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=hidden_dim,
                bias=False,
            ),
            _make_group_norm(hidden_dim),
            nn.SiLU(inplace=True),

            nn.Conv2d(
                hidden_dim,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
        )

        self.alpha_logit = nn.Parameter(
            torch.tensor(-1.386)
        )

        self._init_router()

    def _init_router(self):
        last_channel = self.channel_router[-1]
        nn.init.zeros_(last_channel.weight)
        nn.init.zeros_(last_channel.bias)

        last_spatial = self.spatial_router[-1]
        nn.init.zeros_(last_spatial.weight)
        nn.init.zeros_(last_spatial.bias)

    @staticmethod
    def _pad_to_even(x: torch.Tensor) -> torch.Tensor:

        pad_h = x.shape[-2] % 2
        pad_w = x.shape[-1] % 2

        if pad_h == 0 and pad_w == 0:
            return x

        return F.pad(
            x,
            pad=(0, pad_w, 0, pad_h),
            mode="replicate",
        )

    @staticmethod
    def _extract_detail(p2: torch.Tensor) -> torch.Tensor:

        local_background = F.avg_pool2d(
            p2,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        signed_detail = p2 - local_background
        magnitude_detail = torch.abs(signed_detail)

        return torch.cat(
            [signed_detail, magnitude_detail],
            dim=1,
        )

    def forward(self, inputs) -> torch.Tensor:

        if not isinstance(inputs, (list, tuple)):
            raise TypeError(
                "CSER expects inputs=[p2, p3]."
            )

        if len(inputs) != 2:
            raise ValueError(
                f"CSER expects 2 inputs, got {len(inputs)}."
            )

        p2, p3 = inputs

        base = self.p3_proj(p3)


        detail = self._extract_detail(p2)
        detail = self._pad_to_even(detail)


        edge = F.pixel_unshuffle(
            detail,
            downscale_factor=2,
        )

        edge = self.edge_encoder(edge)

        if edge.shape[-2:] != base.shape[-2:]:
            edge = F.interpolate(
                edge,
                size=base.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )


        semantic = self.semantic_proj(base)


        channel_gate = self.channel_router(
            semantic
        )

        channel_gate = (
            2.0 * torch.sigmoid(channel_gate)
        )

        routed_edge = (
            edge * channel_gate
        )


        spatial_input = torch.cat(
            [routed_edge, semantic],
            dim=1,
        )

        spatial_gate = torch.sigmoid(
            self.spatial_router(spatial_input)
        )


        edge_delta = self.edge_out(
            routed_edge
        )

        edge_delta = (
            edge_delta * spatial_gate
        )

        alpha = torch.sigmoid(
            self.alpha_logit
        )

        out = (
            base
            + alpha * edge_delta
        )

        return out