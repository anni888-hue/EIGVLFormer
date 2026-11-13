import cv2
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeTextAdapter(nn.Module):

    def __init__(
        self,
        text_dim=512,
        hidden_dim=256,
        num_tokens=64,
        edge_size=32,
    ):
        super().__init__()

        if edge_size < 2:
            raise ValueError("edge_size must be at least 2.")

        self.num_tokens = num_tokens
        self.edge_size = edge_size

        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

        self.edge_encoder = nn.Sequential(
            nn.Conv2d(
                2, 16,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.GroupNorm(4, 16),
            nn.SiLU(inplace=True),

            nn.Conv2d(
                16, hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.GroupNorm(16, hidden_dim),
            nn.SiLU(inplace=True),
        )

        self.edge_proj = nn.Linear(
            hidden_dim,
            hidden_dim
        )
        self.edge_norm = nn.LayerNorm(hidden_dim)

        self.text_input_norm = nn.LayerNorm(text_dim)
        self.text_q = nn.Linear(text_dim, hidden_dim)

        self.edge_k = nn.Linear(hidden_dim, hidden_dim)
        self.edge_v = nn.Linear(hidden_dim, text_dim)

        initial_alpha = 0.05
        self.alpha_logit = nn.Parameter(
            torch.tensor(
                math.log(
                    initial_alpha /
                    (1.0 - initial_alpha)
                ),
                dtype=torch.float32
            )
        )

        self.out_norm = nn.LayerNorm(text_dim)

    def forward(self, image, text):
        image_bs = image.shape[0]

        if text.dim() == 2:
            text = text.unsqueeze(0)

        if text.shape[0] == 1 and image_bs > 1:
            text = text.expand(image_bs, -1, -1)
        elif text.shape[0] != image_bs:
            raise ValueError(
                f"Text batch {text.shape[0]} does not "
                f"match image batch {image_bs}."
            )

        gray = image.mean(dim=1, keepdim=True)

        sobel_x = self.sobel_x.to(dtype=gray.dtype)
        sobel_y = self.sobel_y.to(dtype=gray.dtype)

        edge_x = F.conv2d(
            gray,
            sobel_x,
            padding=1
        )
        edge_y = F.conv2d(
            gray,
            sobel_y,
            padding=1
        )

        edge = torch.sqrt(
            edge_x.square()
            + edge_y.square()
            + 1e-6
        )

        edge_max = F.adaptive_max_pool2d(
            edge,
            (self.edge_size, self.edge_size)
        )
        edge_avg = F.adaptive_avg_pool2d(
            edge,
            (self.edge_size, self.edge_size)
        )

        edge = torch.cat(
            [edge_max, edge_avg],
            dim=1
        )

        feat = self.edge_encoder(edge)

        batch_size, channels, height, width = feat.shape
        feat_flat = feat.flatten(2)

        scores = torch.norm(
            feat_flat,
            p=2,
            dim=1
        )

        top_k = min(
            self.num_tokens,
            height * width
        )

        top_idx = torch.topk(
            scores,
            k=top_k,
            dim=-1
        ).indices

        tokens = torch.gather(
            feat_flat,
            dim=-1,
            index=top_idx.unsqueeze(1).expand(
                -1,
                channels,
                -1
            )
        ).transpose(1, 2)

        tokens = self.edge_norm(
            self.edge_proj(tokens)
        )

        q = self.text_q(
            self.text_input_norm(text)
        )
        k = self.edge_k(tokens)
        v = self.edge_v(tokens)

        attn = torch.matmul(
            q,
            k.transpose(-1, -2)
        )
        attn = attn / (q.shape[-1] ** 0.5)
        attn = torch.softmax(attn, dim=-1)

        edge_context = torch.matmul(attn, v)

        alpha = torch.sigmoid(
            self.alpha_logit
        )

        out = self.out_norm(
            text + alpha * edge_context
        )

        return out
