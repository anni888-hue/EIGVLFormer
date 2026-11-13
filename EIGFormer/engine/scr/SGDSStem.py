import torch
import torch.nn as nn
import torch.nn.functional as F

class SGDSStem(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, use_lab=False):
        super().__init__()
        self.use_lab = use_lab

        self.detail_conv = ConvBNAct(
            in_chs, mid_chs,
            kernel_size=3, stride=1, padding=1,
            use_lab=use_lab
        )
        self.semantic_conv = ConvBNAct(
            in_chs, mid_chs,
            kernel_size=5, stride=2, padding=2,
            use_lab=use_lab
        )
        self.detail_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.stem3 = ConvBNAct(
            mid_chs, mid_chs,
            kernel_size=3, stride=2, padding=1,
            use_lab=use_lab
        )

        self.stem3_identity = nn.MaxPool2d(kernel_size=2, stride=2)


        self.stem4 = ConvBNAct(
            mid_chs, out_chs,
            kernel_size=1, stride=1,
            use_lab=use_lab
        )

        if mid_chs != out_chs:
            self.stem4_identity = nn.Conv2d(mid_chs, out_chs, kernel_size=1, bias=False)
        else:
            self.stem4_identity = nn.Identity()


        self.edge_extractor = nn.Conv2d(
            in_chs, 1,
            kernel_size=3, stride=1, padding=1,
            bias=True
        )

        self.edge_scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))


        self.fusion_alpha = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        nn.init.normal_(self.edge_extractor.weight, mean=0.0, std=0.01)

    def forward(self, x):

        edge_logits = self.edge_extractor(x)                 # (B,1,H,W)
        edge_down = F.avg_pool2d(edge_logits, 2, stride=2)   # (B,1,H/2,W/2)
        weight = torch.sigmoid(self.edge_scale * edge_down)  # (B,1,H/2,W/2)

        detail = self.detail_conv(x)                         # (B,mid,H,W)
        detail_down = self.detail_pool(detail)               # (B,mid,H/2,W/2)
        semantic = self.semantic_conv(x)                     # (B,mid,H/2,W/2)


        fused = weight * detail_down + (1 - weight) * semantic
        fused = fused * self.fusion_alpha


        out3 = self.stem3(fused)                             # (B, mid, H/4, W/4)
        identity3 = self.stem3_identity(fused)               # (B, mid, H/4, W/4)
        out3 = out3 + identity3


        out4 = self.stem4(out3)                              # (B, out_chs, H/4, W/4)
        identity4 = self.stem4_identity(out3)                # (B, out_chs, H/4, W/4)
        out4 = out4 + identity4

        return out4