import torch  
import torch.nn as nn
import torch.nn.functional as F     
from torch.nn import init
     
from ..logger_module import get_logger

logger = get_logger(__name__)   
RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m" 


def rescale_proj_out_to_peak(adapter: "TextAdapter", reference: torch.Tensor, peak: float = 100.0) -> None:
    """按 reference 前向峰值把 adapter 输出规范到 ~peak（缩放折进 proj_out 权重）。 

    下游 ClassEmbed 为 L2 归一化点积（尺度不变），对模型输出数学等价；但不同   
    checkpoint 训练后 adapter 输出量级差异极大（0.26 ~ 9e4），9e4 超出 fp16 上限 
    （65504），TensorRT FP16 构建会得到 inf/NaN score。只改权重、不改图结构， 
    避免影响 TRT myelin 融合形态。需由调用方保证不重复缩放。
    """
    if reference is None or reference.numel() == 0:
        return  
    with torch.no_grad():
        out = adapter(reference.detach().float())
        cur_peak = out.abs().amax().clamp_min(1e-12)
        # 仅当峰值处于 fp16 危险区才干预：过大（>1000，逼近 65504 上限）会 inf，
        # 过小（<1，经 ClassEmbed 内 2^-6 后落入次正规区 <6e-5）损失精度；
        # 区间内不缩放，避免无谓改变权重数值导致 TRT tactic/融合选择变化
        if 1.0 <= cur_peak.item() <= 1000.0:     
            return    
        scale = (peak / cur_peak).item()
        if abs(scale - 1.0) < 1e-6:
            return
        adapter.proj_out.weight.mul_(scale)   
        adapter.proj_out.bias.mul_(scale)
    logger.info( 
        ORANGE + f"TextAdapter proj_out rescaled by {scale:.6g} for fp16-safe " 
        f"text feature peak (was {cur_peak.item():.6g})" + RESET  
    ) 
   

class RMSNorm(nn.Module):    
    def __init__(self, dim: int, eps: float = 1e-12):
        super().__init__() 
        self.eps = eps 
        self.scale = nn.Parameter(torch.ones(dim))   
  
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.scale     
  
     
class GatedFFNBlock(nn.Module):     
    def __init__(self, dim: int):
        super().__init__()
        self.w12 = nn.Linear(dim, 4 * dim)    
        self.w3 = nn.Linear(2 * dim, dim)
        self.norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(self.norm(x))
        x1, x2 = x12.chunk(2, dim=-1)    
        return x + self.w3(F.silu(x1) * x2) 
 

class TextAdapter(nn.Module):    
    def __init__(self, text_dim: int, img_dim: int, num_layers: int = 1):
        super().__init__() 
        self.layers = nn.ModuleList([GatedFFNBlock(text_dim) for _ in range(num_layers)])   
        self.proj_out = nn.Linear(text_dim, img_dim)    
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for layer in self.layers:
            init.xavier_uniform_(layer.w12.weight)     
            init.constant_(layer.w12.bias, 0)
            init.xavier_uniform_(layer.w3.weight)
            init.constant_(layer.w3.bias, 0) 
        init.xavier_uniform_(self.proj_out.weight)
        init.constant_(self.proj_out.bias, 0)

    def forward(self, text_feats: torch.Tensor) -> torch.Tensor:  
        x = text_feats 
        for layer in self.layers:   
            x = layer(x)
        return self.proj_out(x)
