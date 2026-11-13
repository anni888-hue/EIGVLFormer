import math

import torch
import torch.nn as nn   
import torch.nn.functional as F
    

class ClassEmbed(nn.Module): 
    def __init__(self, init_bias: float = 100.0, init_scale: float = 15.0):     
        super().__init__()     
        self.lang_bias = nn.Parameter(torch.full((), -math.log(init_bias)))  
        self.lang_scale = nn.Parameter(torch.tensor(init_scale).log())  

    def forward(self, image_embeds: torch.Tensor, lang_embeds: torch.Tensor, mask=None) -> torch.Tensor:
        # 归一化对尺度不变：先乘 2^-6 缩小再计算，避免 fp16 下平方溢出
        # （text_adapter 输出的 lang 特征量级可达数万，平方和远超 fp16 上限 65504，
        # TensorRT FP16 构建会使 score 崩塌）；同时以 fp32 计算保证数值稳定。   
        # 注意缩放必须保证缩放后的值域远离 fp16 次正规区（<6e-5），下游由
        # TextAdapter 将特征峰值规范到 ~100，此处 2^-6 后为 ~1.5，两边均安全
        out_dtype = image_embeds.dtype     
        image_norm = F.normalize(image_embeds.float() * 0.015625, p=2, dim=-1)   
        lang_norm = F.normalize(lang_embeds.float() * 0.015625, p=2, dim=-1)  
        logits = image_norm @ lang_norm.transpose(2, 1)
        logits = logits * torch.exp(self.lang_scale) + self.lang_bias    
        if mask is not None:
            logits = logits.masked_fill(~mask.unsqueeze(1), float("-inf"))   
        return logits.to(out_dtype)
