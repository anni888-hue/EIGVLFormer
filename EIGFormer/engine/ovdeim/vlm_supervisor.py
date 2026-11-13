""" 
VLM4RSDet 风格的 VLM 辅助监督（仅训练期使用，推理/部署时移除）。  
  
移植自 VLM4RSDet (CVPR 2026) mmdet/models/detectors/two_stage.py： 
检测器多尺度特征 -> 1x1 conv(->1024) -> 插值 32x32 -> Florence-2 projector -> GLCA
-> 与 <OD>/<ROD> prompt 拼接 -> Florence-2 语言模型（BART）以 LHPS 分组后的 GT 序列做 next-token CE。   
"""
import json
import math
from pathlib import Path     
from typing import List, Optional, Sequence   

import torch
import torch.nn as nn  
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint   

from ..logger_module import get_logger
from ._shared import box_cxcywh_to_xyxy, obb_cxcywha_to_polyxy, polygon_area
  
logger = get_logger(__name__)
ORANGE, RESET = "\033[38;5;208m", "\033[0m"

__all__ = ["Florence2Supervisor"]   
 
# Florence2Processor 在 tokenizer 上追加的特殊 token（顺序决定 id，必须与原实现一致）   
FLORENCE2_EXTRA_TOKENS = (  
    ['<od>', '</od>', '<ocr>', '</ocr>']
    + [f'<loc_{x}>' for x in range(1000)]    
    + ['<cap>', '</cap>', '<ncap>', '</ncap>', '<dcap>', '</dcap>', '<grounding>', '</grounding>',
       '<seg>', '</seg>', '<sep>', '<region_cap>', '</region_cap>', '<region_to_desciption>',    
       '</region_to_desciption>', '<proposal>', '</proposal>', '<poly>', '</poly>', '<and>']     
) 

TASK_PROMPTS = {     
    '<OD>': 'Locate the objects with category name in the image.',
    '<ROD>': 'Locate the objects tightly with quad boxes and category names in the image.', 
}  

   
class GlobalLocalCrossAttention(nn.Module):    
    """无参数 GLCA：Softmax(N_i N_g^T / sqrt(d)) N_g + N_i。"""   

    def forward(self, local_feat: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor: 
        d = local_feat.shape[-1]
        attn = torch.matmul(local_feat, global_feat.transpose(-2, -1)) / math.sqrt(d)
        return torch.matmul(F.softmax(attn, dim=-1), global_feat) + local_feat 
 

def lhps_split(objects: list, num_levels: int) -> List[list]:    
    """按面积升序排序后，按 ceil(beta*_i * M) 切分；beta 恒为 1（与源码实际行为一致）。"""     
    groups = [[] for _ in range(num_levels)]
    total = len(objects)   
    if total == 0:
        return groups 
    objects = sorted(objects, key=lambda obj: obj["area"])
    size = math.ceil(total / num_levels)
    start = 0
    for i in range(num_levels):
        if start >= total:   
            break     
        end = min(start + size, total)    
        groups[i] = objects[start:end]
        start = end
    if start < total:    
        groups[-1].extend(objects[start:])  
    return groups


def group_to_target_string(objects: list) -> str:
    """Florence-2 <OD>/<ROD> 目标格式：按类别首次出现顺序分组，同类框以 <sep> 分隔
    （HBB 每个框 4 个 <loc_k>，OBB 每个框 8 个 <loc_k>）。"""   
    if len(objects) == 0:    
        return ""     
    class_order, class_to_coords = [], {}
    for obj in objects:
        name = obj["class_name"]
        if name not in class_to_coords: 
            class_order.append(name)    
            class_to_coords[name] = []
        class_to_coords[name].append(obj["coords"])    
    parts = []
    for name in class_order:  
        parts.append(name)
        parts.append("<sep>".join("".join(f"<loc_{int(v)}>" for v in coords) for coords in class_to_coords[name]))     
    return "".join(parts)    


def build_level_targets(
    targets: Sequence[dict],  
    class_names: Sequence[str],     
    num_levels: int,  
    use_lhps: bool = True,
    num_loc_bins: int = 1000, 
    max_objs_per_level: Optional[int] = None,     
) -> List[List[str]]:
    """targets 中 boxes 为归一化 cxcywh；OBB（cxcywha，angle 已除 pi）按 VLM4RSDet <ROD> 约定编码为
    四顶点 8 个 <loc_k> token。类名使用固定 class_names[raw_labels]（无 raw_labels 时用 labels）。

    floor(x_norm * bins) 与源码 floor(x_pix * bins / W) 等价；同图内面积排序与像素面积排序等价。     
    """
    level_targets = [[] for _ in range(num_levels)]
    for target in targets:
        labels = target.get("raw_labels", target["labels"])  
        boxes = target["boxes"].float()    
        if boxes.shape[-1] == 5:
            if "size" not in target:
                raise KeyError("OBB VLM supervision requires target['size'] (resized [W, H]).")   
            width, height = target["size"][0].item(), target["size"][1].item()     
            polys = obb_cxcywha_to_polyxy(boxes, target["size"]) 
            scale = boxes.new_tensor([num_loc_bins / width, num_loc_bins / height]).repeat(4)
            quantized = torch.floor(polys * scale).clamp(0, num_loc_bins - 1).long().tolist() 
            areas = polygon_area(polys).tolist()  
        else:
            boxes = box_cxcywh_to_xyxy(boxes) 
            quantized = torch.floor(boxes * num_loc_bins).clamp(0, num_loc_bins - 1).long().tolist() 
            areas = ((boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)).tolist() 
        objects = []
        for label, coords, area in zip(labels.tolist(), quantized, areas):
            if not 0 <= int(label) < len(class_names):  
                raise IndexError(f"Invalid class index for VLM supervision: {label}")
            objects.append({"class_name": class_names[int(label)], "coords": coords, "area": area})

        # use_lhps=False 时每一层都预测全部目标（Florence-2 原始单层预测方式）
        groups = lhps_split(objects, num_levels) if use_lhps else [list(objects) for _ in range(num_levels)] 
        for level, group in enumerate(groups):    
            if max_objs_per_level is not None and len(group) > max_objs_per_level:  
                keep = sorted(torch.randperm(len(group))[:max_objs_per_level].tolist())   
                group = [group[i] for i in keep]  
            level_targets[level].append(group_to_target_string(group)) 
    return level_targets
    

def _resolve_class_names(class_names) -> List[str]:
    """支持三种形式：类别名列表、COCO 标注 json 路径（按 categories 顺序）、  
    ultralytics 风格 data yaml 路径（取 names 字段，与 OBB 数据集 _load_names 约定一致，
    mapping 按索引升序排列）。"""     
    if isinstance(class_names, (list, tuple)):  
        return [str(name) for name in class_names] 
    if isinstance(class_names, str) and Path(class_names).is_file():
        with open(class_names, encoding="utf-8") as f:
            text = f.read()
        try:  
            data = json.loads(text)
        except json.JSONDecodeError:   
            import yaml
            data = yaml.safe_load(text)  
        if isinstance(data, dict) and "categories" in data:    
            return [cat["name"] for cat in data["categories"]]  
        if isinstance(data, dict) and "names" in data:
            names = data["names"]     
            if isinstance(names, dict):
                return [str(names[idx]) for idx in sorted(names, key=lambda k: int(k))] 
            return [str(name) for name in names]  
        raise ValueError(f"Cannot resolve class names from {class_names}: expect COCO json or data yaml with 'names'.")
    raise ValueError("vlm_cfg.class_names must be a list of names, a COCO annotation json path, or a data yaml path.")    
   

class Florence2Supervisor(nn.Module):
    def __init__(
        self,
        in_channels: Sequence[int],
        class_names,
        model_path: str, 
        task_prompt: str = "<OD>",
        use_glca: bool = True,    
        use_lhps: bool = True,    
        visual_dim: int = 1024,   
        visual_size: int = 32,
        max_target_len: int = 2048,
        num_loc_bins: int = 1000,
        max_objs_per_level: Optional[int] = None,
        freeze_lm: bool = False,
        ce_chunk_size: int = 4096,   
    ):
        super().__init__()    
        if task_prompt not in TASK_PROMPTS:
            raise ValueError(f"Unsupported task_prompt: {task_prompt} (HBB '<OD>' / OBB '<ROD>').") 
        self.class_names = _resolve_class_names(class_names)     
        self.num_levels = len(in_channels)
        self.use_glca = use_glca
        self.use_lhps = use_lhps 
        self.visual_size = visual_size
        self.max_target_len = max_target_len
        self.num_loc_bins = num_loc_bins     
        self.max_objs_per_level = max_objs_per_level  
        self.ce_chunk_size = ce_chunk_size
     
        self.tokenizer = self._build_tokenizer(model_path) 
        self.lm, image_projection, proj_norm_state = self._build_language_model(model_path, max_target_len)
        if image_projection.shape[0] != visual_dim:
            raise ValueError(f"Expected projector input dim {visual_dim}, got {image_projection.shape[0]}")     
        self.llm_dim = image_projection.shape[1]
        self.image_projection = nn.Parameter(image_projection)
        self.image_proj_norm = nn.LayerNorm(self.llm_dim)
        self.image_proj_norm.load_state_dict(proj_norm_state)
 
        self.visual_adapters = nn.ModuleList([nn.Conv2d(c, visual_dim, kernel_size=1) for c in in_channels])
        self.glca = GlobalLocalCrossAttention() 

        prompt_ids = self.tokenizer([TASK_PROMPTS[task_prompt]], return_tensors="pt")["input_ids"]
        self.register_buffer("prompt_ids", prompt_ids, persistent=False)
        self.pad_token_id = self.lm.config.pad_token_id
        self.decoder_start_token_id = self.lm.config.decoder_start_token_id   

        if freeze_lm:     
            for p in self.lm.parameters():     
                p.requires_grad_(False)    
 
        logger.info(  
            ORANGE + f"Florence2Supervisor: levels={self.num_levels}, in_channels={list(in_channels)}, "
            f"glca={use_glca}, lhps={use_lhps}, max_target_len={max_target_len}, freeze_lm={freeze_lm}, " 
            f"classes={self.class_names}" + RESET
        )

    @staticmethod 
    def _build_tokenizer(model_path: str):
        from transformers import AutoTokenizer 

        tokenizer = AutoTokenizer.from_pretrained(model_path) 
        # 与 Florence2Processor.__init__ 一致：追加 <od>/<loc_k>/<sep> 等特殊 token
        tokenizer.add_tokens(FLORENCE2_EXTRA_TOKENS, special_tokens=True)
        return tokenizer     
   
    @staticmethod
    def _build_language_model(model_path: str, max_target_len: int):
        """从 Florence-2 checkpoint 只构建 BART 语言模型 + projector，不构建 DaViT 视觉塔与图像位置编码。"""
        from safetensors.torch import load_file    
        from transformers import BartConfig, BartForConditionalGeneration

        model_dir = Path(model_path)     
        with open(model_dir / "config.json", encoding="utf-8") as f:     
            text_config = json.load(f)["text_config"]
        config = BartConfig(**text_config)
        lm = BartForConditionalGeneration(config)   

        state = load_file(str(model_dir / "model.safetensors"))   
        prefix = "language_model."
        lm_state = {k[len(prefix):]: v.float() for k, v in state.items() if k.startswith(prefix)}
        missing, unexpected = lm.load_state_dict(lm_state, strict=False)
        # lm_head / embed_tokens 与 shared 权重绑定，checkpoint 中不单独存放 
        missing = [k for k in missing if k not in ("lm_head.weight", "model.encoder.embed_tokens.weight",    
                                                   "model.decoder.embed_tokens.weight")]     
        if missing or unexpected:  
            raise RuntimeError(f"Florence-2 LM weight mismatch: missing={missing}, unexpected={unexpected}")    
        lm.tie_weights()   

        image_projection = state["image_projection"].float()   
        proj_norm_state = {
            "weight": state["image_proj_norm.weight"].float(), 
            "bias": state["image_proj_norm.bias"].float(),   
        }    
        Florence2Supervisor._resize_position_embeddings(lm, max_target_len)  
        return lm, image_projection, proj_norm_state   

    @staticmethod
    def _resize_position_embeddings(lm: nn.Module, max_length: int) -> None: 
        """encoder/decoder 位置编码扩展到 max_length(+offset)，新增行 N(0, 0.02)，与源码一致。"""
        lm.config.max_position_embeddings = max_length
        for module_owner in (lm.model.encoder, lm.model.decoder):     
            old = module_owner.embed_positions
            offset = getattr(old, "offset", 2)  
            required = max_length + offset  
            if old.weight.shape[0] >= required:
                continue   
            new = type(old)(max_length, old.weight.shape[1])     
            with torch.no_grad():
                new.weight[: old.weight.shape[0]].copy_(old.weight)     
                nn.init.normal_(new.weight[old.weight.shape[0]:], mean=0.0, std=0.02)   
            module_owner.embed_positions = new

    def _prepare_visual_features(self, feats: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        projected = []
        for adapter, feat in zip(self.visual_adapters, feats):   
            feat = adapter(feat)
            feat = F.interpolate(feat, size=(self.visual_size, self.visual_size), mode="bilinear", align_corners=False)
            feat = feat.flatten(2).transpose(1, 2).contiguous()
            feat = torch.matmul(feat, self.image_projection.to(feat.dtype))
            projected.append(self.image_proj_norm(feat))
        return projected

    def _apply_glca(self, projected: List[torch.Tensor]) -> List[torch.Tensor]:     
        global_feat = projected[-1] 
        return [self.glca(local, global_feat) for local in projected[:-1]] + [global_feat]

    def _shift_tokens_right(self, labels: torch.Tensor) -> torch.Tensor:
        shifted = labels.new_zeros(labels.shape)
        shifted[:, 1:] = labels[:, :-1].clone()    
        shifted[:, 0] = self.decoder_start_token_id
        shifted.masked_fill_(shifted == -100, self.pad_token_id)  
        return shifted
     
    def _loss_level(self, visual: torch.Tensor, prompt_embeds: torch.Tensor, texts: List[str]) -> torch.Tensor:     
        device = visual.device 
        batch_size = visual.shape[0]    
        encoder_embeds = torch.cat([visual, prompt_embeds.to(visual.dtype)], dim=1)   
        encoder_mask = torch.ones(encoder_embeds.shape[:2], dtype=torch.long, device=device)
     
        decoder_inputs = self.tokenizer(   
            text=texts, return_tensors="pt", padding="longest", padding_side="right",
            truncation=True, max_length=self.max_target_len, return_token_type_ids=False,
        ) 
        labels = decoder_inputs["input_ids"].to(device) 
        labels = labels.masked_fill(~decoder_inputs["attention_mask"].to(device).bool(), -100)    
   
        outputs = self.lm.model(    
            inputs_embeds=encoder_embeds,  
            attention_mask=encoder_mask,
            decoder_input_ids=self._shift_tokens_right(labels),     
            use_cache=False,
        )
        hidden = outputs.last_hidden_state
        assert hidden.shape[0] == batch_size
        # 与源码 CrossEntropyLoss() 数学等价：对非 -100 位置取均值。   
        # 为省显存只对有效位置分块计算 lm_head + CE，并用 checkpoint 避免保留 [N, vocab] logits。
        valid = labels != -100   
        hidden, labels = hidden[valid], labels[valid]
        loss_sum = hidden.new_zeros((), dtype=torch.float32)
        for i in range(0, labels.numel(), self.ce_chunk_size):
            loss_sum = loss_sum + checkpoint(
                self._ce_sum, hidden[i:i + self.ce_chunk_size], labels[i:i + self.ce_chunk_size], use_reentrant=False     
            )
        return loss_sum / max(labels.numel(), 1) 
    
    def _ce_sum(self, hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = self.lm.lm_head(hidden) + self.lm.final_logits_bias.to(hidden.dtype)
        # fp32 计算避免 bf16 精度损失 
        return F.cross_entropy(logits.float(), labels, reduction="sum")  
 
    def forward(self, feats: Sequence[torch.Tensor], targets: Sequence[dict]) -> torch.Tensor:   
        if len(feats) != self.num_levels:
            raise ValueError(f"Florence2Supervisor expects {self.num_levels} feature levels, got {len(feats)}")
        projected = self._prepare_visual_features(feats)   
        visual_levels = self._apply_glca(projected) if self.use_glca and self.num_levels > 1 else projected

        batch_size = feats[0].shape[0]
        prompt_embeds = self.lm.get_input_embeddings()(self.prompt_ids).expand(batch_size, -1, -1)   
        level_targets = build_level_targets( 
            targets, self.class_names, self.num_levels, use_lhps=self.use_lhps,   
            num_loc_bins=self.num_loc_bins, max_objs_per_level=self.max_objs_per_level,   
        )
        losses = [self._loss_level(v, prompt_embeds, t) for v, t in zip(visual_levels, level_targets)] 
        return torch.stack(losses).sum()    
