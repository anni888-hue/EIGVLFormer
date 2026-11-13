import torch
import torch.nn.functional as F
from torchvision.ops.boxes import box_area  
import math   


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clip(min=0.0, max=1.0)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))


def bias_init_with_prob(prior_prob: float = 0.01) -> float:     
    return float(-torch.log(torch.tensor((1 - prior_prob) / prior_prob)))

     
def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:     
    x_c, y_c, w, h = x.unbind(-1)
    return torch.stack(
        [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)],
        dim=-1,    
    )  

 
def obb_cxcywha_to_polyxy(x: torch.Tensor, size: torch.Tensor, angle_factor: float = math.pi) -> torch.Tensor:    
    """归一化 OBB [cx, cy, w, h, angle/angle_factor] -> 像素四顶点 [N, 8]（VLM <ROD> 监督用）。     
     
    size 为 [W, H]（OBB 数据管线约定，见 OBBResize）。角度来自 cv2.minAreaRect 约定    
    （_canonical_min_area_rect），顶点展开使用其逆变换——与 mmrotate obb2polyxy 公式
    相差一个角度符号（cv2 与 mmrotate 旋转方向相反），顶点顺序 (x1,y1)...(x4,y4) 与
    VLM4RSDet 的 quad box 编码一致。   
    """ 
    x_c, y_c, w, h, a = x.unbind(-1)
    width, height = size[0].to(x.dtype), size[1].to(x.dtype)
    cx, cy = x_c * width, y_c * height 
    w, h = w * width, h * height
    cos, sin = torch.cos(a * angle_factor), -torch.sin(a * angle_factor)    
    hx, hy = 0.5 * w, 0.5 * h
    x1 = cx - hx * cos - hy * sin   
    y1 = cy + hx * sin - hy * cos 
    x2 = cx + hx * cos - hy * sin    
    y2 = cy - hx * sin - hy * cos
    x3 = cx + hx * cos + hy * sin   
    y3 = cy - hx * sin + hy * cos  
    x4 = cx - hx * cos + hy * sin    
    y4 = cy + hx * sin + hy * cos
    return torch.stack([x1, y1, x2, y2, x3, y3, x4, y4], dim=-1)  
 
  
def polygon_area(polyxy: torch.Tensor) -> torch.Tensor:
    """像素四顶点 [N, 8]（x1,y1,...,x4,y4 顺序）的鞋带公式面积。"""
    xs, ys = polyxy[:, 0::2], polyxy[:, 1::2]
    return (0.5 * (xs * ys.roll(-1, dims=1) - xs.roll(-1, dims=1) * ys).sum(dim=1)).abs()

  
def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)], dim=-1)  


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    area1 = box_area(boxes1)  
    area2 = box_area(boxes2)
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2]) 
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]    
    union = area1[:, None] + area2 - inter     
    iou = inter / union
    return iou, union
     

def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:  
    # assert (boxes1[:, 2:] >= boxes1[:, :2]).all(), f"boxes1: {boxes1}" 
    # assert (boxes2[:, 2:] >= boxes2[:, :2]).all(), f"boxes2: {boxes2}"     
    iou, union = box_iou(boxes1, boxes2)  

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2]) 
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / area   
