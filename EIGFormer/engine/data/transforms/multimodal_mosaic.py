""" 
Multimodal Mosaic for RGB + NPY samples.
"""
  
import random     
  
import torch   
import torch.nn.functional as torch_F 
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F  
from PIL import Image as PILImage     

from .._misc import convert_to_tv_tensor 
from .._misc import Image as TVImage  
from .mosaic import merge_mosaic_targets
from ...core import register 
    

@register()
class MultimodalMosaic(T.Transform):
    def __init__(  
        self,   
        output_size=320,
        max_size=None,     
        rotation_range=0,   
        translation_range=(0.1, 0.1),
        scaling_range=(0.5, 1.5),  
        probability=1.0,   
        fill_value=114,
        use_cache=True,
        max_cached_images=50,  
        random_pop=True,     
    ) -> None:
        super().__init__()    
        self.resize = T.Resize(size=output_size, max_size=max_size)
        self.probability = probability     
        self.affine_transform = T.RandomAffine(
            degrees=rotation_range,
            translate=translation_range,
            scale=scaling_range, 
            fill=fill_value,
        ) 
        self.use_cache = use_cache 
        self.mosaic_cache = []     
        self.max_cached_images = max_cached_images
        self.random_pop = random_pop
   
    def load_samples_from_dataset(self, sample, target, dataset):
        sample, target = self._resize_sample_and_target(sample, target)     
        mosaic_samples = [{"sample": sample, "target": target}]    
        max_height, max_width = self._get_size(sample["rgb"]) 
     
        sample_indices = random.choices(range(len(dataset)), k=3)     
        for idx in sample_indices:
            sampled, sampled_target = dataset.load_item(idx)     
            sampled, sampled_target = self._resize_sample_and_target(sampled, sampled_target)
            h, w = self._get_size(sampled["rgb"])
            max_height, max_width = max(max_height, h), max(max_width, w)  
            mosaic_samples.append({"sample": sampled, "target": sampled_target})     

        return mosaic_samples, max_height, max_width  
  
    def load_samples_from_cache(self, sample, target, cache):
        sample, target = self._resize_sample_and_target(sample, target)
        cache.append({"sample": self._clone_sample(sample), "target": self._clone_target(target)})

        if len(cache) > self.max_cached_images:
            if self.random_pop:     
                index = random.randint(0, len(cache) - 2)  
            else:  
                index = 0
            cache.pop(index)  

        sample_indices = random.choices(range(len(cache)), k=3)  
        mosaic_samples = [   
            {"sample": self._clone_sample(cache[idx]["sample"]), "target": self._clone_target(cache[idx]["target"])}
            for idx in sample_indices 
        ]    
        mosaic_samples = [{"sample": self._clone_sample(sample), "target": self._clone_target(target)}] + mosaic_samples 
 
        sizes = [self._get_size(one["sample"]["rgb"]) for one in mosaic_samples]
        max_height = max(size[0] for size in sizes)
        max_width = max(size[1] for size in sizes)  

        return mosaic_samples, max_height, max_width

    def create_mosaic(self, mosaic_samples, max_height, max_width):
        placement_offsets = [[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]   
        ref_sample = mosaic_samples[0]["sample"]   
        ref_tensors = {key: self._to_chw_tensor(value) for key, value in ref_sample.items()}

        merged = { 
            key: torch.zeros(
                (value.shape[0], max_height * 2, max_width * 2),    
                dtype=value.dtype,
                device=value.device,
            ) 
            for key, value in ref_tensors.items()  
        }    
  
        merged_targets = []
        for i, one in enumerate(mosaic_samples):
            sample = one["sample"]     
            target = self._clone_target(one["target"])
            x_offset, y_offset = placement_offsets[i]
  
            sample_tensors = {key: self._to_chw_tensor(value) for key, value in sample.items()}
            h, w = sample_tensors["rgb"].shape[-2:]  
 
            for key, tensor in sample_tensors.items():  
                merged[key][:, y_offset : y_offset + h, x_offset : x_offset + w] = tensor   

            merged_targets.append(target) 
 
        merged_sample = {} 
        for key, tensor in merged.items():  
            if isinstance(ref_sample[key], PILImage.Image):
                merged_sample[key] = F.to_pil_image(tensor)
            elif hasattr(ref_sample[key], "__class__") and ref_sample[key].__class__.__name__ == "Image":
                merged_sample[key] = ref_sample[key].__class__(tensor)  
            else:
                merged_sample[key] = tensor     
   
        merged_target = merge_mosaic_targets(
            merged_targets,
            placement_offsets,    
            mosaic_height=max_height * 2,
            mosaic_width=max_width * 2,
        )
    
        return merged_sample, merged_target   
   
    @staticmethod  
    def _clone_target(tensor_dict):
        cloned = {}
        for key, value in tensor_dict.items():
            cloned[key] = value.clone() if hasattr(value, "clone") else value     
        return cloned    
  
    @staticmethod
    def _clone_sample(sample):
        cloned = {}    
        for key, value in sample.items():
            if isinstance(value, PILImage.Image):  
                cloned[key] = value.copy() 
            else:
                cloned[key] = value.clone() if hasattr(value, "clone") else value    
        return cloned
   
    @staticmethod
    def _get_size(image):
        get_size_func = F.get_size if hasattr(F, "get_size") else F.get_spatial_size     
        return get_size_func(image)

    @staticmethod    
    def _to_tensor(image):
        if isinstance(image, PILImage.Image):
            return F.pil_to_tensor(image)
        if isinstance(image, torch.Tensor):     
            return image 
        return torch.as_tensor(image)
  
    @staticmethod
    def _to_chw_tensor(image):
        tensor = MultimodalMosaic._to_tensor(image)  
        if tensor.ndim == 2: 
            tensor = tensor.unsqueeze(0)
        return tensor

    def _resize_sample_and_target(self, sample, target):
        rgb, target = self.resize(sample["rgb"], target)
        resize_h, resize_w = self._get_size(rgb)     
    
        out = {"rgb": rgb}
        for key, value in sample.items():
            if key == "rgb":
                continue   
            npy = self._to_chw_tensor(value) 
            orig_dtype = npy.dtype
            npy = torch_F.interpolate(
                npy.unsqueeze(0).to(dtype=torch.float32), 
                size=(resize_h, resize_w),   
                mode="bilinear", 
                align_corners=False,    
            ).squeeze(0) 
            if orig_dtype != torch.float32:   
                npy = npy.to(dtype=orig_dtype)   
            out[key] = npy

        return out, target
    
    def forward(self, *inputs):
        if len(inputs) == 1: 
            inputs = inputs[0]    
        sample, target, dataset = inputs

        if self.probability < 1.0 and random.random() > self.probability:    
            return sample, target, dataset
  
        if self.use_cache:  
            mosaic_samples, max_height, max_width = self.load_samples_from_cache(sample, target, self.mosaic_cache)
        else:
            mosaic_samples, max_height, max_width = self.load_samples_from_dataset(sample, target, dataset)    

        mosaic_sample, mosaic_target = self.create_mosaic(mosaic_samples, max_height, max_width)

        if "boxes" in mosaic_target:     
            mosaic_target["boxes"] = convert_to_tv_tensor(    
                mosaic_target["boxes"],
                "boxes",   
                box_format="xyxy",  
                spatial_size=mosaic_sample["rgb"].size[::-1] if isinstance(mosaic_sample["rgb"], PILImage.Image)    
                else mosaic_sample["rgb"].shape[-2:],
            ) 
        if "masks" in mosaic_target: 
            mosaic_target["masks"] = convert_to_tv_tensor(mosaic_target["masks"], "masks")     
  
        affine_sample = self._prepare_sample_for_affine(mosaic_sample)  
        affine_sample, mosaic_target = self.affine_transform(affine_sample, mosaic_target)
        mosaic_sample = self._restore_sample_after_affine(affine_sample, mosaic_sample)    
        return mosaic_sample, mosaic_target, dataset

    @staticmethod 
    def _prepare_sample_for_affine(sample):    
        out = dict(sample)     
        for key, value in out.items():
            if isinstance(value, torch.Tensor) and type(value).__name__ != "Image":    
                out[key] = TVImage(value if value.ndim == 3 else value.unsqueeze(0))    
        return out   

    @staticmethod    
    def _restore_sample_after_affine(sample, ref_sample):   
        out = dict(sample)  

        for key, ref_value in ref_sample.items():
            if isinstance(ref_value, PILImage.Image) and type(out[key]).__name__ != "Image":     
                out[key] = F.to_pil_image(out[key]) 
            elif isinstance(ref_value, torch.Tensor) and type(out[key]).__name__ == "Image":
                out[key] = out[key].as_subclass(torch.Tensor)
        return out   
