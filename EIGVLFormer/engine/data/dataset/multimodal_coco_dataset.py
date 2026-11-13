"""     
Multimodal COCO dataset for RGB + up to two extra modalities (npy or image format).   
"""
   
import random
from pathlib import Path
from typing import Optional    

import numpy as np  
import psutil    
import torch
import torch.nn.functional as F 
from PIL import Image    

from ...core import register     
from ...misc.modality_utils import normalize_image_convert, npy_array_to_chw
from .coco_dataset import CocoDetection
   

class _ModalitySpec:
    """Internal helper describing one extra modality source."""
     
    def __init__(self, key, folder, fmt, size_mismatch, image_convert="L"): 
        self.key = key     
        self.folder = str(folder) if folder is not None else None
        self.format = fmt 
        self.size_mismatch = size_mismatch
        self.image_convert = image_convert   
        self.cache = {}
        self.paths = {} 

   
@register()
class MultimodalCocoDetection(CocoDetection):
    __inject__ = ["transforms"]    
    __share__ = ["remap_mscoco_category", "cache_imgsz", "ram_cache", "num_classes", "text_cache_file"]    
   
    EXTRA_MODALITY_KEYS = ("npy", "npy2")
  
    def __init__(   
        self,    
        img_folder, 
        npy_folder=None,    
        ann_file=None,
        transforms=None,
        modality_folder=None,
        modality_format="npy",
        modality_size_mismatch=None,
        image_convert="L",   
        npy_folder2=None,
        modality_folder2=None,  
        modality_format2=None,
        modality_size_mismatch2=None,   
        image_convert2=None,
        return_masks=False,
        remap_mscoco_category=False,
        cache_imgsz=None,     
        ram_cache=False,
        num_classes=None,
        text_cache_file: Optional[str] = None,
        npy_dtype="float32",
        image_extensions=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"),    
    ):   
        if modality_format not in {"npy", "image"}:
            raise ValueError("modality_format must be one of {'npy', 'image'}") 
        if modality_format == "npy" and npy_folder is None:  
            raise ValueError("npy_folder is required when modality_format='npy'")   
        if modality_format == "image" and modality_folder is None:
            raise ValueError("modality_folder is required when modality_format='image'")  
        if modality_size_mismatch is None:    
            modality_size_mismatch = "error" 
        if modality_size_mismatch not in {"error", "resize"}:
            raise ValueError("modality_size_mismatch must be one of {'error', 'resize'}")
    
        has_modality2 = any(x is not None for x in (npy_folder2, modality_folder2))
        if has_modality2:
            if modality_format2 is None:    
                modality_format2 = modality_format   
            if modality_format2 not in {"npy", "image"}:
                raise ValueError("modality_format2 must be one of {'npy', 'image'}")   
            if modality_format2 == "npy" and npy_folder2 is None:
                raise ValueError("npy_folder2 is required when modality_format2='npy'")  
            if modality_format2 == "image" and modality_folder2 is None:  
                raise ValueError("modality_folder2 is required when modality_format2='image'")
            if modality_size_mismatch2 is None:   
                modality_size_mismatch2 = modality_size_mismatch     
            if modality_size_mismatch2 not in {"error", "resize"}:
                raise ValueError("modality_size_mismatch2 must be one of {'error', 'resize'}")     
            if image_convert2 is None:
                image_convert2 = image_convert     
  
        requested_ram_cache = ram_cache
        image_convert = normalize_image_convert(image_convert)     
        if has_modality2 and image_convert2 is not None:  
            image_convert2 = normalize_image_convert(image_convert2)
        super().__init__(     
            img_folder=img_folder,
            ann_file=ann_file,
            transforms=transforms,
            return_masks=return_masks,     
            remap_mscoco_category=remap_mscoco_category, 
            cache_imgsz=cache_imgsz,
            ram_cache=False,    
            num_classes=num_classes,   
            text_cache_file=text_cache_file,  
        )     

        spec1_folder = modality_folder if modality_folder is not None else npy_folder 
        self.modalities = [_ModalitySpec("npy", spec1_folder, modality_format, modality_size_mismatch, image_convert)]    
        if has_modality2:     
            spec2_folder = modality_folder2 if modality_folder2 is not None else npy_folder2     
            self.modalities.append(_ModalitySpec("npy2", spec2_folder, modality_format2, modality_size_mismatch2, image_convert2))
   
        # Legacy attributes for the first modality (kept for backward compatibility).
        self.npy_folder = str(npy_folder) if npy_folder is not None else None   
        self.modality_folder = str(modality_folder) if modality_folder is not None else self.npy_folder
        self.modality_format = modality_format
        self.modality_size_mismatch = modality_size_mismatch  
        self.npy_folder2 = str(npy_folder2) if npy_folder2 is not None else None 
        self.modality_folder2 = str(modality_folder2) if modality_folder2 is not None else self.npy_folder2
        self.modality_format2 = modality_format2
        self.modality_size_mismatch2 = modality_size_mismatch2     
        self.image_convert = image_convert   
        self.image_convert2 = image_convert2  
        self.npy_dtype = npy_dtype     
        self._image_extensions = tuple(image_extensions)
   
        self._modality_paths = {}
        for spec in self.modalities: 
            if spec.format == "npy":     
                spec.paths = self._build_npy_paths(spec.folder)
            self._modality_paths[spec.key] = spec.paths     
        self._npy_paths = self.modalities[0].paths
        self._npy_cache = self.modalities[0].cache
 
        self.ram_cache = requested_ram_cache   
        if self.ram_cache:     
            if self.check_cache_ram():
                self._populate_ram_cache()   
            else:
                self.ram_cache = False  
     
    def __getitem__(self, idx):
        sample, target = self.load_item(idx)     
        if self._transforms is not None:    
            sample, target, _ = self._transforms(sample, target, self)
        return sample, target 

    def load_item(self, idx):
        rgb, target = super().load_item(idx) 
        image_id = self.ids[idx] 
        sample = {"rgb": rgb}
        for spec in self.modalities:    
            sample[spec.key] = self._load_modality(spec, image_id, rgb)     
        return sample, target
    
    def _load_modality(self, spec, image_id, rgb_image):
        if spec.format == "image": 
            return self._load_image_modality(spec, image_id, rgb_image)
        return self._load_npy(spec, image_id, rgb_image)

    def _load_image_modality(self, spec, image_id, rgb_image):   
        modality_path = self._resolve_image_modality_path(spec, image_id)    
        modality = Image.open(modality_path).convert(spec.image_convert)   
        if modality.size != rgb_image.size:   
            if spec.size_mismatch == "error":    
                raise ValueError(
                    f"Image modality/RGB size mismatch for image_id={image_id} ({spec.key}): "  
                    f"modality={modality.size}, rgb={rgb_image.size}" 
                )    
            modality = modality.resize(rgb_image.size, resample=Image.BILINEAR)
        return modality    

    def _load_npy(self, spec, image_id, rgb_image):   
        if self.ram_cache and image_id in spec.cache:     
            npy_tensor = spec.cache[image_id]     
        else:   
            npy_path = spec.paths[image_id]
            raw = np.load(npy_path, allow_pickle=True) 
            if type(raw) is np.lib.npyio.NpzFile:  
                try:
                    keys = list(raw.files) 
                    if "arr_0" in keys:     
                        npy = raw["arr_0"]
                    elif len(keys) == 1:     
                        npy = raw[keys[0]]
                    else:
                        raise ValueError(    
                            f"NPZ file with multiple arrays must contain key 'arr_0', got keys={keys} for {npy_path}"   
                        )
                except KeyError as e:  
                    raise ValueError( 
                        f"NPZ file must contain a single array or key 'arr_0', got keys={list(raw.files)} for {npy_path}"  
                    ) from e 
                finally:
                    raw.close()
            else:
                npy = raw
 
            npy_tensor = npy_array_to_chw(npy, source=str(npy_path))   
            if self.npy_dtype != "float32":  
                npy_tensor = npy_tensor.to(getattr(torch, self.npy_dtype))
   
            if self.ram_cache:  
                spec.cache[image_id] = npy_tensor
    
        rgb_w, rgb_h = rgb_image.size 
        npy_h, npy_w = npy_tensor.shape[-2:]  
        if (npy_h, npy_w) != (rgb_h, rgb_w): 
            if spec.size_mismatch == "error":
                raise ValueError(   
                    f"NPY/RGB size mismatch for image_id={image_id} ({spec.key}): "
                    f"npy=({npy_h}, {npy_w}), rgb=({rgb_h}, {rgb_w})"
                )
 
            npy_tensor = F.interpolate(
                npy_tensor.unsqueeze(0), 
                size=(rgb_h, rgb_w),
                mode="bilinear",    
                align_corners=False,    
            ).squeeze(0)
   
        return npy_tensor

    def _resolve_npy_path(self, image_id):  
        return self.modalities[0].paths[image_id] 

    def _resolve_image_modality_path(self, spec, image_id):
        image_info = self.coco.loadImgs(image_id)[0]
        rel_path = Path(image_info["file_name"])
        modality_root = Path(spec.folder)
        modality_path = modality_root / rel_path    
        if modality_path.exists():   
            return modality_path

        stem_path = modality_root / rel_path.with_suffix("")
        candidates = [stem_path.with_suffix(ext) for ext in self._image_extensions]
        for candidate in candidates:     
            if candidate.exists():
                return candidate

        flat_stem = Path(image_info["file_name"]).stem
        flat_candidates = [modality_root / f"{flat_stem}{ext}" for ext in self._image_extensions]  
        for candidate in flat_candidates:  
            if candidate.exists():
                return candidate

        tried = [modality_path, *candidates, *flat_candidates]
        raise FileNotFoundError(
            f"Missing image modality '{spec.key}' for image_id={image_id}. Tried: {', '.join(str(path) for path in tried)}"     
        )

    def _build_npy_paths(self, npy_folder):     
        npy_paths = {}
        for image_id in self.ids:    
            image_info = self.coco.loadImgs(image_id)[0] 
            stem = Path(image_info["file_name"]).stem     
            npy_path = Path(npy_folder) / f"{stem}.npy" 
            npz_path = Path(npy_folder) / f"{stem}.npz"

            if npy_path.exists():
                npy_paths[image_id] = str(npy_path) 
            elif npz_path.exists():
                npy_paths[image_id] = str(npz_path)  
            else:     
                raise FileNotFoundError(  
                    f"Missing modality file for image_id={image_id}. Tried: {npy_path} and {npz_path}"    
                )    

        return npy_paths
     
    def _estimate_npy_cache_bytes(self, sample_ids):
        total_bytes = 0 
        for spec in self.modalities:   
            if spec.format != "npy":    
                continue     
            for image_id in sample_ids:  
                path = spec.paths[image_id]
                raw = np.load(path, allow_pickle=True)
                if type(raw) is np.lib.npyio.NpzFile:    
                    try:  
                        keys = list(raw.files)
                        if "arr_0" in keys:
                            arr = raw["arr_0"] 
                        elif len(keys) == 1:     
                            arr = raw[keys[0]]   
                        else:    
                            raise ValueError(    
                                f"NPZ file with multiple arrays must contain key 'arr_0', got keys={keys} for {path}"
                            )
                        total_bytes += np.squeeze(arr).nbytes
                    finally: 
                        raw.close()    
                else: 
                    total_bytes += np.squeeze(raw).nbytes    
        return total_bytes

    def check_cache_ram(self, safety_margin=0.5):  
        """Check RGB + NPY cache requirements vs available memory.""" 
        if self.ni == 0:
            return True 
    
        gb = 1 << 30   
        n = min(self.ni, 30) 
        sample_ids = random.sample(self.ids, n)    
        total_bytes = self._estimate_image_cache_bytes(sample_ids)    
        total_bytes += self._estimate_npy_cache_bytes(sample_ids)    
        mem_required = total_bytes * self.ni / n * (1 + safety_margin)     
        mem = psutil.virtual_memory()
        cache = mem_required < mem.available
        if not cache:   
            print(    
                f'{mem_required / gb:.1f}GB RAM required to cache RGB+NPY samples '    
                f'with {int(safety_margin * 100)}% safety margin but only '
                f'{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, '    
                f"{'caching samples ✅' if cache else 'not caching samples ⚠️'}"     
            )
        return cache 
  
    def extra_repr(self) -> str:   
        s = super().extra_repr()     
        for spec in self.modalities: 
            s += f"\n {spec.key}_format: {spec.format}"
            s += f"\n {spec.key}_folder: {spec.folder}"    
            s += f"\n {spec.key}_size_mismatch: {spec.size_mismatch}"   
            if spec.format == "image":  
                s += f"\n {spec.key}_image_convert: {spec.image_convert}"  
        return s   
