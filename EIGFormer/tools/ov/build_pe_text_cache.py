#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

try:
    from tools.ov.category_loader import load_categories
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution
    from category_loader import load_categories


MAX_LEN = 32


def parse_args():
    parser = argparse.ArgumentParser(description="Build PE-Core text embedding cache from COCO or OBB categories.")
    parser.add_argument("--ann", required=True, help="COCO annotation JSON or OBB data.yaml path.")
    parser.add_argument("--out", required=True, help="Output .pth path.")
    parser.add_argument("--prompt_template", default="a photo of a {}", help="Prompt template for category names.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-path", required=True, help="Local PE-Core text encoder TorchScript path.")
    parser.add_argument("--tokenizer-path", required=True, help="Local PE-Core tokenizer directory.")
    return parser.parse_args()


def resolve_tokenizer(tokenizer_path: str | Path):
    return AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)


def _require_local_path(path: str | Path, kind: str) -> Path:
    path = Path(path)
    if path.exists():
        return path
    if kind == "model":
        raise FileNotFoundError(
            f"text-encoder 模型权重[{path}]不存在。"
            "请先运行 tools/ov/export_pe_text_encoder.py，或从网盘下载到 weight/text-encoder/。"
        )
    raise FileNotFoundError(
        f"tokenizer[{path}]不存在。"
        "请先运行 tools/ov/export_pe_text_encoder.py，或从网盘下载到 weight/text-encoder/。"
    )


def build_text_cache(
    ann_path: str | Path,
    out_path: str | Path,
    prompt_template: str,
    model_path: str | Path,
    tokenizer_path: str | Path,
    device: str | torch.device,
):
    ann_path = Path(ann_path)
    out_path = Path(out_path)
    device = torch.device(device)
    model_path = _require_local_path(model_path, "model")
    tokenizer_path = _require_local_path(tokenizer_path, "tokenizer")

    categories = load_categories(ann_path)
    prompts = [prompt_template.format(cat["name"]) for cat in categories]

    tokenizer = resolve_tokenizer(tokenizer_path)
    encoded = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    encoder = torch.jit.load(str(model_path), map_location=device).eval()
    with torch.inference_mode():
        text_feats = encoder(encoded["input_ids"].to(device)).float().cpu()

    text_feats = F.normalize(text_feats, p=2, dim=-1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "text_feats": text_feats,
            "prompts": prompts,
            "categories": [cat["name"] for cat in categories],
            "prompt_template": prompt_template,
        },
        out_path,
    )

    print(f"saved={out_path}")
    print(f"shape={tuple(text_feats.shape)}")
    print("categories=" + ",".join(cat["name"] for cat in categories))


def main():
    args = parse_args()
    build_text_cache(
        ann_path=args.ann,
        out_path=args.out,
        prompt_template=args.prompt_template,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        device=args.device,
    )


if __name__ == "__main__":
    main()
