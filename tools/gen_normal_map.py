#!/usr/bin/env python3
"""
Generate a normal map from an image using MarigoldNormalsPipeline (diffusers).
Model: prs-eth/marigold-normals-lcm-v0-1  (~700MB, fast)
   or: prs-eth/marigold-normals-v1-0       (higher quality, slower)

Usage:
  python tools/gen_normal_map.py assets/tests/grave/tombs.png
  python tools/gen_normal_map.py assets/tests/grave/tombs.png output_n.png --hq
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance

import torch
from diffusers import MarigoldNormalsPipeline


MODEL_FAST = "prs-eth/marigold-normals-lcm-v0-1"  # 1-step, ~700MB
MODEL_HQ   = "prs-eth/marigold-normals-v1-0"       # 10-step, sharper

FLAT_NORMAL = np.array([128, 128, 255], dtype=np.uint8)  # (0,0,1) pointing out of screen


def remove_baked_lighting(img_rgb: np.ndarray) -> np.ndarray:
    """
    Desaturate to remove pre-baked color lighting, then stretch contrast
    so the model sees shape detail rather than color gradients.
    """
    pil = Image.fromarray(img_rgb)
    # Full desaturate → pure luminance carries shape info, not light color
    gray = ImageEnhance.Color(pil).enhance(0.0)
    # Boost contrast so stone texture detail survives
    gray = ImageEnhance.Contrast(gray).enhance(1.5)
    return np.array(gray.convert("RGB"))


def generate_normal_map(input_path: str, output_path: str | None = None, high_quality: bool = False) -> str:
    src = Path(input_path)
    dst = Path(output_path) if output_path else src.with_stem(src.stem + "_n")

    raw = Image.open(src)
    has_alpha = raw.mode in ("RGBA", "LA") or (raw.mode == "P" and "transparency" in raw.info)
    print(f"Input : {src}  ({raw.width}×{raw.height})  alpha={has_alpha}")

    # Extract alpha mask before any conversion
    if has_alpha:
        rgba = np.array(raw.convert("RGBA"))
        alpha_mask = rgba[:, :, 3]          # 0=transparent, 255=opaque
        rgb = rgba[:, :, :3]
    else:
        rgb = np.array(raw.convert("RGB"))
        alpha_mask = None

    # Remove pre-baked lighting so the model focuses on geometry
    rgb_clean = remove_baked_lighting(rgb)
    img_for_model = Image.fromarray(rgb_clean)

    model_id = MODEL_HQ if high_quality else MODEL_FAST
    print(f"Model : {model_id}")

    if torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    elif torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    else:
        device, dtype = "cpu", torch.float32
    print(f"Device: {device}")

    print("Loading pipeline (first run downloads weights)...")
    pipe = MarigoldNormalsPipeline.from_pretrained(
        model_id,
        variant="fp16" if dtype == torch.float16 else None,
        torch_dtype=dtype,
    ).to(device)

    print("Running inference...")
    result = pipe(img_for_model, num_inference_steps=10 if high_quality else 1, match_input_resolution=True)

    # result.prediction: (H, W, 3) float32 in [-1, 1], OpenGL convention
    normals = result.prediction[0]
    normal_uint8 = ((normals * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)

    # Fill transparent pixels with flat normal (0,0,1) = pointing straight out
    if alpha_mask is not None:
        transparent = alpha_mask < 16          # threshold for near-transparent
        normal_uint8[transparent] = FLAT_NORMAL

    normal_img = Image.fromarray(normal_uint8, mode="RGB")

    dst.parent.mkdir(parents=True, exist_ok=True)
    normal_img.save(dst)
    print(f"Saved : {dst}")
    return str(dst)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <input_image> [output_image] [--hq]")
        print(f"Example: python {sys.argv[0]} assets/tests/grave/tombs.png")
        sys.exit(1)

    args = sys.argv[1:]
    hq = "--hq" in args
    args = [a for a in args if a != "--hq"]

    generate_normal_map(args[0], args[1] if len(args) > 1 else None, high_quality=hq)
