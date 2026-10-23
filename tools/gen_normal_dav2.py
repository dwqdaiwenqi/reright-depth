#!/usr/bin/env python3
"""
Generate a normal map from an anime illustration using Depth Anything V2.

Install:
  pip install transformers torch torchvision pillow numpy flask flask-cors

CLI Usage:
  python tools/gen_normal_dav2.py assets/tests/chara.png
  python tools/gen_normal_dav2.py assets/tests/chara.png out_n.png --size base --scale 6
  python tools/gen_normal_dav2.py assets/tests/chara.png --depth-only

HTTP Server Usage:
  python tools/gen_normal_dav2.py --serve [--port 7860] [--size base]

  POST /depth
    Body (multipart/form-data or JSON base64):
      image   : file upload  OR  base64 string (field: image_b64)
      size    : small | base | large  (default: small)
    Response:
      { "depth": "<base64 png>" }

Options:
  --size   small | base | large   model size (default: small, ~100MB)
  --scale  float                  bump intensity (default: 5.0)
  --depth-only                    only save depth map, skip normal conversion
  --serve                         start HTTP server instead of CLI
  --port   int                    server port (default: 7860)
"""

import sys
import io
import base64
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter

import torch
from transformers import pipeline as hf_pipeline

FLAT_NORMAL = np.array([128, 128, 255], dtype=np.uint8)

MODEL = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base":  "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def estimate_depth(img: Image.Image, model_id: str, device: str) -> np.ndarray:
    pipe = hf_pipeline("depth-estimation", model=model_id, device=device)
    result = pipe(img)
    depth = np.array(result["depth"].convert("L")).astype(np.float32) / 255.0
    return depth


def depth_to_normal(depth: np.ndarray, scale: float = 5.0) -> np.ndarray:
    # light smooth to reduce gradient noise
    pil = Image.fromarray((depth * 255).astype(np.uint8))
    pil = pil.filter(ImageFilter.GaussianBlur(radius=2))
    depth = np.array(pil).astype(np.float32) / 255.0

    dz_dx = np.gradient(depth, axis=1) * scale
    dz_dy = np.gradient(depth, axis=0) * scale

    nx = -dz_dx
    ny =  dz_dy
    nz = np.ones_like(depth)

    length = np.sqrt(nx**2 + ny**2 + nz**2)
    nx /= length
    ny /= length
    nz /= length

    rgb = np.stack([nx * 0.5 + 0.5, ny * 0.5 + 0.5, nz * 0.5 + 0.5], axis=-1)
    return (rgb * 255).clip(0, 255).astype(np.uint8)


def run(input_path: str, output_path: str | None, size: str, scale: float, depth_only: bool):
    src = Path(input_path)
    dst = Path(output_path) if output_path else src.with_stem(src.stem + "_n")
    dst_depth = src.with_stem(src.stem + "_depth")

    raw = Image.open(src)
    has_alpha = raw.mode in ("RGBA", "LA")
    print(f"Input : {src}  ({raw.width}×{raw.height})  alpha={has_alpha}")

    if has_alpha:
        rgba = np.array(raw.convert("RGBA"))
        alpha_mask = rgba[:, :, 3]
        img_rgb = Image.fromarray(rgba[:, :, :3])
    else:
        img_rgb = raw.convert("RGB")
        alpha_mask = None

    model_id = MODEL[size]
    device = get_device()
    print(f"Model : {model_id}")
    print(f"Device: {device}")

    print("Estimating depth...")
    depth = estimate_depth(img_rgb, model_id, device)

    depth_vis = (depth * 255).astype(np.uint8)
    if alpha_mask is not None:
        depth_vis[alpha_mask < 16] = 0
    dst_depth.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(depth_vis).save(dst_depth)
    print(f"Depth : {dst_depth}")

    if depth_only:
        return

    print(f"Depth → normal  (scale={scale})...")
    normal = depth_to_normal(depth, scale=scale)

    if alpha_mask is not None:
        normal[alpha_mask < 16] = FLAT_NORMAL

    Image.fromarray(normal, "RGB").save(dst)
    print(f"Normal: {dst}")


def image_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def b64_to_image(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data))


def depth_image(img_rgb: Image.Image, alpha_mask, model_id: str, device: str) -> Image.Image:
    depth = estimate_depth(img_rgb, model_id, device)
    depth_vis = (depth * 255).astype(np.uint8)
    if alpha_mask is not None:
        depth_vis[alpha_mask < 16] = 0
    return Image.fromarray(depth_vis)


def serve(default_size: str, port: int):
    from flask import Flask, request, jsonify
    from flask_cors import CORS

    app = Flask(__name__)
    CORS(app)

    # Cache loaded pipeline to avoid reloading on every request
    _cache: dict = {}

    def get_pipe(size: str):
        if size not in _cache:
            model_id = MODEL.get(size, MODEL["small"])
            device = get_device()
            print(f"Loading model {model_id} on {device}...")
            _cache[size] = hf_pipeline("depth-estimation", model=model_id, device=device)
            print("Model ready.")
        return _cache[size]

    @app.post("/depth")
    def depth_endpoint():
        try:
            size = request.form.get("size") or (request.json or {}).get("size") or default_size
            if size not in MODEL:
                return jsonify({"error": f"size must be one of: {', '.join(MODEL)}"}), 400

            # Accept file upload or base64 JSON
            if "image" in request.files:
                raw = Image.open(request.files["image"].stream)
            elif request.is_json and "image_b64" in request.json:
                raw = b64_to_image(request.json["image_b64"])
            else:
                return jsonify({"error": "Provide image file or image_b64"}), 400

            has_alpha = raw.mode in ("RGBA", "LA")
            if has_alpha:
                rgba = np.array(raw.convert("RGBA"))
                alpha_mask = rgba[:, :, 3]
                img_rgb = Image.fromarray(rgba[:, :, :3])
            else:
                img_rgb = raw.convert("RGB")
                alpha_mask = None

            pipe = get_pipe(size)
            result = pipe(img_rgb)
            depth = np.array(result["depth"].convert("L")).astype(np.float32) / 255.0
            depth_vis = (depth * 255).astype(np.uint8)
            if alpha_mask is not None:
                depth_vis[alpha_mask < 16] = 0

            depth_img = Image.fromarray(depth_vis)
            return jsonify({"depth": image_to_b64(depth_img)})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    print(f"Server starting on http://0.0.0.0:{port}")
    print(f"Default model size: {default_size}")
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    def pop_arg(flag, default=None):
        if flag in argv:
            i = argv.index(flag)
            val = argv[i + 1]
            argv[i:i+2] = []
            return val
        return default

    if "--serve" in argv:
        argv.remove("--serve")
        size = pop_arg("--size", "small")
        port = int(pop_arg("--port", 7860))
        serve(size, port)
        sys.exit(0)

    depth_only = "--depth-only" in argv
    if depth_only:
        argv.remove("--depth-only")

    size  = pop_arg("--size",  "small")
    scale = float(pop_arg("--scale", 5.0))

    if size not in MODEL:
        print(f"--size must be one of: {', '.join(MODEL)}")
        sys.exit(1)

    positional = [a for a in argv if not a.startswith("--")]
    run(positional[0], positional[1] if len(positional) > 1 else None, size, scale, depth_only)
