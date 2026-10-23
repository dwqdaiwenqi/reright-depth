# ═══════════════════════════════════════════════════════════════════════
# DLSS 风格超分辨率服务器（使用 Swin2SR Transformer 模型）
#
# 原理（类比 DLSS）：
#   DLSS 核心：低分辨率帧 → 神经网络推理 → 高分辨率帧
#   本实现：低分辨率图片 → Swin2SR（Window-based Transformer）→ 4× 高分辨率图片
#
# 模型 Swin2SR vs DLSS 对比：
#   DLSS：Tensor Core 加速的 CNN+时域累积，需要运动向量
#   Swin2SR：纯空域 Transformer，不需要运动向量，单帧超分
#   相同点：都是神经网络学习"幻觉出"高频细节，而非插值
#
# 为什么用 Swin2SR？
#   - 开源、可在本地运行、Apple MPS 加速
#   - Transformer 注意力机制 → 全局感受野 → 细节重建远优于双线性插值
#   - 模型约 60MB，推理快
#   - 比 ESRGAN/SRCNN 更现代，质量更好
#
# 启动：
#   cd server
#   pip install -r requirements_superres.txt
#   python superres_server.py
# ═══════════════════════════════════════════════════════════════════════

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import Swin2SRForImageSuperResolution, AutoImageProcessor
import torch
import numpy as np
from PIL import Image
import io
import uvicorn
import time

app = FastAPI(title="DLSS 风格超分辨率服务器 - Swin2SR")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 选择设备 ──────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = "mps"
    print("使用 Apple Silicon MPS 加速")
elif torch.cuda.is_available():
    DEVICE = "cuda"
    print("使用 NVIDIA CUDA 加速")
else:
    DEVICE = "cpu"
    print("使用 CPU（速度较慢）")


# ── 加载 Swin2SR 模型 ─────────────────────────────────────────────────
#
# Swin2SR 架构（Swin Transformer V2 for Image Restoration）：
#
#   输入低分辨率图 (H×W)
#       ↓  Shallow Feature Extraction（3×3 Conv）
#   特征图 (H×W×C, C=180 通道)
#       ↓  Deep Feature Extraction（6个 RSTB 残差块）
#           每个 RSTB = 6 层 Swin Transformer Layer
#           Swin Transformer = Window Attention + Shift-Window Attention
#           ← 这是关键！Attention 让模型理解全局结构
#       ↓  Image Reconstruction
#           PixelShuffle（亚像素卷积）× 4 → 输出 4H×4W
#   输出高分辨率图 (4H×4W)
#
# 训练数据：DIV2K + Flickr2K（高质量照片数据集）
# 损失函数：L1 Loss（像素对齐）
#
# 模型大小：~60MB，下载后缓存于 ~/.cache/huggingface/
#
MODEL_ID = "caidas/swin2SR-classical-sr-x4-64"

print(f"正在加载 Swin2SR 模型（{MODEL_ID}）...")
print("首次运行会下载约 60MB 模型权重...")

processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = Swin2SRForImageSuperResolution.from_pretrained(MODEL_ID)
model = model.to(DEVICE)
model.eval()

print("✓ Swin2SR 模型加载完成")
print("服务器运行中：http://localhost:8001")
print("API：POST /superres  上传低分辨率图片，返回 4× 超分结果")


def pad_to_multiple(img: Image.Image, multiple: int = 64) -> tuple[Image.Image, tuple]:
    """
    Swin2SR 的 window size=64，输入必须是 64 的倍数
    超出部分用反射填充（比零填充保留边缘细节）
    """
    w, h = img.size
    pad_w = (multiple - w % multiple) % multiple
    pad_h = (multiple - h % multiple) % multiple
    if pad_w == 0 and pad_h == 0:
        return img, (0, 0, w, h)
    padded = Image.new("RGB", (w + pad_w, h + pad_h))
    padded.paste(img, (0, 0))
    # 右边/下边用镜像填充
    if pad_w:
        right = img.crop((w - pad_w, 0, w, h)).transpose(Image.FLIP_LEFT_RIGHT)
        padded.paste(right, (w, 0))
    if pad_h:
        bottom = img.crop((0, h - pad_h, w, h)).transpose(Image.FLIP_TOP_BOTTOM)
        padded.paste(bottom, (0, h))
    return padded, (0, 0, w * 4, h * 4)  # 裁剪区域是 4× 后的原始尺寸


@app.post("/superres")
async def super_resolution(file: UploadFile = File(...)):
    """
    接收低分辨率图片，用 Swin2SR 神经网络生成 4× 高分辨率版本

    工作流程（类比 DLSS 的神经网络推理）：
      1. 解码低分辨率输入图
      2. 归一化到 [0,1]，转为 Tensor
      3. 送入 Swin2SR（6× RSTB × 6× Swin-Transformer-Layer + PixelShuffle）
      4. 模型输出 4× 高分辨率特征图
      5. clip 到 [0,1] → 转回 uint8 PNG
    """
    t0 = time.time()

    # 读取并转为 RGB
    image_bytes = await file.read()
    lr_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = lr_image.size

    # 大图保护：超过 512px 的边先缩小（避免 MPS OOM）
    MAX_DIM = 512
    if max(orig_w, orig_h) > MAX_DIM:
        scale = MAX_DIM / max(orig_w, orig_h)
        lr_image = lr_image.resize(
            (int(orig_w * scale), int(orig_h * scale)),
            Image.LANCZOS
        )
        orig_w, orig_h = lr_image.size

    # 填充到 64 的倍数（Swin window size 要求）
    padded_img, crop_box = pad_to_multiple(lr_image, 64)

    # ── 神经网络推理 ──────────────────────────────────────────────────
    #
    # processor：自动做归一化 [0,255] → [0,1]，转为 (1, 3, H, W) Tensor
    # model：Swin2SR forward pass
    #   - 浅层特征提取（1× Conv）
    #   - 深层特征提取（6 个 RSTB Residual Swin Transformer Block）
    #     每个 RSTB：6 层 Swin Transformer + 1× Conv + 残差连接
    #     Swin Transformer：W-MSA（窗口注意力）+ SW-MSA（移位窗口注意力）
    #   - 重建：PixelShuffle 上采样 ×4（把通道数转换为空间分辨率）
    # output.reconstruction：shape = (1, 3, 4H, 4W)，值域 [0,1]
    #
    inputs = processor(padded_img, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output = model(**inputs)

    # 取出结果 tensor → numpy
    # squeeze: (1,3,4H,4W) → (3,4H,4W)
    # permute: → (4H,4W,3)
    hr_tensor = output.reconstruction.squeeze(0)
    hr_np = hr_tensor.permute(1, 2, 0).cpu().float().numpy()

    # clip 防止数值溢出，转 uint8
    hr_np = np.clip(hr_np, 0.0, 1.0)
    hr_img = Image.fromarray((hr_np * 255).astype(np.uint8))

    # 裁掉 padding（只保留原始内容对应的 4× 区域）
    hr_img = hr_img.crop(crop_box)

    elapsed = time.time() - t0

    # 编码为 PNG 返回
    buf = io.BytesIO()
    hr_img.save(buf, format="PNG")
    buf.seek(0)

    # 把推理时间写到响应头（方便前端显示）
    headers = {
        "X-Inference-Time": f"{elapsed:.2f}s",
        "X-Input-Size": f"{orig_w}×{orig_h}",
        "X-Output-Size": f"{orig_w*4}×{orig_h*4}",
        "X-Model": "Swin2SR-x4",
    }

    return Response(
        content=buf.read(),
        media_type="image/png",
        headers=headers,
    )


@app.get("/")
def root():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": DEVICE,
        "scale_factor": 4,
        "architecture": "Swin Transformer V2 (RSTB × 6, Window Attention)",
        "usage": "POST /superres  上传低分辨率图片，获取 4× 超分结果",
    }


@app.get("/model-info")
def model_info():
    """返回模型架构信息（用于前端展示）"""
    param_count = sum(p.numel() for p in model.parameters())
    return {
        "model": MODEL_ID,
        "parameters": f"{param_count / 1e6:.1f}M",
        "architecture": {
            "type": "Swin Transformer V2",
            "depth": "6 RSTB blocks × 6 STL layers",
            "attention": "Window-based Multi-head Self-Attention (W-MSA + SW-MSA)",
            "window_size": 64,
            "embed_dim": 180,
            "upsample": "PixelShuffle × 4",
        },
        "training": {
            "dataset": "DIV2K + Flickr2K",
            "loss": "L1",
            "degradation": "bicubic downsampling (×4)",
        },
        "device": DEVICE,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
