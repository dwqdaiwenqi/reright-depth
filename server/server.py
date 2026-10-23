# ═══════════════════════════════════════════════════════════════════════
# 法线图估计服务器（使用 Marigold-Normal 专用法线模型）
#
# 为什么换成 Marigold-Normal？
#   之前的方案：Depth Anything V2（深度图）→ Sobel 算法推算法线
#     问题：深度图是为深度感知训练的，不是为法线优化的
#          Sobel 推算法线精度有限，细节丢失
#
#   现在的方案：Marigold-Normal → 直接输出法线图
#     - 专门为法线估计训练的模型
#     - 基于 Stable Diffusion（训练数据包含大量动漫/插画/游戏原画）
#     - 直接理解图片的几何结构，不需要先算深度再推法线
#     - 对二次元/插画效果明显更好
#     - LCM 加速版只需 4 步推理，比完整版快很多
#
# 模型信息：
#   名称：Marigold-Normal LCM v0.1
#   来源：ETH Zurich（苏黎世联邦理工学院）
#   论文：Marigold: Repurposing Diffusion-Based Image Generators for Monocular Depth
#   大小：~4GB（首次运行自动下载，缓存后不再下载）
#   速度：4 步推理，约 5-15 秒/张（Apple Silicon M 系列）
#
# 启动方法：
#   cd server
#   pip install -r requirements.txt    ← 第一次需要
#   python server.py
# ═══════════════════════════════════════════════════════════════════════

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from diffusers import MarigoldNormalsPipeline
import torch
import numpy as np
from PIL import Image
import io
import uvicorn

# ── FastAPI 初始化 ────────────────────────────────────────────────────
app = FastAPI(title="法线图估计服务器 - Marigold-Normal")

# CORS：允许浏览器跨域请求
# 因为网页（file:// 或 localhost:端口）和服务器（localhost:8000）域不同
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 选择运行设备 ──────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = "mps"    # Apple Silicon GPU（M1/M2/M3/M4）
    DTYPE  = torch.float32  # MPS 对 float16 支持不完整，用 float32 更稳定
elif torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE  = torch.float16  # NVIDIA GPU 支持 float16，更快
else:
    DEVICE = "cpu"
    DTYPE  = torch.float32

print(f"使用设备：{DEVICE}（数据精度：{DTYPE}）")


# ── 加载 Marigold-Normal 模型 ─────────────────────────────────────────
#
# 对比之前的 Depth Anything V2：
#   Depth Anything：图片 → 深度图（每像素一个距离值）
#   Marigold-Normal：图片 → 法线图（每像素一个 XYZ 方向向量）
#
# Marigold-Normal 是扩散模型（Diffusion Model）：
#   不是一步算出结果，而是从随机噪声迭代 N 步"雕刻"出法线图
#   LCM（Latent Consistency Model）版本用 4 步就能得到高质量结果
#   原版需要 50 步，LCM 版快了约 10 倍
#
# 首次运行会下载约 4GB 模型权重，之后缓存在：
#   Mac：~/.cache/huggingface/hub/
print("正在加载 Marigold-Normal 模型（首次运行约下载 4GB，请耐心等待）...")

pipe = MarigoldNormalsPipeline.from_pretrained(
    "prs-eth/marigold-normals-lcm-v0-1",   # ETH Zurich 发布的 LCM 加速版
    torch_dtype=DTYPE,
).to(DEVICE)

# 内存优化：不用梯度（推理不需要反向传播）
# 对应 WebGPU 里的"只读 Buffer"，推理时权重不会被更新
pipe.unet.eval()

print("✓ 模型加载完成")
print("服务器运行中：http://localhost:8000")
print("在浏览器 index.html 里点「生成法线图」按钮即可使用")


# ── API：生成法线图 ───────────────────────────────────────────────────
@app.post("/normal")
async def estimate_normal(file: UploadFile = File(...)):
    """
    接收图片，用 Marigold-Normal 生成法线图，返回 PNG

    法线图格式（和游戏引擎标准一致）：
      R 通道 = X 方向（左=-1→0, 右=+1→255）
      G 通道 = Y 方向（下=-1→0, 上=+1→255）
      B 通道 = Z 方向（朝里=-1→0, 朝外=+1→255）
      中性法线（正面朝相机）= RGB(128, 128, 255) ← 这是法线图呈现蓝紫色的原因
    """
    # 读取上传的图片
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # ── 运行 Marigold-Normal 推理 ─────────────────────────────────────
    # num_inference_steps=4：LCM 版 4 步推理，质量和速度的平衡点
    #   步数越少越快但细节略少，步数越多越精细但更慢
    #   LCM 4步 ≈ 原版 50步 的质量
    #
    # output_type="np"：输出 numpy 数组（而不是 PIL Image）
    #   shape: (1, H, W, 3)，值域：[-1.0, 1.0]（XYZ 法线分量）
    with torch.no_grad():  # 关闭梯度计算，节省显存
        result = pipe(
            image,
            num_inference_steps=4,
            output_type="np",
        )

    # result.prediction shape: (batch=1, H, W, 3)，取第一张
    # 值域 [-1, 1]：[-1, -1, -1] = 朝里下左，[1, 1, 1] = 朝外上右
    normal = result.prediction[0]  # shape: (H, W, 3)

    # 转换到 RGB [0, 255]：
    #   公式：rgb = (normal + 1) / 2 * 255
    #   -1 → 0, 0 → 128, 1 → 255
    normal_rgb = ((normal + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)

    # 返回法线图 PNG
    normal_img = Image.fromarray(normal_rgb)
    buf = io.BytesIO()
    normal_img.save(buf, format="PNG")
    buf.seek(0)

    return Response(content=buf.read(), media_type="image/png")


# ── API：健康检查 ─────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "ok",
        "model": "prs-eth/marigold-normals-lcm-v0-1",
        "device": DEVICE,
        "message": "POST /normal 上传图片获取法线图",
    }


# ── 启动 ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
