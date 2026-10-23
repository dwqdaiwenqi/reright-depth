# ═══════════════════════════════════════════════════════════════════════
# 把 Swin2SR PyTorch 模型导出为 ONNX，供浏览器 onnxruntime-web + WebGPU 使用
#
# 为什么用 optimum 而不是 torch.onnx.export？
#   Swin2SR 的 Window Attention 内部有大量 reshape + permute
#   裸 torch.onnx.export 会产生大量 Gather/Unsqueeze 节点，
#   动态 shape 支持差，ONNX Runtime WebGPU EP 跑起来容易出错
#   optimum 专门处理 transformers 模型，生成的 ONNX 更干净
#
# 运行方法：
#   pip install optimum[exporters] onnx onnxsim
#   python export_onnx.py
#
# 输出文件：
#   ../DLSS/swin2sr_x4.onnx     ← 浏览器加载这个
#   ../DLSS/swin2sr_x4_sim.onnx ← 简化版（推荐，体积更小推理更快）
# ═══════════════════════════════════════════════════════════════════════

import os, sys, argparse
import torch
import numpy as np
from pathlib import Path

OUTPUT_DIR  = Path(__file__).parent.parent / "DLSS"
MODEL_ID    = "caidas/swin2SR-classical-sr-x4-64"
ONNX_NAME   = "swin2sr_x4.onnx"
SIM_NAME    = "swin2sr_x4_sim.onnx"

def export_with_optimum():
    """优先用 optimum — 对 transformers 模型生成更干净的 ONNX 计算图"""
    try:
        from optimum.exporters.onnx import main_export
    except ImportError:
        print("optimum 未安装，跳过（尝试裸 torch.onnx 方式）")
        return False

    print("使用 optimum 导出 Swin2SR → ONNX ...")
    out_dir = OUTPUT_DIR / "swin2sr_onnx_tmp"
    out_dir.mkdir(exist_ok=True)

    main_export(
        model_name_or_path=MODEL_ID,
        output=out_dir,
        task="image-to-image",
        opset=17,
        device="cpu",
    )

    # optimum 输出到子目录，搬到目标位置
    src = out_dir / "model.onnx"
    if src.exists():
        import shutil
        shutil.copy(src, OUTPUT_DIR / ONNX_NAME)
        print(f"✓ 导出完成：{OUTPUT_DIR / ONNX_NAME}")
        return True
    return False


def export_with_torch():
    """
    裸 torch.onnx.export — 不依赖 optimum
    注意：Swin2SR 用 window attention，动态 shape 支持有限
    固定输入尺寸可以正常跑，但不能自由改变 H/W
    """
    from transformers import Swin2SRForImageSuperResolution

    print(f"加载 {MODEL_ID} ...")
    model = Swin2SRForImageSuperResolution.from_pretrained(MODEL_ID)
    model.eval()

    # Swin2SR window_size=64，输入必须是 64 的倍数
    # 固定 256×256（4× 后输出 1024×1024）
    H, W = 256, 256
    dummy = torch.zeros(1, 3, H, W, dtype=torch.float32)

    onnx_path = OUTPUT_DIR / ONNX_NAME
    print(f"导出 ONNX（固定尺寸 {H}×{W}）→ {onnx_path} ...")

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy,),
            str(onnx_path),
            input_names=["pixel_values"],
            output_names=["reconstruction"],
            # 不声明 dynamic_axes，固定尺寸，WebGPU 兼容性最好
            opset_version=17,
            do_constant_folding=True,
        )

    print(f"✓ 导出完成：{onnx_path}")
    print(f"  模型输入：float32 (1, 3, {H}, {W})  值域 [0, 1]")
    print(f"  模型输出：float32 (1, 3, {H*4}, {W*4})  值域 [0, 1]")
    return True


def simplify_onnx(input_path: Path, output_path: Path):
    """
    用 onnxsim 简化计算图
    作用：
      - 合并冗余常量节点（Constant + Gather → 直接嵌入）
      - 消除 Identity 节点
      - 折叠可静态确定的子图
    结果：节点数减少约 30%，推理速度提升，WebGPU EP 兼容性更好
    """
    try:
        import onnx
        from onnxsim import simplify
    except ImportError:
        print("onnxsim 未安装，跳过简化步骤（pip install onnxsim）")
        return

    print("简化 ONNX 计算图 ...")
    model = onnx.load(str(input_path))
    simplified, ok = simplify(model)
    if ok:
        onnx.save(simplified, str(output_path))
        orig_mb  = input_path.stat().st_size  / 1024 / 1024
        sim_mb   = output_path.stat().st_size / 1024 / 1024
        print(f"✓ 简化完成：{orig_mb:.1f} MB → {sim_mb:.1f} MB")
        print(f"  原始节点数：{len(model.graph.node)}")
        print(f"  简化节点数：{len(simplified.graph.node)}")
    else:
        print("⚠ 简化失败（不影响正确性），使用原始 ONNX")
        import shutil
        shutil.copy(input_path, output_path)


def verify_onnx(onnx_path: Path):
    """用 onnxruntime CPU EP 验证模型输出是否正常"""
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime 未安装，跳过验证")
        return

    print("验证 ONNX 模型 ...")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # 取输入尺寸（固定 shape 时可直接读 dim_value）
    inp = sess.get_inputs()[0]
    shape = inp.shape
    H = shape[2] if isinstance(shape[2], int) else 256
    W = shape[3] if isinstance(shape[3], int) else 256

    dummy = np.random.rand(1, 3, H, W).astype(np.float32)
    out = sess.run(None, {"pixel_values": dummy})[0]

    assert out.shape == (1, 3, H * 4, W * 4), f"输出 shape 异常：{out.shape}"
    assert out.min() >= -0.1 and out.max() <= 1.1, f"输出值域异常：[{out.min():.3f}, {out.max():.3f}]"
    print(f"✓ 验证通过：输入 (1,3,{H},{W}) → 输出 (1,3,{H*4},{W*4})")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. 导出（优先 optimum，失败则裸 torch）
    ok = export_with_optimum()
    if not ok:
        ok = export_with_torch()
    if not ok:
        print("❌ 导出失败")
        sys.exit(1)

    # 2. 简化
    simplify_onnx(OUTPUT_DIR / ONNX_NAME, OUTPUT_DIR / SIM_NAME)

    # 3. 验证
    verify_onnx(OUTPUT_DIR / SIM_NAME)

    print()
    print("=" * 60)
    print("导出完成！")
    print(f"  ONNX 文件：{OUTPUT_DIR / SIM_NAME}")
    print()
    print("浏览器加载方式：")
    print("  用 onnxruntime-web WebGPU EP 加载 swin2sr_x4_sim.onnx")
    print("  详见 DLSS/superres-webgpu.html")
    print("=" * 60)
