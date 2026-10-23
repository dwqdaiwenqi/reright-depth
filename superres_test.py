from transformers import Swin2SRForImageSuperResolution, AutoImageProcessor
from PIL import Image
import torch, numpy as np

IMG = "DLSS/15_icon_airi 1.png"
OUT = "DLSS/15_icon_airi_x4.png"

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"设备: {device}")

print("加载模型...")
processor = AutoImageProcessor.from_pretrained("caidas/swin2SR-classical-sr-x4-64")
model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x4-64").to(device).eval()

img = Image.open(IMG).convert("RGB")
print(f"输入: {img.size}")

inputs = processor(img, return_tensors="pt").to(device)
with torch.no_grad():
    out = model(**inputs).reconstruction

hr = out.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
hr = (hr.clip(0, 1) * 255).astype(np.uint8)
result = Image.fromarray(hr)
print(f"输出: {result.size}")

result.save(OUT)
print(f"已保存: {OUT}")

# 对比图：左=双线性放大（无超分），右=Swin2SR
lr_upscaled = img.resize(result.size, Image.BILINEAR)
compare = Image.new("RGB", (result.width * 2 + 8, result.height), (30, 30, 30))
compare.paste(lr_upscaled, (0, 0))
compare.paste(result, (result.width + 8, 0))

from PIL import ImageDraw, ImageFont
draw = ImageDraw.Draw(compare)
label_y = result.height - 28
draw.rectangle([0, label_y, result.width, result.height], fill=(0, 0, 0, 180))
draw.rectangle([result.width + 8, label_y, result.width * 2 + 8, result.height], fill=(0, 30, 60, 180))
draw.text((12, label_y + 6), "双线性插值（无超分）", fill=(200, 200, 200))
draw.text((result.width + 20, label_y + 6), "Swin2SR ×4（神经网络超分）", fill=(56, 189, 248))

compare.save("DLSS/compare.png")
print("对比图: DLSS/compare.png")
compare.show()
