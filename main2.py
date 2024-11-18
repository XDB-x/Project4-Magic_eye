import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

# 下载预训练的 MiDaS 模型
model_type = "DPT_Large"  # 选择模型类型，例如 "DPT_Large", "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)

# 设置模型到评估模式
midas.eval()

# 使用 GPU 或 CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)

# 获取模型变换
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

# 加载输入图像
input_image = Image.open("leaf.jpg")

# 手动将输入图像转换为张量
input_image_np = np.array(input_image) / 255.0  # 归一化
input_image_tensor = torch.from_numpy(input_image_np).permute(2, 0, 1).unsqueeze(0)  # 转换为 (B, C, H, W) 格式
input_image_tensor = input_image_tensor.to(device).float()  # 确保数据类型为浮点数

# 推理
with torch.no_grad():
    prediction = midas(input_image_tensor)

    # 调整尺寸
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=input_image.size[::-1],  # PIL 图像的尺寸是 (宽, 高)，需要反转以匹配 (高, 宽)
        mode="bicubic",
        align_corners=False,
    ).squeeze()

# 转换为 NumPy 数组
depth_map = prediction.cpu().numpy()

# 将深度图标准化为 0 到 255
depth_map = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# 保存深度图
cv2.imwrite("depth_map.png", depth_map)
