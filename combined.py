import numpy as np
import matplotlib.pyplot as plt
import skimage, skimage.io, skimage.color, skimage.filters
import torch
import cv2
from torchvision import transforms
from PIL import Image

plt.rcParams['figure.dpi'] = 150

# 显示图像的辅助函数
def display(img, title='', colorbar=False, cmap='gray'):
    plt.figure(figsize=(10, 10))
    i = plt.imshow(img, cmap=cmap if len(img.shape) == 2 else None)
    if colorbar:
        plt.colorbar(i, shrink=0.5, label='depth')
    plt.axis('off')  
    plt.title(title)
    plt.tight_layout()
    plt.show()

# 生成详细图案的函数
def make_detailed_pattern(shape=(128, 128), levels=64):
    pattern = np.random.randint(0, levels, (*shape, 3)) / levels  
    pattern = skimage.filters.gaussian(pattern, sigma=0.5)  
    return pattern

# 归一化深度图
def normalize(depthmap):
    if depthmap.max() > depthmap.min():
        return (depthmap - depthmap.min()) / (depthmap.max() - depthmap.min())
    else:
        return depthmap

# 生成 autostereogram 的函数
def make_autostereogram(depthmap, pattern, shift_amplitude=0.2, invert=False):
    depthmap = normalize(depthmap)
    if invert:
        depthmap = 1 - depthmap
    autostereogram = np.zeros((depthmap.shape[0], depthmap.shape[1], 3), dtype=pattern.dtype)
    for r in np.arange(autostereogram.shape[0]):
        for c in np.arange(autostereogram.shape[1]):
            if c < pattern.shape[1]:
                autostereogram[r, c] = pattern[r % pattern.shape[0], c % pattern.shape[1]]
            else:
                shift = int(depthmap[r, c] * shift_amplitude * pattern.shape[1])
                autostereogram[r, c] = autostereogram[r, c - pattern.shape[1] + shift]
    return autostereogram

# 使用 MiDaS 模型生成深度图
def generate_depth_map(image_path, model_type="DPT_Large"):
    # 下载预训练的 MiDaS 模型
    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    # 设置模型到评估模式
    midas.eval()

    # 使用 GPU 或 CPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)

    # 获取模型变换
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    # 加载输入图像
    input_image = Image.open(image_path).convert('RGB')

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

    # 将深度图标准化为 0 到 255，修正 dtype 为 OpenCV 格式
    depth_map = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return depth_map

# 主流程
try:
    # 生成深度图
    depth_map = generate_depth_map("image.jpg")

    # 显示原始深度图
    display(depth_map, title='Original Depth Map', colorbar=True)

    # 生成详细图案
    pattern = make_detailed_pattern(shape=(128, 128))

    # 生成 autostereogram
    autostereogram = make_autostereogram(depth_map, pattern, shift_amplitude=0.2)

    # 显示和保存 autostereogram
    display(autostereogram, title='Autostereogram')
except FileNotFoundError:
    print("Image file not found.")