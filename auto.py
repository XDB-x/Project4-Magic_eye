import numpy as np
import cv2

def generate_autostereogram(background_image, depth_map, max_offset=50, repeat_width=20):
    """
    根据背景图像和深度图生成 autostereogram。
    
    参数：
    - background_image: 背景图像（彩色图像，形状为 (H, W, 3)）
    - depth_map: 深度图（灰度图像，形状为 (H, W)）
    - max_offset: 最大的水平偏移量，用于确定最大深度效果
    - repeat_width: 每行用于生成重复图案的宽度
    
    返回：
    - autostereogram: 生成的 autostereogram 图像
    """
    height, width = depth_map.shape
    autostereogram = np.copy(background_image)  # 复制背景图像作为起点

    # 使用 repeat_width 生成重复的条纹，增加背景的结构性
    for y in range(height):
        for x in range(repeat_width, width):
            depth = depth_map[y, x] / 255.0  # 将深度图的像素值标准化为 [0, 1] 范围
            offset = int(max_offset * depth)  # 根据深度计算水平偏移量
            
            # 计算当前位置与偏移位置之间的值
            if x - offset >= 0:
                autostereogram[y, x] = autostereogram[y, x - offset]
            else:
                autostereogram[y, x] = autostereogram[y, x % repeat_width]

    return autostereogram

# 加载深度图并增强对比度
depth_map = cv2.imread("depth_map.png", cv2.IMREAD_GRAYSCALE)
depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
depth_map = cv2.equalizeHist(depth_map)

# 创建一个随机噪声背景图像，但限制一定的重复性
height, width = depth_map.shape
random_strip = np.random.randint(0, 256, (height, 20, 3), dtype=np.uint8)
background_image = np.tile(random_strip, (1, int(width / 20) + 1, 1))[:, :width, :]

# 调用生成函数
autostereogram = generate_autostereogram(background_image, depth_map)

# 显示和保存生成的 autostereogram
cv2.imshow("Autostereogram", autostereogram)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("final_autostereogram.png", autostereogram)
