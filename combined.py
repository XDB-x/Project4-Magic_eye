import os
import numpy as np
import matplotlib.pyplot as plt
import skimage, skimage.io, skimage.color, skimage.filters
import torch
import cv2
from PIL import Image

plt.rcParams['figure.dpi'] = 150

def display(img, title='', colorbar=False, cmap='gray'):
    plt.figure(figsize=(10, 10))
    i = plt.imshow(img, cmap=cmap if len(img.shape) == 2 else None)
    if colorbar:
        plt.colorbar(i, shrink=0.5, label='depth')
    plt.axis('off')  
    plt.title(title)
    plt.tight_layout()
    plt.show()

def make_detailed_pattern(shape=(128, 128)):

    texture_image_path = "texture_basic/forest.jpg"
    texture_image = Image.open(texture_image_path).convert('RGB').resize(shape)
    pattern = np.array(texture_image) / 255.0
    return pattern

def normalize(depthmap):
    if depthmap.max() > depthmap.min():
        return (depthmap - depthmap.min()) / (depthmap.max() - depthmap.min())
    else:
        return depthmap

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

def generate_depth_map(image_path, model_type="DPT_Large"):
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    input_image = Image.open(image_path).convert('RGB')
    input_image_np = np.array(input_image) / 255.0
    input_image_tensor = torch.from_numpy(input_image_np).permute(2, 0, 1).unsqueeze(0)
    input_image_tensor = input_image_tensor.to(device).float()
    with torch.no_grad():
        prediction = midas(input_image_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=input_image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map = prediction.cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return depth_map

def apply_edge_gradient(depthmap, fade_width=20):
    h, w = depthmap.shape
    y, x = np.ogrid[:h, :w]

    distance_to_edge = np.minimum(np.minimum(x, w - x - 1), np.minimum(y, h - y - 1))
    gradient_mask = np.clip(distance_to_edge / fade_width, 0, 1)

    return depthmap * gradient_mask

try:
    depth_map = generate_depth_map("inputPhoto/tree.jpg")
    depth_map = apply_edge_gradient(depth_map, fade_width=30)

    display(depth_map, title='Original Depth Map', colorbar=True)

    depth_output_folder = "depthMap"
    os.makedirs(depth_output_folder, exist_ok=True)
    depth_save_path = os.path.join(depth_output_folder, "combined_depth_map.png")
    cv2.imwrite(depth_save_path, depth_map)

    pattern = make_detailed_pattern(shape=(256, 256))
    autostereogram = make_autostereogram(depth_map, pattern, shift_amplitude=0.2)
    display(autostereogram, title='Autostereogram')

    autostereogram_output_folder = "autostereogram"
    os.makedirs(autostereogram_output_folder, exist_ok=True)
    autostereogram_save_path = os.path.join(autostereogram_output_folder, "combined_autostereogram.png")
    autostereogram_bgr = cv2.cvtColor((autostereogram * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(autostereogram_save_path, autostereogram_bgr)

except FileNotFoundError:
    print("Image file not found.")
