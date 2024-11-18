import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)

midas.eval()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

input_image = Image.open("leaf.jpg")

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

cv2.imwrite("depth_map.png", depth_map)
