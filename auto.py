import numpy as np
import cv2
import os

def generate_autostereogram(background_image, depth_map, max_offset=50, repeat_width=20):
    height, width = depth_map.shape
    autostereogram = np.copy(background_image)

    for y in range(height):
        for x in range(repeat_width, width):
            depth = depth_map[y, x] / 255.0
            offset = int(max_offset * depth)

            if x - offset >= 0:
                autostereogram[y, x] = autostereogram[y, x - offset]
            else:
                autostereogram[y, x] = autostereogram[y, x % repeat_width]

    return autostereogram

depth_map = cv2.imread("depthMap/final_leaf.png", cv2.IMREAD_GRAYSCALE)
depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
depth_map = cv2.equalizeHist(depth_map)

height, width = depth_map.shape
random_strip = np.random.randint(0, 256, (height, 20, 3), dtype=np.uint8)
background_image = np.tile(random_strip, (1, int(width / 20) + 1, 1))[:, :width, :]

autostereogram = generate_autostereogram(background_image, depth_map)

output_folder = "autostereogram"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
save_path = os.path.join(output_folder, "auto_autostereogram.png")

cv2.imshow("Autostereogram", autostereogram)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(save_path, autostereogram)
