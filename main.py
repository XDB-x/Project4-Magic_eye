import numpy as np
import matplotlib.pyplot as plt
import skimage, skimage.io, skimage.color, skimage.exposure, skimage.filters
import os
import cv2


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

def enhance_depthmap_contrast(depthmap):
    enhanced_depthmap = skimage.exposure.equalize_adapthist(depthmap, clip_limit=0.03)
    enhanced_depthmap = skimage.filters.gaussian(enhanced_depthmap, sigma=1)
    return enhanced_depthmap

def make_detailed_pattern(shape=(128, 128), levels=64):
    pattern = np.random.randint(0, levels, (*shape, 3)) / levels  
    pattern = skimage.filters.gaussian(pattern, sigma=0.5)  
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

try:
    img = skimage.io.imread('inputPhoto/chair.jpg')
    original_img = img.copy()  
    if img.shape[2] == 4:  
        img = img[:, :, :3]
    depthmap = skimage.color.rgb2gray(img)  
    enhanced_depthmap = enhance_depthmap_contrast(depthmap)
except FileNotFoundError:
    depthmap = None

output_folder = "depthMap"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
save_path = os.path.join(output_folder, "main_depthMap.png")
cv2.imwrite(save_path, depthmap)
cv2.imshow("depth map", depthmap)
cv2.waitKey(0)
cv2.destroyAllWindows()

if depthmap is not None:
    display(original_img, title='Original Image')
    display(enhanced_depthmap, title='Enhanced Depth Map', colorbar=True)
    pattern = make_detailed_pattern(shape=(128, 128))
    autostereogram = make_autostereogram(enhanced_depthmap, pattern, shift_amplitude=0.2)
    display(autostereogram, title='Autostereogram')

output_folder = "autostereogram"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
save_path = os.path.join(output_folder, "main_autostereogram.png")
cv2.imwrite(save_path, autostereogram)
cv2.imshow("autostereogram", autostereogram)
cv2.waitKey(0)
cv2.destroyAllWindows()
