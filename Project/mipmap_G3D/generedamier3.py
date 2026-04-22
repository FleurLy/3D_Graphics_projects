from PIL import Image, ImageFilter
import numpy as np
import os

palette = np.array([
    (160,  82,  45),
    ( 85, 107,  47),
    (130, 130, 130),
    (101,  67,  33),
], dtype=np.int16)

size, nb, joint = 1024, 32, 4
tw = size // nb

idx = np.random.randint(0, len(palette), size=(nb, nb))
tile_colors = palette[idx]

img = np.repeat(np.repeat(tile_colors, tw, axis=0), tw, axis=1).astype(np.uint8)

half = joint // 2
for k in range(nb):
    img[k * tw : k * tw + half, :] = 255
    img[(k + 1) * tw - half : (k + 1) * tw, :] = 255
    img[:, k * tw : k * tw + half] = 255
    img[:, (k + 1) * tw - half : (k + 1) * tw] = 255

pil = Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=0.5))

script_dir = os.path.dirname(os.path.abspath(__file__))
pil.save(os.path.join(script_dir, "renderer", "texture", "damier3.png"))
