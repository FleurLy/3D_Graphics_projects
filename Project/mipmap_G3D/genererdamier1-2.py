from PIL import Image
import numpy as np
import os

size, tile = 512, 2
dam = np.zeros((size, size, 3), dtype=np.uint8)
for i in range(size):
    for j in range(size):
        if (i // tile + j // tile) % 2 == 0:
            dam[i, j] = 255

script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "renderer", "texture", "damier.png")

Image.fromarray(dam).save(output_path)

output_path = os.path.join(script_dir, "renderer", "texture", "damier2.png")

Image.fromarray(dam).save(output_path)