from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from camera import Camera
from graphicPipeline import GraphicPipeline
from projection import Projection
from readply import readply


# Keep the same camera setup as TP4 to compare results easily.
WIDTH = 960
HEIGHT = 540

CAMERA_POSITION = np.array([1.1, 1.1, 1.1])
LOOK_AT = np.array([-0.577, -0.577, -0.577])
UP = np.array([0.33333333, 0.33333333, -0.66666667])
RIGHT = np.array([-0.57735027, 0.57735027, 0.0])

NEAR_PLANE = 0.1
FAR_PLANE = 10.0
FOV = 1.91986

LIGHT_POSITION = np.array([10.0, 0.0, 10.0])


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent

    # Reuse TP4 assets directly.
    mesh_path = base_dir.parent / "TP4" / "TP4" / "suzanne.ply"
    texture_path = base_dir.parent / "TP4" / "TP4" / "suzanne.png"

    vertices, triangles = readply(str(mesh_path))

    # PNG in this TP has an alpha channel: keep only RGB.
    texture = mpimg.imread(texture_path)[:, :, :3].astype(np.float64)

    cam = Camera(CAMERA_POSITION, LOOK_AT, UP, RIGHT)
    proj = Projection(NEAR_PLANE, FAR_PLANE, FOV, WIDTH / HEIGHT)

    data = {
        "view_matrix": cam.get_matrix(),
        "proj_matrix": proj.get_matrix(),
        "camera_position": CAMERA_POSITION,
        "light_position": LIGHT_POSITION,
        "texture": texture,
    }

    output_dir = base_dir / "output_renders"
    output_dir.mkdir(exist_ok=True)

    chosen_mipmap = "med"
    render_mode = "trilinear"
    mipmap_algorithms = ["miNe", "moy", "med", "filtre"]

    # Deux passes : isotropique (max_anisotropy=1) et anisotropique (max_anisotropy=8)
    aniso_configs = [
        ("isotropic",   1),
        ("anisotropic", 8),
    ]

    for (label, max_aniso) in aniso_configs:
        mode_dir = output_dir / label
        mode_dir.mkdir(exist_ok=True)

        # Nearest et Bilinear (structure de base conservée dans chaque dossier)
        for basic_mode in ["nearest", "bilinear"]:
            pipeline = GraphicPipeline(WIDTH, HEIGHT, filter_mode=basic_mode,
                                       mipmap_mode=chosen_mipmap, max_anisotropy=max_aniso)
            pipeline.draw(vertices, triangles, data)
            final_img = np.clip(pipeline.image.copy(), 0.0, 1.0)
            output_file = mode_dir / f"render_{basic_mode}_{chosen_mipmap}.png"
            plt.imsave(output_file, final_img)
            print(f"Saved: {output_file}")

        # Trilinear x tous les algos de génération dans un sous-dossier
        trilinear_dir = mode_dir / "trilinear"
        trilinear_dir.mkdir(exist_ok=True)

        for algo in mipmap_algorithms:
            pipeline = GraphicPipeline(WIDTH, HEIGHT, filter_mode=render_mode,
                                       mipmap_mode=algo, max_anisotropy=max_aniso)
            pipeline.draw(vertices, triangles, data)
            final_img = np.clip(pipeline.image.copy(), 0.0, 1.0)
            output_file = trilinear_dir / f"render_{render_mode}_{algo}.png"
            plt.imsave(output_file, final_img)
            print(f"Saved: {output_file}")
