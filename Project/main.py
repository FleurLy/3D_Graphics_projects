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

    render_modes = ["nearest", "bilinear", "trilinear"]
    rendered_images = {}

    for mode in render_modes:
        print(f"Rendering mode: {mode}")
        pipeline = GraphicPipeline(WIDTH, HEIGHT, filter_mode=mode)
        pipeline.draw(vertices, triangles, data)
        rendered_images[mode] = pipeline.image.copy()

        output_file = base_dir / f"render_{mode}.png"
        plt.imsave(output_file, pipeline.image)
        print(f"Saved: {output_file}")

    # One side-by-side image for the report/poster.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for index, mode in enumerate(render_modes):
        axes[index].imshow(rendered_images[mode])
        axes[index].set_title(mode)
        axes[index].axis("off")

    comparison_path = base_dir / "comparison_filters.png"
    fig.tight_layout()
    fig.savefig(comparison_path, dpi=200)
    plt.close(fig)

    print(f"Saved: {comparison_path}")
