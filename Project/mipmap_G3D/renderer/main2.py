import numpy as np
import time
import os
import matplotlib.pyplot as plt
from PIL import Image

from camera          import Camera
from projection      import Projection
from graphicPipeline import GraphicPipeline
from readply         import readply
from mipmap          import build_mipmaps, mipmap_atlas

_dir = os.path.dirname(os.path.abspath(__file__))


# CAMERAS PRESETS

def make_camera_damier():
    position = np.array([0.0, 15.0, -10.0])
    lookAt   = np.array([0.0, -0.555, 0.832])
    up       = np.array([0.0,  0.832, 0.555])
    right    = np.array([1.0,  0.0,   0.0  ])
    return Camera(position, lookAt, up, right), position


def make_camera_suzanne():
    position = np.array([1.1, 1.1, 1.1])
    lookAt   = np.array([-0.577, -0.577, -0.577])
    up       = np.array([ 0.333,  0.333, -0.667])
    right    = np.array([-0.577,  0.577,  0.0  ])
    return Camera(position, lookAt, up, right), position

cameras = {"damier": make_camera_damier, "suzanne": make_camera_suzanne}


# scene setup

def make_projection(width, height, near=0.1, far=100.0, fov=1.91986):
    return Projection(near, far, fov, width / height)


def load_scene(name, cam, cam_position, proj, light_position):
    ply_path     = os.path.join(_dir, "ply",     f"{name}.ply")
    texture_path = os.path.join(_dir, "texture", f"{name}.png")
    vertices, triangles = readply(ply_path)
    texture = np.asarray(Image.open(texture_path).convert('RGB'))
    data = {
        'viewMatrix'    : cam.getMatrix(),
        'projMatrix'    : proj.getMatrix(),
        'cameraPosition': cam_position,
        'lightPosition' : light_position,
        'texture'       : texture,
    }
    return vertices, triangles, texture, data


# RENDU

def render_with_filter(vertices, triangles, data, width, height,
                        filter_mode, downsample_filter="box"):
    pipeline = GraphicPipeline(width, height,
                                filter_mode=filter_mode,
                                downsample_filter=downsample_filter)
    t0 = time.time()
    pipeline.draw(vertices, triangles, data)
    return pipeline.image, time.time() - t0


def save_image(img, directory, filename):
    os.makedirs(directory, exist_ok=True)
    out_path = os.path.join(directory, filename)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(out_path)


def _filter_dir(image_name, filter_mode, downsample_filter=None):
    parts = ["output", image_name]
    if filter_mode == "anisotropic":
        parts.append("anisotropic")
    else:
        parts += ["isotropic", filter_mode]
    if downsample_filter:
        parts.append(downsample_filter)
    return os.path.join(*parts)


def _comparison_dir(image_name):
    return os.path.join("output", image_name, "comparison")


def _mipmap_dir(image_name, downsample_filter):
    return os.path.join("output", image_name, "mipmap", downsample_filter)


# modes

def mode_single(vertices, triangles, data, width, height,
                filter_mode, downsample_filter,
                image_name="", save=True, show=False):
    img, dt = render_with_filter(vertices, triangles, data, width, height,
                                  filter_mode, downsample_filter)
    if save:
        d = _filter_dir(image_name, filter_mode, downsample_filter)
        save_image(img, d, "render.png")

    plt.figure(figsize=(10, 6))
    plt.title(f"{image_name}  |  Filtre : {filter_mode}  |  Downsample : {downsample_filter}  |  {dt:.1f}s")
    plt.imshow(img); plt.axis('off'); plt.tight_layout()
    if show:
        plt.show()


def mode_compare(vertices, triangles, data, width, height,
                  downsample_filter="box",
                  sampling_filters=None,
                  downsample_filters=None,
                  image_name="", save=True, show=False):
    if sampling_filters is None:
        sampling_filters = ["nearest", "bilinear", "trilinear", "anisotropic"]
    if downsample_filters is None:
        downsample_filters = ["box", "gaussian", "lanczos", "median"]

    results = {}; times = {}
    for fm in sampling_filters:
        img, dt = render_with_filter(vertices, triangles, data, width, height,
                                      fm, downsample_filter)
        results[fm] = img; times[fm] = dt
        if save:
            d = _filter_dir(image_name, fm, downsample_filter)
            save_image(img, d, "render.png")

        if fm == "anisotropic" and save:
            for dsf in downsample_filters:
                if dsf == downsample_filter:
                    continue
                img_dsf, _ = render_with_filter(vertices, triangles, data, width, height,
                                                 fm, dsf)
                save_image(img_dsf, _filter_dir(image_name, fm, dsf), "render.png")

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle(
        f"{image_name}  —  Comparaison filtres de texture (downsample={downsample_filter})",
        fontsize=14, fontweight='bold')
    for ax, fm in zip(axes.flat, sampling_filters):
        ax.imshow(results[fm])
        ax.set_title(f"{fm.capitalize()}  ({times[fm]:.1f}s)", fontsize=11)
        ax.axis('off')
    plt.tight_layout()
    if save:
        d = _comparison_dir(image_name)
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, "filters.png"), dpi=150, bbox_inches='tight')
    if show:
        plt.show()

    ds_results = {}; ds_times = {}
    for dsf in downsample_filters:
        img, dt = render_with_filter(vertices, triangles, data, width, height,
                                      "trilinear", dsf)
        ds_results[dsf] = img; ds_times[dsf] = dt
        if save:
            save_image(img, _filter_dir(image_name, "trilinear", dsf), "render.png")

    fig2, axes2 = plt.subplots(1, len(downsample_filters), figsize=(6 * len(downsample_filters), 5))
    fig2.suptitle(
        f"{image_name}  —  Comparaison filtres de downsampling (sampling=trilinear)",
        fontsize=13, fontweight='bold')
    for ax, dsf in zip(axes2, downsample_filters):
        ax.imshow(ds_results[dsf])
        ax.set_title(f"Downsample : {dsf}  ({ds_times[dsf]:.1f}s)", fontsize=11)
        ax.axis('off')
    plt.tight_layout()
    if save:
        d = _comparison_dir(image_name)
        os.makedirs(d, exist_ok=True)
        fig2.savefig(os.path.join(d, "downsample.png"), dpi=150, bbox_inches='tight')
    if show:
        plt.show()


def mode_mipmap_vis(texture, downsample_filter="box",
                     image_name="", save=True, show=False):
    mips  = build_mipmaps(texture, downsample_filter)
    atlas = mipmap_atlas(mips)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5),
                              gridspec_kw={'width_ratios': [3, 1]})
    fig.suptitle(f"{image_name}  —  Pyramide MIP  |  filtre : {downsample_filter}", fontsize=13)

    axes[0].imshow(atlas); axes[0].axis('off')
    axes[0].set_title("Atlas des niveaux MIP (L0 -> Lmax)")

    lvls   = list(range(len(mips)))
    pixels = [m.shape[0] * m.shape[1] for m in mips]
    axes[1].semilogy(lvls, pixels, 'o-', color='steelblue', linewidth=2)
    axes[1].set_xlabel("Niveau MIP"); axes[1].set_ylabel("Pixels (log)")
    axes[1].set_title("Taille par niveau"); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        d = _mipmap_dir(image_name, downsample_filter)
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, "pyramid.png"), dpi=150, bbox_inches='tight')
    if show:
        plt.show()


# main

def main(
    mode="compare",
    filter_mode="trilinear",
    downsample_filter="box",
    width=512,
    height=288,
    name="damier",
    light_position=None,
    save_images=True,
    show_plots=False,
):
    if light_position is None:
        light_position = np.array([10.0, 0.0, 10.0])

    if name not in cameras:
        raise ValueError(f"Nom inconnu : {name!r}. Valeurs possibles : {list(cameras)}")

    cam, cam_pos = cameras[name]()
    proj = make_projection(width, height)
    vertices, triangles, texture, data = load_scene(name, cam, cam_pos, proj, light_position)

    modes = {
        "single"     : lambda: mode_single(vertices, triangles, data, width, height,
                                            filter_mode, downsample_filter,
                                            image_name=name, save=save_images,
                                            show=show_plots),
        "compare"    : lambda: mode_compare(vertices, triangles, data, width, height,
                                             downsample_filter=downsample_filter,
                                             image_name=name, save=save_images,
                                             show=show_plots),
        "mipmap_vis" : lambda: mode_mipmap_vis(texture, downsample_filter,
                                                image_name=name, save=save_images,
                                                show=show_plots),
    }
    if mode not in modes:
        raise ValueError(f"Mode inconnu. Valeurs possibles : {list(modes)}")
    modes[mode]()


if __name__ == "__main__":
    main(
        mode="compare",
        filter_mode="trilinear",
        downsample_filter="box",
        width=512,
        height=288,
        name="damier",
        light_position=None,
        save_images=True,
        show_plots=False,
    )
