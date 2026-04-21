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



# scene setup

def make_projection(width, height, near=0.1, far=100.0, fov=1.91986):
    return Projection(near, far, fov, width / height)


def load_scene(ply_path, texture_path, cam, cam_position, proj, light_position):
    vertices, triangles = readply(ply_path)
    print(f"[INFO] Mesh : {vertices.shape[0]} sommets, {triangles.shape[0]} triangles")

    texture = np.asarray(Image.open(texture_path).convert('RGB'))
    print(f"[INFO] Texture : {texture.shape[1]}x{texture.shape[0]} px")

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
    print(f"[SAVE] {out_path}")



# healpers hierarchie

_ANISOTROPIC_FILTERS = {"anisotropic"}


def _filter_dir(base, image_name, filter_mode, downsample_filter=None):
    """
    Construit le chemin de stockage selon la hierarchie :
      base/{image_name}/anisotropic/{downsample_filter}/
      base/{image_name}/isotropic/{filter_mode}/{downsample_filter}/
    """
    parts = [base, image_name]
    if filter_mode in _ANISOTROPIC_FILTERS:
        parts.append("anisotropic")
    else:
        parts += ["isotropic", filter_mode]
    if downsample_filter:
        parts.append(downsample_filter)
    return os.path.join(*parts)


def _comparison_dir(base, image_name):
    return os.path.join(base, image_name, "comparison")


def _mipmap_dir(base, image_name, downsample_filter):
    return os.path.join(base, image_name, "mipmap", downsample_filter)



# modes : mode single ou  mode comparaison ou visualisation pyramide MIP

def mode_single(vertices, triangles, data, width, height,
                filter_mode, downsample_filter,
                image_name="", output_dir=None, save=True, show=False):
    print(f"\n=== Rendu : {filter_mode} / {downsample_filter} ===")
    img, dt = render_with_filter(vertices, triangles, data, width, height,
                                  filter_mode, downsample_filter)
    print(f"[INFO] Rendu termine en {dt:.2f}s")

    if save and output_dir:
        d = _filter_dir(output_dir, image_name, filter_mode, downsample_filter)
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
                  image_name="", output_dir=None, save=True, show=False):
    if sampling_filters is None:
        sampling_filters = ["nearest", "bilinear", "trilinear", "anisotropic"]
    if downsample_filters is None:
        downsample_filters = ["box", "gaussian", "lanczos", "median"]

    results = {}; times = {}
    for fm in sampling_filters:
        print(f"\n=== Rendu : {fm} ===")
        img, dt = render_with_filter(vertices, triangles, data, width, height,
                                      fm, downsample_filter)
        results[fm] = img; times[fm] = dt
        if save and output_dir:
            d = _filter_dir(output_dir, image_name, fm, downsample_filter)
            save_image(img, d, "render.png")

        if fm in _ANISOTROPIC_FILTERS and save and output_dir:
            for dsf in downsample_filters:
                if dsf == downsample_filter:
                    continue
                print(f"  -> {fm} / downsampling={dsf}")
                img_dsf, _ = render_with_filter(vertices, triangles, data, width, height,
                                                 fm, dsf)
                d = _filter_dir(output_dir, image_name, fm, dsf)
                save_image(img_dsf, d, "render.png")

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle(
        f"{image_name}  —  Comparaison filtres de texture (downsample={downsample_filter})",
        fontsize=14, fontweight='bold')
    for ax, fm in zip(axes.flat, sampling_filters):
        ax.imshow(results[fm])
        ax.set_title(f"{fm.capitalize()}  ({times[fm]:.1f}s)", fontsize=11)
        ax.axis('off')
    plt.tight_layout()
    if save and output_dir:
        d = _comparison_dir(output_dir, image_name)
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, "filters.png"), dpi=150, bbox_inches='tight')
        print(f"[SAVE] {os.path.join(d, 'filters.png')}")
    if show:
        plt.show()

    print("\n=== Comparaison des filtres de downsampling ===")
    ds_results = {}; ds_times = {}
    for dsf in downsample_filters:
        print(f"  -> downsampling={dsf}")
        img, dt = render_with_filter(vertices, triangles, data, width, height,
                                      "trilinear", dsf)
        ds_results[dsf] = img; ds_times[dsf] = dt
        if save and output_dir:
            d = _filter_dir(output_dir, image_name, "trilinear", dsf)
            save_image(img, d, "render.png")

    fig2, axes2 = plt.subplots(1, len(downsample_filters), figsize=(6 * len(downsample_filters), 5))
    fig2.suptitle(
        f"{image_name}  —  Comparaison filtres de downsampling (sampling=trilinear)",
        fontsize=13, fontweight='bold')
    for ax, dsf in zip(axes2, downsample_filters):
        ax.imshow(ds_results[dsf])
        ax.set_title(f"Downsample : {dsf}  ({ds_times[dsf]:.1f}s)", fontsize=11)
        ax.axis('off')
    plt.tight_layout()
    if save and output_dir:
        d = _comparison_dir(output_dir, image_name)
        os.makedirs(d, exist_ok=True)
        fig2.savefig(os.path.join(d, "downsample.png"), dpi=150, bbox_inches='tight')
        print(f"[SAVE] {os.path.join(d, 'downsample.png')}")
    if show:
        plt.show()


def mode_mipmap_vis(texture, downsample_filter="box",
                     image_name="", output_dir=None, save=True, show=False):
    print("\n=== Visualisation de la pyramide MIP ===")
    mips  = build_mipmaps(texture, downsample_filter)
    atlas = mipmap_atlas(mips)

    print(f"[INFO] {len(mips)} niveaux MIP :")
    for i, m in enumerate(mips):
        print(f"  Level {i:2d} : {m.shape[1]:4d} x {m.shape[0]:4d} px")

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
    if save and output_dir:
        d = _mipmap_dir(output_dir, image_name, downsample_filter)
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, "pyramid.png"), dpi=150, bbox_inches='tight')
        print(f"[SAVE] {os.path.join(d, 'pyramid.png')}")
    if show:
        plt.show()



# MAIN

def main(
    mode="compare",
    filter_mode="trilinear",
    downsample_filter="box",
    width=512,
    height=288,
    ply_path=None,
    texture_path=None,
    camera_preset="damier",
    light_position=None,
    output_dir="output",
    save_images=True,
    show_plots=False,
):
    _dir = os.path.dirname(os.path.abspath(__file__))

    if ply_path is None:
        ply_path = os.path.join(_dir, "ply", "damier.ply")
    if texture_path is None:
        texture_path = os.path.join(_dir, "texture", "damier.png")
    if light_position is None:
        light_position = np.array([10.0, 0.0, 10.0])

    out_dir = os.path.join(_dir, output_dir) if save_images else None

    cameras = {"damier": make_camera_damier, "suzanne": make_camera_suzanne}
    if camera_preset not in cameras:
        raise ValueError(f"Camera preset inconnu : {camera_preset!r}")
    cam, cam_pos = cameras[camera_preset]()

    proj = make_projection(width, height)
    vertices, triangles, texture, data = load_scene(
        ply_path, texture_path, cam, cam_pos, proj, light_position
    )

    image_name = os.path.splitext(os.path.basename(texture_path))[0]

    modes = {
        "single"     : lambda: mode_single(vertices, triangles, data, width, height,
                                            filter_mode, downsample_filter,
                                            image_name=image_name,
                                            output_dir=out_dir, save=save_images,
                                            show=show_plots),
        "compare"    : lambda: mode_compare(vertices, triangles, data, width, height,
                                             downsample_filter=downsample_filter,
                                             image_name=image_name,
                                             output_dir=out_dir, save=save_images,
                                             show=show_plots),
        "mipmap_vis" : lambda: mode_mipmap_vis(texture, downsample_filter,
                                                image_name=image_name,
                                                output_dir=out_dir, save=save_images,
                                                show=show_plots),
    }
    if mode not in modes:
        raise ValueError(f"Mode inconnu : {mode!r}. Valeurs possibles : {list(modes)}")
    modes[mode]()


if __name__ == "__main__":
    main(
    mode="compare",
    filter_mode="trilinear",
    downsample_filter="box",
    width=512,
    height=288,
    ply_path=None,
    texture_path=None,
    camera_preset="damier",
    light_position=None,
    output_dir="output",
    save_images=True,
    show_plots=False,
    )
