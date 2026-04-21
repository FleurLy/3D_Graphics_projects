import os
import sys
import numpy as np


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convCPyth

# ============================================================
#  mipmap.py  —  Pyramide MIP + filtres de texture
#
#  Filtres de sampling (interpolation) :
#         - sample_nearest   : nearest-neighbour (référence / baseline)
#         - sample_bilinear  : interpolation bilinéaire 4 voisins
#         - sample_trilinear : bilinéaire × 2 niveaux + lerp (standard)
#         - sample_aniso     : anisotropique simplifié (multi-tap)
#
#  Principe du mipmapping :
#    - On précalcule N versions de la texture, chaque niveau
#      étant 2× plus petit que le précédent.
#    - Lors du rendu, on choisit le niveau dont la résolution
#      correspond à la surface couverte dans l'image.
#    - Cela évite l'aliasing (scintillement) et améliore la qualité.
#
#  Calcul du LOD (méthode précise OpenGL) :
#    rho = max( sqrt(dU/dx²+dV/dx²), sqrt(dU/dy²+dV/dy²) )
#    L   = log2( rho × texture_size )
# ============================================================


# ──────────────────────────────────────────────────────────────
#  HANDLES CTYPES (chargés une seule fois à l'import)
# ──────────────────────────────────────────────────────────────

_box_fn         = convCPyth.boxPyth()
_gaussian_fn    = convCPyth.gaussianPyth()
_lkernel_fn     = convCPyth.lanczosKernelPyth()
_lanczos_fn     = convCPyth.lanczosPyth()
_med_fn         = convCPyth.medPyth()


# ──────────────────────────────────────────────────────────────
#  FILTRES DE DOWNSAMPLING  (construction de la pyramide)
# ──────────────────────────────────────────────────────────────

def _box_downsample(img):
    """
    Box filter : moyenne uniforme de blocs 2x2.
    Filtre standard pour les mipmaps : rapide et correct.
    """
    H, W = img.shape[:2]
    img_c = np.ascontiguousarray(img.astype(np.float32))
    out   = np.empty((H // 2, W // 2, 3), dtype=np.float32)
    out_c = np.ascontiguousarray(out)
    _box_fn(img_c, H, W, out_c)
    return out_c


def _gaussian_downsample(img):
    """
    Filtre gaussien 4 taps (sigma ~0.85) + sous-echantillonnage x2.

    Le filtre gaussien attenue mieux les hautes frequences que le
    box filter, ce qui reduit l'aliasing residuel dans la pyramide.
    Le noyau separe 1D est applique horizontalement puis verticalement.
    """
    H, W = img.shape[:2]
    img_c = np.ascontiguousarray(img.astype(np.float32))
    out   = np.empty((H // 2, W // 2, 3), dtype=np.float32)
    out_c = np.ascontiguousarray(out)
    _gaussian_fn(img_c, H, W, out_c)
    return out_c


def _lanczos_kernel_vals(x, a=2):
    """Noyau de Lanczos : sinc(x) * sinc(x/a), |x| < a."""
    x_f32 = np.ascontiguousarray(np.asarray(x, dtype=np.float32).ravel())
    result = np.empty(x_f32.size, dtype=np.float32)
    res_c  = np.ascontiguousarray(result)
    _lkernel_fn(x_f32, x_f32.size, int(a), res_c)
    return res_c


def _lanczos_downsample(img, a=2):
    """
    Filtre de Lanczos (ordre a=2) + sous-echantillonnage x2.

    Chaque pixel de sortie est la somme ponderee par le noyau de
    Lanczos de 2a x 2a pixels de l'image source. Produit des
    mipmaps plus nets que le box filter, au prix d'un cout plus eleve.
    """
    H, W = img.shape[:2]
    img_c = np.ascontiguousarray(img.astype(np.float32))
    out   = np.empty((H // 2, W // 2, 3), dtype=np.float32)
    out_c = np.ascontiguousarray(out)
    _lanczos_fn(img_c, H, W, int(a), out_c)
    return out_c


def _med_downsample(img):
    """Filtre mediane 2x2 : mediane des 4 voisins par canal, RGB float32."""
    H, W = img.shape[:2]
    img_c = np.ascontiguousarray(img.astype(np.float32))
    out   = np.empty((H // 2, W // 2, 3), dtype=np.float32)
    out_c = np.ascontiguousarray(out)
    _med_fn(img_c, H, W, out_c)
    return out_c


# ──────────────────────────────────────────────────────────────
#  CONSTRUCTION DE LA PYRAMIDE MIP
# ──────────────────────────────────────────────────────────────

DOWNSAMPLE_METHODS = {
    "box":      _box_downsample,
    "gaussian": _gaussian_downsample,
    "lanczos":  _lanczos_downsample,
    "median":   _med_downsample,
}


def _downsample_1d(texture, downsample_filter="box"):
    """
    Downsample a 1D texture stored as (1, W, 3) or (H, 1, 3).

    This path is only used when the original texture is already 1D.
    Regular 2D textures keep the existing behavior and stop once one axis
    reaches 1.
    """
    H, W = texture.shape[:2]
    if H > 1 and W > 1:
        raise ValueError("_downsample_1d expects a 1D texture")
    if H == 1 and W == 1:
        return texture.copy()

    horizontal = H == 1
    size = W if horizontal else H
    out_size = max(1, size // 2)
    out_shape = (1, out_size, 3) if horizontal else (out_size, 1, 3)
    out = np.empty(out_shape, dtype=np.float32)

    if downsample_filter == "box":
        for i in range(out_size):
            start = 2 * i
            end = min(start + 2, size)
            if horizontal:
                out[0, i] = np.mean(texture[0, start:end], axis=0)
            else:
                out[i, 0] = np.mean(texture[start:end, 0], axis=0)
        return out

    if downsample_filter == "gaussian":
        kernel = np.array([0.0625, 0.4375, 0.4375, 0.0625], dtype=np.float32)
        offsets = (-1, 0, 1, 2)
        for i in range(out_size):
            center = 2 * i
            acc = np.zeros(3, dtype=np.float32)
            weight_sum = 0.0
            for weight, offset in zip(kernel, offsets):
                idx = center + offset
                if 0 <= idx < size:
                    sample = texture[0, idx] if horizontal else texture[idx, 0]
                    acc += weight * sample
                    weight_sum += weight
            if horizontal:
                out[0, i] = acc / weight_sum
            else:
                out[i, 0] = acc / weight_sum
        return out

    if downsample_filter == "lanczos":
        a = 2
        for i in range(out_size):
            center = 2.0 * i + 0.5
            start = max(0, int(np.floor(center - a + 1)))
            end = min(size, int(np.ceil(center + a)))
            positions = np.arange(start, end, dtype=np.float32)
            weights = _lanczos_kernel_vals(positions - center, a=a)
            weight_sum = float(np.sum(weights))
            acc = np.zeros(3, dtype=np.float32)
            if weight_sum > 1e-8:
                norm = weights / weight_sum
                for weight, idx in zip(norm, positions.astype(int)):
                    sample = texture[0, idx] if horizontal else texture[idx, 0]
                    acc += weight * sample
            if horizontal:
                out[0, i] = np.clip(acc, 0.0, 1.0)
            else:
                out[i, 0] = np.clip(acc, 0.0, 1.0)
        return out

    raise ValueError(f"Unknown downsample filter: {downsample_filter}")


def build_mipmaps(texture, downsample_filter="box"):
    """
    Construit la pyramide MIP complete a partir de la texture originale.

    Parametres :
      texture          : np.ndarray (H, W, 3) uint8  [0..255]
      downsample_filter: "box" | "gaussian" | "lanczos"
                         Filtre utilise pour calculer chaque niveau.

    Retourne :
      list de np.ndarray (Hi, Wi, 3) float32 [0.0..1.0]
        mips[0] = texture originale normalisee
        mips[1] = moitie de taille
        ...
        mips[N] = 1x1 pixel (couleur moyenne de toute la texture)
    """
    fn      = DOWNSAMPLE_METHODS.get(downsample_filter, _box_downsample)
    current = texture.astype(np.float32) / 255.0
    mips    = [current]
    original_is_1d = current.shape[0] == 1 or current.shape[1] == 1

    if original_is_1d:
        while current.shape[0] > 1 or current.shape[1] > 1:
            down = _downsample_1d(current, downsample_filter)
            mips.append(down)
            current = down
    else:
        while current.shape[0] > 1 and current.shape[1] > 1:
            down    = fn(current)
            mips.append(down)
            current = down

    return mips


# ──────────────────────────────────────────────────────────────
#  FILTRES DE SAMPLING  (interpolation lors du rendu)
# ──────────────────────────────────────────────────────────────

def sample_nearest(mip_level, u, v):
    """
    Nearest-neighbour : texel le plus proche, sans interpolation.

    Methode de reference (baseline). Produit un effet pixelise
    de pres et du scintillement au loin. Tres rapide.
    """
    u = u % 1.0; v = v % 1.0
    H, W = mip_level.shape[:2]
    tx = int(round(u * (W - 1)))
    ty = int(round((1.0 - v) * (H - 1)))
    return mip_level[np.clip(ty, 0, H-1), np.clip(tx, 0, W-1)]


def sample_bilinear(mip_level, u, v):
    """
    Filtrage bilineaire : interpolation ponderee des 4 texels voisins.

    Evite l'effet pixelise du nearest-neighbour mais peut etre flou
    si le LOD n'est pas adapte. Bon compromis vitesse/qualite.
    """
    u = u % 1.0; v = v % 1.0
    H, W = mip_level.shape[:2]

    tx = u * (W - 1);       ty = (1.0 - v) * (H - 1)
    x0 = int(tx);           x1 = min(x0 + 1, W - 1)
    y0 = int(ty);           y1 = min(y0 + 1, H - 1)
    fx = tx - x0;           fy = ty - y0

    c00 = mip_level[y0, x0]; c10 = mip_level[y0, x1]
    c01 = mip_level[y1, x0]; c11 = mip_level[y1, x1]

    top    = c00 * (1.0 - fx) + c10 * fx
    bottom = c01 * (1.0 - fx) + c11 * fx
    return top * (1.0 - fy) + bottom * fy


def sample_trilinear(mips, u, v, lod):
    """
    Filtrage trilineaire : bilineaire sur deux niveaux MIP adjacents
    + interpolation lineaire entre les deux.

    Evite les transitions brusques entre niveaux MIP (bandes visibles
    avec un LOD entier seulement). C'est le standard OpenGL/Vulkan.

    Parametres :
      mips : pyramide MIP (build_mipmaps)
      u, v : UV dans [0, 1]
      lod  : level of detail flottant >= 0

    Retourne : couleur (3,) float [0,1]
    """
    max_level = len(mips) - 1
    lod = max(0.0, min(lod, float(max_level)))

    l0   = int(np.floor(lod))
    l1   = min(l0 + 1, max_level)
    frac = lod - l0

    c0 = sample_bilinear(mips[l0], u, v)
    if frac < 1e-6 or l0 == l1:
        return c0

    c1 = sample_bilinear(mips[l1], u, v)
    return c0 * (1.0 - frac) + c1 * frac


def sample_anisotropic(mips, u, v, lod, dudx, dvdx, dudy, dvdy,
                        max_samples=32):
    """
    Filtrage anisotropique simplifie (approximation EWA multi-tap).

    Le filtrage trilineaire est isotropique : il applique le meme flou
    dans toutes les directions. Sur une surface vue en oblique, la
    texture est sur-filtree dans un axe (floue) ou aliasee dans l'autre.

    L'approche anisotropique :
      1. Calcule deux LODs directionnels (selon x et y ecran).
      2. Prend le min comme niveau de base (preserve la nettete).
      3. Accumule plusieurs samples le long de l'axe le plus etire.

    Parametres :
      mips             : pyramide MIP
      u, v             : UV dans [0, 1]
      lod              : LOD isotropique de reference
      dudx,dvdx        : derivees partielles de UV selon x
      dudy,dvdy        : derivees partielles de UV selon y
      max_samples      : nombre max de taps (qualite / cout)

    Retourne : couleur (3,) float [0,1]
    """
    max_level = len(mips) - 1
    tex_size  = max(mips[0].shape[0], mips[0].shape[1])

    mag_x = np.sqrt(dudx**2 + dvdx**2) * tex_size + 1e-10
    mag_y = np.sqrt(dudy**2 + dvdy**2) * tex_size + 1e-10

    lod_min = max(0.0, min(np.log2(min(mag_x, mag_y)), float(max_level)))
    ratio   = max(mag_x, mag_y) / min(mag_x, mag_y)
    n_samp  = int(np.clip(round(ratio), 1, max_samples))

    step_count = max(n_samp, 1)
    if mag_x >= mag_y:
        step_u = dudx / step_count
        step_v = dvdx / step_count
    else:
        step_u = dudy / step_count
        step_v = dvdy / step_count

    color = np.zeros(3, dtype=np.float32)
    u_s   = u - step_u * (n_samp - 1) / 2.0
    v_s   = v - step_v * (n_samp - 1) / 2.0

    l0 = int(np.floor(lod_min))
    for _ in range(n_samp):
        color += sample_bilinear(mips[l0], u_s, v_s)
        u_s   += step_u
        v_s   += step_v

    return color / n_samp


# ──────────────────────────────────────────────────────────────
#  LOD PRECIS  (formule OpenGL / EXT_texture_lod)
# ──────────────────────────────────────────────────────────────

def compute_lod_accurate(dudx, dvdx, dudy, dvdy, tex_size):
    """
    Calcul precis du LOD selon la formule OpenGL.

    rho = max( ||dUV/dx||, ||dUV/dy|| ) x tex_size
    LOD = log2(rho)

    ou ||dUV/dx|| = sqrt(dudx^2 + dvdx^2)
       ||dUV/dy|| = sqrt(dudy^2 + dvdy^2)

    Plus correct que max(|dU/dx|, |dV/dy|) qui ignore
    les derivees croisees dV/dx et dU/dy.
    """
    rho_x = np.sqrt(dudx**2 + dvdx**2) * tex_size
    rho_y = np.sqrt(dudy**2 + dvdy**2) * tex_size
    return max(0.0, np.log2(max(rho_x, rho_y, 1e-10)))


# ──────────────────────────────────────────────────────────────
#  UTILITAIRE : atlas de visualisation de la pyramide MIP
# ──────────────────────────────────────────────────────────────

def mipmap_atlas(mips):
    """
    Assemble tous les niveaux MIP en une seule image horizontale.

    Les niveaux sont disposes du plus grand (gauche) au plus petit
    (droite), separes d'une marge de 2px noirs.
    Utile pour le rapport et le poster.

    Retourne : np.ndarray (H_max, W_total, 3) float32
    """
    H_max   = mips[0].shape[0]
    gap     = 2
    W_total = sum(m.shape[1] for m in mips) + gap * (len(mips) - 1)
    atlas   = np.zeros((H_max, W_total, 3), dtype=np.float32)
    x_off   = 0
    for m in mips:
        h, w = m.shape[:2]
        atlas[:h, x_off:x_off + w] = m
        x_off += w + gap
    return atlas


#