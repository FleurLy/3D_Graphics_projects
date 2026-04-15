import numpy as np

# ============================================================
#  mipmap.py
#  Construction de la pyramide MIP et sampling avec filtrage
#  trilinéaire (bilinéaire + interpolation entre niveaux).
#
#  Principe du mipmapping :
#    - On pré-calcule N versions de la texture, chaque niveau
#      étant 2× plus petit que le précédent.
#    - Lors du rendu, on choisit le niveau dont la résolution
#      correspond à la surface couverte dans l'image.
#    - Cela évite l'aliasing (scintillement) sur les surfaces
#      lointaines et améliore la qualité globale.
#
#  Calcul du niveau L :
#    On estime la variation des UV (dU/dx, dV/dy) dans l'espace
#    écran, puis :
#      L = log2( max(|dU/dx|, |dV/dy|) × texture_size )
#    Plus L est grand → objet loin → on utilise une petite texture.
#
#  Filtrage trilinéaire :
#    = bilinéaire sur le niveau floor(L)
#    + bilinéaire sur le niveau ceil(L)
#    + interpolation linéaire entre les deux résultats selon frac(L)
# ============================================================


def build_mipmaps(texture):
    """
    Construit la pyramide MIP à partir de la texture originale.

    Paramètre :
      texture : np.ndarray (H, W, 3) uint8  [0..255]

    Retourne :
      list de np.ndarray (Hi, Wi, 3) float32 [0.0..1.0]
        mips[0] = texture originale normalisée
        mips[1] = moitié de taille
        mips[2] = quart de taille
        ...
        mips[N] = 1×1 pixel (couleur moyenne de toute la texture)

    Méthode de downsampling : moyenne de blocs 2×2 (box filter).
    C'est le filtre standard pour les mipmaps : rapide et correct.
    """

    # Niveau 0 : texture originale convertie en float [0,1]
    current = texture.astype(np.float32) / 255.0
    mips    = [current]

    # On divise par 2 tant que la texture fait plus de 1×1
    while current.shape[0] > 1 and current.shape[1] > 1:

        H, W = current.shape[0], current.shape[1]

        # On s'assure que H et W sont pairs pour le regroupement 2×2.
        # Si la dimension est impaire, on coupe le dernier pixel.
        H_even = H - (H % 2)
        W_even = W - (W % 2)
        cropped = current[:H_even, :W_even]

        # Regroupement 2×2 → moyenne des 4 voisins.
        # reshape en blocs puis moyenne sur les axes des blocs.
        #
        # Avant reshape : (H_even, W_even, 3)
        # Après reshape  : (H_even//2, 2, W_even//2, 2, 3)
        # Moyenne sur axes 1 et 3 → (H_even//2, W_even//2, 3)
        H2, W2 = H_even // 2, W_even // 2
        down = (cropped
                .reshape(H2, 2, W2, 2, 3)
                .mean(axis=(1, 3))
                .astype(np.float32))

        mips.append(down)
        current = down

    return mips


def sample_bilinear(mip_level, u, v):
    """
    Échantillonne un niveau de mipmap donné avec filtrage BILINÉAIRE.

    Le filtrage bilinéaire mélange les 4 texels voisins autour
    du point d'échantillonnage, ce qui évite l'effet "pixélisé"
    du nearest-neighbour.

    Paramètres :
      mip_level : np.ndarray (H, W, 3) float [0,1]
      u, v      : coordonnées UV dans [0, 1]

    Retourne :
      np.ndarray (3,) couleur dans [0, 1]
    """

    H, W = mip_level.shape[0], mip_level.shape[1]

    # Wrapping : replie les UV hors [0,1] (texture répétée)
    u = u % 1.0
    v = v % 1.0

    # Conversion UV → position continue en texels
    # v est inversé car l'axe Y image descend vers le bas
    tx = u * (W - 1)
    ty = (1.0 - v) * (H - 1)

    # Indices entiers des 4 texels voisins
    x0 = int(tx)
    y0 = int(ty)
    x1 = min(x0 + 1, W - 1)   # clamp au bord droit
    y1 = min(y0 + 1, H - 1)   # clamp au bord bas

    # Fractions : distance par rapport au coin supérieur gauche
    fx = tx - x0   # fraction horizontale dans [0,1)
    fy = ty - y0   # fraction verticale   dans [0,1)

    # Les 4 texels voisins (déjà en float [0,1])
    c00 = mip_level[y0, x0]   # haut-gauche
    c10 = mip_level[y0, x1]   # haut-droit
    c01 = mip_level[y1, x0]   # bas-gauche
    c11 = mip_level[y1, x1]   # bas-droit

    # Interpolation bilinéaire en deux passes :
    #   1. Interpolation horizontale sur chaque ligne
    top    = c00 * (1.0 - fx) + c10 * fx   # ligne du haut
    bottom = c01 * (1.0 - fx) + c11 * fx   # ligne du bas
    #   2. Interpolation verticale entre les deux lignes
    return top * (1.0 - fy) + bottom * fy


def sample_trilinear(mips, u, v, lod):
    """
    Échantillonne la pyramide MIP avec filtrage TRILINÉAIRE.

    Le filtrage trilinéaire = bilinéaire sur deux niveaux adjacents
    + interpolation linéaire entre les deux résultats.

    Cela évite les transitions brusques entre niveaux (effet de
    "bandes" visibles avec un LOD entier seulement).

    Paramètres :
      mips : list de niveaux (construit par build_mipmaps)
      u, v : coordonnées UV dans [0, 1]
      lod  : level of detail (flottant, calculé par le rasterizer)
              lod = 0 → texture pleine résolution
              lod = 1 → moitié
              lod = N → 2^N fois plus petit

    Retourne :
      np.ndarray (3,) couleur dans [0, 1]
    """

    max_level = len(mips) - 1

    # Clamp du LOD dans [0, max_level]
    lod = max(0.0, min(lod, float(max_level)))

    # Niveau bas (entier inférieur) et haut (entier supérieur)
    l0 = int(np.floor(lod))
    l1 = min(l0 + 1, max_level)

    # Fraction entre les deux niveaux
    frac = lod - l0

    # Couleur au niveau bas (bilinéaire)
    c0 = sample_bilinear(mips[l0], u, v)

    if frac < 1e-6 or l0 == l1:
        # On est exactement sur un niveau entier : pas besoin
        # d'interpoler → économise un sample_bilinear
        return c0

    # Couleur au niveau haut (bilinéaire)
    c1 = sample_bilinear(mips[l1], u, v)

    # Interpolation linéaire entre les deux niveaux
    return c0 * (1.0 - frac) + c1 * frac
