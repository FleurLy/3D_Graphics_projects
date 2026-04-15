# Software 3D Renderer — Python

Renderer 3D complet implémenté en Python pur (NumPy).
Simule le pipeline graphique d'un GPU en logiciel.

---

## Structure du projet

```
renderer/
├── main.py             → Point d'entrée
├── camera.py           → View Matrix (monde → caméra)
├── projection.py       → Projection Matrix (perspective)
├── graphicPipeline.py  → Pipeline complet (VS, Rasterizer, FS)
├── mipmap.py           → Construction pyramide MIP + sampling trilinéaire
├── readply.py          → Lecteur fichiers PLY ASCII
├── suzanne.ply         → Mesh 3D (à fournir)
└── suzanne.png         → Texture (à fournir)
```

---

## Pipeline graphique

```
Sommets .ply (x, y, z, nx, ny, nz, u, v)
        │
        ▼
┌──────────────────┐
│  build_mipmaps() │  Pyramide MIP pré-calculée (une fois)
│  mipmap.py       │  256→128→64→32→16→...→1 px
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Vertex Shader   │  monde → NDC (clip space)
│                  │  calcul N, V, L par sommet
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Rasterizer     │  back-face culling
│                  │  AABB + test pixel-dans-triangle
│                  │  interpolation perspective-correcte
│                  │  calcul du LOD (dU/dx, dV/dy)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Fragment Shader  │  Phong (ambiant + diffus + spéculaire)
│                  │  Toon shading (quantification)
│                  │  sample_trilinear(mips, u, v, LOD)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Depth Test     │  dépthBuffer → ne garde que le plus proche
└────────┬─────────┘
         │
         ▼
    Image finale (H × W × 3) float [0,1]
```

---

## Mipmapping — explication

Le mipmapping résout le problème d'aliasing (scintillement)
sur les textures appliquées à des surfaces éloignées.

**Construction (build_mipmaps) :**
On crée une pyramide de versions de la texture, chaque niveau
étant 2× plus petit que le précédent via une moyenne 2×2 (box filter).

```
Niveau 0 : 256×256  ← texture originale
Niveau 1 : 128×128
Niveau 2 :  64×64
Niveau 3 :  32×32
...
Niveau N :   1×1    ← couleur moyenne de toute la texture
```

**Calcul du LOD (Rasterizer) :**
On mesure à quelle vitesse les UV varient par pixel :
```
LOD = log2( max(dU/dx, dV/dy) × taille_texture )
```
- Objet proche : faible variation UV → LOD ≈ 0 → texture pleine résolution
- Objet loin   : grande variation UV → LOD élevé → petit niveau MIP

**Filtrage trilinéaire (sample_trilinear) :**
On échantillonne les deux niveaux entiers adjacents (floor et ceil du LOD)
avec un filtrage bilinéaire sur chacun, puis on interpole entre les deux.
Cela évite les "bandes" visibles entre niveaux MIP.

---

## Format des données

**Sommet PLY (8 valeurs) :**
| Index | Contenu |
|-------|---------|
| 0-2 | position monde (x, y, z) |
| 3-5 | normale (nx, ny, nz) |
| 6-7 | UV texture (u, v) |

**Après Vertex Shader (16 valeurs) :**
| Index | Contenu |
|-------|---------|
| 0-2 | position NDC |
| 3-5 | normale monde |
| 6-8 | vecteur V (vue) |
| 9-11 | vecteur L (lumière) |
| 12-13 | UV |
| 14-15 | NDC (x, y) pour LOD |

---

## Dépendances

```bash
pip install numpy Pillow matplotlib
```

## Lancement

```bash
python main.py
```

---

## Corrections et améliorations vs code initial

| Fichier | Problème original | Correction |
|---------|-------------------|------------|
| `readply.py` | Crash Windows (`\r` non supprimé) | `.strip()` global |
| `readply.py` | Quads non triangulés | Fan triangulation |
| `graphicPipeline.py` | UV inversés dans sample() | Axes corrigés |
| `graphicPipeline.py` | Nearest-neighbour (pixélisé) | Filtrage bilinéaire |
| `graphicPipeline.py` | Interpolation non perspective-correcte | Pondération par 1/z |
| `graphicPipeline.py` | Commentaire MIP non implémenté | Mipmapping trilinéaire complet |
| `graphicPipeline.py` | Division par zéro possible | Guards `< 1e-8` |
| `camera.py` | Vecteurs non normalisés | Normalisation dans `__init__` |
| **mipmap.py** | Fichier inexistant | **Nouveau fichier dédié** |
