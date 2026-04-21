# Software 3D Renderer — Python (Mipmapping & Anisotropic Filtering)

Renderer 3D complet implémenté en Python pur (NumPy) avec accélération C (ctypes).  
Simule le pipeline graphique d'un GPU en logiciel.  
**Projet ENSIMAG** — Amélioration du pipeline de rendu par **texture mipmapping** et **filtrage anisotropique**.

---

## Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Structure du projet](#structure-du-projet)
3. [Pipeline graphique](#pipeline-graphique)
4. [Méthode implémentée : Mipmapping + Filtrage anisotropique](#méthode-implémentée)
5. [Filtres disponibles](#filtres-disponibles)
6. [Calcul du LOD](#calcul-du-lod)
7. [Installation et lancement](#installation-et-lancement)
8. [Modes de rendu](#modes-de-rendu)
9. [Résultats](#résultats)
10. [Limitations et perspectives](#limitations-et-perspectives)
11. [Références](#références)

---

## Vue d'ensemble

Ce projet implémente un pipeline de rendu 3D logiciel complet en Python/NumPy, incluant :

- **Vertex Shader** : transformation monde → NDC avec calcul d'éclairage Phong
- **Rasterizer** : back-face culling, test pixel-dans-triangle, interpolation perspective-correcte
- **Fragment Shader** : éclairage de Phong, toon shading, filtrage de texture configurable
- **Depth Test** : z-buffer pour la gestion des occultations

L'amélioration principale porte sur le **filtrage de texture avec mipmapping**, qui résout le problème d'**aliasing** (scintillement) lors de l'affichage de surfaces éloignées ou obliques.

---

## Structure du projet

```
renderer/
├── main.py               → Point d'entrée principal + modes de comparaison
├── main2.py              → Point d'entrée refactorisé (scènes multiples)
├── camera.py             → View Matrix (monde → caméra)
├── projection.py         → Projection Matrix (perspective)
├── graphicPipeline.py    → Pipeline complet (VS, Rasterizer, FS, Depth Test)
├── mipmap.py             → Pyramide MIP + tous les filtres de texture
├── readply.py            → Lecteur fichiers PLY ASCII
├── convCPyth.py          → Chargement des bibliothèques C via ctypes
├── c-so/                 → Bibliothèques C compilées (.so / .dll)
│   ├── box.so
│   ├── gaussian_downsample.so
│   ├── lanczos_kernel_vals.so
│   └── lanczos_downsample.so
├── ply/                  → Meshes 3D (.ply)
│   ├── suzanne.ply
│   └── damier.ply
├── texture/              → Textures PNG
│   ├── suzanne.png
│   └── damier.png
└── output/               → Images générées
```

---

## Pipeline graphique

```
Sommets .ply (x, y, z, nx, ny, nz, u, v)
        │
        ▼
┌──────────────────┐
│  build_mipmaps() │  Pyramide MIP pré-calculée (une seule fois)
│  mipmap.py       │  Filtres de downsampling : box | gaussian | lanczos
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Vertex Shader   │  monde → NDC (clip space)
│                  │  Calcul N, V, L par sommet
│                  │  Stockage de w pour interpolation perspective-correcte
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Rasterizer     │  Back-face culling (produit vectoriel 2D)
│                  │  AABB + test pixel-dans-triangle (coordonnées barycentriques)
│                  │  Interpolation perspective-correcte (division par w)
│                  │  LOD précis : 4 dérivées partielles (dU/dx, dV/dx, dU/dy, dV/dy)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Fragment Shader  │  Phong (ambiant + diffus + spéculaire)
│                  │  Toon shading (quantification en niveaux discrets)
│                  │  Filtre de texture configurable :
│                  │    nearest | bilinear | trilinear | anisotropic
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Depth Test     │  Z-buffer → ne garde que le fragment le plus proche
└────────┬─────────┘
         │
         ▼
    Image finale (H × W × 3) float [0,1]
```

---

## Méthode implémentée

### Mipmapping

Le **mipmapping** (Williams, 1983) consiste à pré-calculer une texture en plusieurs résolutions décroissantes (pyramide MIP). Lors du rendu, le niveau approprié est sélectionné selon la distance et l'angle d'incidence de la surface.

**Problème résolu** : sans mipmapping, plusieurs texels se projettent sur un même pixel causant de l'**aliasing** (scintillement, moiré).

### Filtrage anisotropique

Le **filtrage anisotropique** améliore le rendu des surfaces obliques en échantillonnant le footprint elliptique réel de la projection, au lieu du footprint carré du filtrage isotropique. L'implémentation utilise une accumulation de `max_samples` échantillons le long de l'axe de distorsion maximale.

---

## Filtres disponibles

### Filtres de downsampling (construction de la pyramide)

| Filtre | Description | Qualité | Coût |
|--------|-------------|---------|------|
| `box` | Moyenne uniforme 2×2 — standard, rapide | ★★ | ★ |
| `gaussian` | Noyau gaussien séparable 4 taps (σ≈0.85) | ★★★ | ★★ |
| `lanczos` | Filtre sinc fenêtré (ordre 2) — plus net | ★★★★ | ★★★ |

> Les filtres de downsampling sont accélérés par des bibliothèques C compilées (ctypes).

### Filtres de sampling (interpolation lors du rendu)

| Filtre | Description | Qualité | Coût |
|--------|-------------|---------|------|
| `nearest` | Nearest-neighbour — référence/baseline | ★ | ★ |
| `bilinear` | Interpolation 4 voisins sur un seul niveau MIP | ★★ | ★★ |
| `trilinear` | Bilinéaire ×2 niveaux + lerp — standard OpenGL | ★★★ | ★★★ |
| `anisotropic` | Multi-tap directionnel — meilleur sur surfaces obliques | ★★★★ | ★★★★ |

---

## Calcul du LOD

Formule précise selon la spécification OpenGL (EXT_texture_lod) :

```
ρ   = max( ||∂UV/∂x||, ||∂UV/∂y|| ) × texture_size
LOD = log₂(ρ)

avec  ||∂UV/∂x|| = √( (∂u/∂x)² + (∂v/∂x)² )
      ||∂UV/∂y|| = √( (∂u/∂y)² + (∂v/∂y)² )
```

Les 4 dérivées partielles sont calculées par résolution d'un système linéaire 2×2 à partir des coordonnées écran et UV des 3 sommets du triangle.

---

## Installation et lancement

### Dépendances Python

```bash
pip install numpy Pillow matplotlib
```

### Compilation des bibliothèques C (Linux/macOS)

```bash
cd renderer/c-so
make
```

### Lancement

```bash
cd renderer/
python main.py
```

---

## Modes de rendu

Configurez la variable `MODE` dans `main.py` :

| Mode | Description |
|------|-------------|
| `"single"` | Rendu avec un seul filtre (défini par `FILTER_MODE`) |
| `"compare"` | Grille comparant les 4 filtres + comparaison downsampling |
| `"mipmap_vis"` | Visualisation de la pyramide MIP (atlas + courbe) |

---

## Résultats

| Méthode | Anti-aliasing | Flou | Surfaces obliques | Temps relatif |
|---------|:---:|:---:|:---:|:---:|
| `nearest` (sans mipmap) | ✗ Aliasing fort | ✗ Pixelisé | ✗ | 1× |
| `bilinear` | ✗ Aliasing lointain | ✓ | ✗ | ~1.2× |
| `trilinear` (standard) | ✓ | ✓ | △ Légèrement flou | ~1.5× |
| `anisotropic` | ✓ | ✓ | ✓ Détails préservés | ~3–5× |

---

## Limitations et perspectives

### Limitations actuelles

- **Filtrage anisotropique simplifié** : utilise un nombre fixe d'échantillons sans pondération par noyau. Le filtrage EWA (Elliptical Weighted Average) serait plus rigoureux.
- **LOD constant par triangle** : calculé depuis les sommets, pas par fragment. Peut causer des artefacts sur les grands triangles.
- **Rendu séquentiel** : aucun parallélisme. Une vectorisation NumPy ou une implémentation GPU apporterait des gains majeurs.
- **Pas d'antialiasing géométrique** : les bords des polygones restent crénelés (pas de MSAA).

### Perspectives

- **EWA filtering** [Heckbert 1989] : modélisation exacte de l'ellipse de projection
- **MSAA** : antialiasing des bords de polygones
- **Vectorisation NumPy** : rasterisation de triangles entiers en opérations matricielles
- **Ray tracing** : remplacement du pipeline pour un éclairage globalement correct

---

## Références

1. **Williams, L.** (1983). *Pyramidal parametrics*. SIGGRAPH Computer Graphics, 17(3), 1–11.
2. **Heckbert, P.** (1989). *Fundamentals of Texture Mapping and Image Warping*. Master's thesis, UC Berkeley.
3. **OpenGL Specification** (2022). *EXT_texture_filter_anisotropic* & LOD formula.
4. **Pharr, M., Jakob, W., & Humphreys, G.** (2023). *Physically Based Rendering* (4th ed.). MIT Press.
5. **Akenine-Möller, T., Haines, E., & Hoffman, N.** (2018). *Real-Time Rendering* (4th ed.). CRC Press.
6. **Lanczos, C.** (1964). *Applied Analysis*. Prentice Hall.