import numpy as np
from mipmap import build_mipmaps, sample_trilinear

# ============================================================
#  graphicPipeline.py
#  Pipeline graphique 3D "software renderer" complet.
#
#  Étapes du pipeline :
#    1. Pré-calcul mipmaps  → pyramide de textures
#    2. Vertex Shader       → transforme les sommets monde → NDC
#    3. Rasterizer          → triangle → fragments (pixels candidats)
#                             + calcul du LOD (level of detail)
#    4. Fragment Shader     → Phong + texture trilinéaire
#    5. Depth Test          → ne garde que le fragment le plus proche
#
#  Données par sommet (8 valeurs dans le .ply) :
#    [0] x, [1] y, [2] z   → position monde
#    [3] nx,[4] ny,[5] nz  → normale
#    [6] u, [7] v          → coordonnées de texture
#
#  Données après vertex shader (16 valeurs) :
#    [0:3]  position NDC
#    [3:6]  normale monde
#    [6:9]  vecteur V (caméra - sommet)
#    [9:12] vecteur L (lumière - sommet)
#    [12:14] coordonnées UV
#    [14:16] coordonnées NDC (x,y) → stockées pour le LOD
# ============================================================


def edgeSide(p, v0, v1):
    """
    Produit vectoriel 2D de (v0→v1) × (v0→p).

    Résultat positif  → p est à gauche de l'arête v0→v1
    Résultat négatif  → p est à droite
    Résultat = 0      → p est sur l'arête

    Usages :
      - aire signée du triangle (back-face culling)
      - test point-dans-triangle (rasterizer)
    """
    return ((p[0]  - v0[0]) * (v1[1] - v0[1])
          - (p[1]  - v0[1]) * (v1[0] - v0[0]))


class Fragment:
    """
    Un Fragment = un pixel candidat émis par le rasterizer.

    Attributs :
      x, y              : coordonnées pixel (entiers)
      depth             : profondeur z NDC (pour le depth test)
      interpolated_data : attributs interpolés (N, V, L, UV, ...)
      lod               : level of detail calculé pour le mipmap
      output            : couleur finale RGB [0,1] (fragment shader)
    """

    def __init__(self, x, y, depth, interpolated_data, lod=0.0):
        self.x                = x
        self.y                = y
        self.depth            = depth
        self.interpolated_data = interpolated_data
        self.lod              = lod
        self.output           = np.zeros(3, dtype=float)


class GraphicPipeline:
    """
    Pipeline graphique complet :
      build_mipmaps → VertexShader → Rasterizer → FragmentShader → DepthTest
    """

    def __init__(self, width, height):
        """
        Paramètres :
          width, height : résolution de l'image de sortie
        """
        self.width  = width
        self.height = height

        # Image de sortie : (H, W, 3) float [0,1]
        self.image = np.zeros((height, width, 3), dtype=float)

        # Depth buffer initialisé à 1.0 (valeur max NDC).
        # Un fragment passe le test si sa profondeur est inférieure
        # à la valeur déjà stockée.
        self.depthBuffer = np.ones((height, width), dtype=float)

        # Pyramide MIP : remplie dans draw() avant le rendu
        self.mips = None

    # ==============================================================
    #  ÉTAPE 1 — VERTEX SHADER
    # ==============================================================

    def VertexShader(self, vertex, data):
        """
        Transforme un sommet monde → NDC et calcule les vecteurs
        d'éclairage (N, V, L) qui seront interpolés sur le triangle.

        Entrée  (vertex, 8 valeurs) :
          [0:3]  position monde
          [3:6]  normale
          [6:8]  UV texture

        Sortie (outputVertex, 16 valeurs) :
          [0:3]  position NDC  → utilisée par le rasterizer
          [3:6]  normale monde → interpolée, utilisée par le frag shader
          [6:9]  vecteur V     → caméra − sommet (vue)
          [9:12] vecteur L     → lumière − sommet
          [12:14] UV
          [14:16] coordonnées NDC (x, y) → pour le calcul du LOD
        """

        out = np.zeros(16, dtype=float)

        # Position homogène du sommet [x, y, z, 1]
        pos = np.array([vertex[0], vertex[1], vertex[2], 1.0])

        # Transformation : monde → caméra → clip space
        clip = data['projMatrix'] @ (data['viewMatrix'] @ pos)

        # Division de perspective : clip space → NDC
        w = clip[3]
        out[0] = clip[0] / w   # x NDC
        out[1] = clip[1] / w   # y NDC
        out[2] = clip[2] / w   # z NDC (depth)

        # Normale (en coordonnées monde, sera interpolée)
        out[3] = vertex[3]   # nx
        out[4] = vertex[4]   # ny
        out[5] = vertex[5]   # nz

        # Vecteur Vue V = position caméra − position sommet
        # (du sommet vers l'œil, pour le terme spéculaire)
        out[6]  = data['cameraPosition'][0] - vertex[0]
        out[7]  = data['cameraPosition'][1] - vertex[1]
        out[8]  = data['cameraPosition'][2] - vertex[2]

        # Vecteur Lumière L = position lumière − position sommet
        out[9]  = data['lightPosition'][0] - vertex[0]
        out[10] = data['lightPosition'][1] - vertex[1]
        out[11] = data['lightPosition'][2] - vertex[2]

        # Coordonnées de texture
        out[12] = vertex[6]   # u
        out[13] = vertex[7]   # v

        # Coordonnées NDC (x, y) répétées → utilisées par le rasterizer
        # pour calculer les dérivées dU/dx et dV/dy (LOD mipmap)
        out[14] = out[0]   # NDC x
        out[15] = out[1]   # NDC y

        return out

    # ==============================================================
    #  ÉTAPE 2 — RASTERIZER avec calcul du LOD
    # ==============================================================

    def Rasterizer(self, v0, v1, v2):
        """
        Convertit un triangle en liste de Fragments.

        Calcul du LOD (level of detail) pour le mipmap :
          On estime les dérivées des coordonnées UV par rapport
          aux pixels écran (dU/dx, dV/dy).
          LOD = log2( max(|dU/dx|, |dV/dy|) × taille_texture )

          Interprétation :
            dU/dx grand → le triangle est loin ou oblique → peu de
            pixels par texel → on peut utiliser un niveau MIP bas
            résolution pour éviter l'aliasing.

          On calcule dU/dx et dV/dy à partir des coordonnées des
          sommets : variation d'UV divisée par variation de position
          en pixels entre les sommets.
        """

        fragments = []

        # ---- Back-face culling --------------------------------
        # Aire signée en 2D : si négative, triangle dos-à-caméra.
        area = edgeSide(v0, v1, v2)
        if area <= 0:
            return fragments

        # ---- Conversion NDC → pixels --------------------------
        def ndc_to_px(v):
            px = (v[0] + 1.0) * 0.5 * self.width
            py = (v[1] + 1.0) * 0.5 * self.height
            return np.array([px, py])

        p0 = ndc_to_px(v0)
        p1 = ndc_to_px(v1)
        p2 = ndc_to_px(v2)

        # ---- AABB clippée aux bords de l'image ----------------
        A = np.floor(np.min([p0, p1, p2], axis=0)).astype(int)
        B = np.ceil( np.max([p0, p1, p2], axis=0)).astype(int)
        A = np.clip(A, [0, 0], [self.width - 1, self.height - 1])
        B = np.clip(B, [0, 0], [self.width - 1, self.height - 1])

        # ---- Calcul du LOD pour le mipmap ---------------------
        # On estime la variation des UV dans l'espace écran.
        # La dérivée dU/dx ≈ (U_v1 - U_v0) / (px_v1 - px_v0)
        # On prend la valeur maximale parmi les arêtes du triangle.

        u0, v_0 = v0[12], v0[13]
        u1, v_1 = v1[12], v1[13]
        u2, v_2 = v2[12], v2[13]

        # Différences de pixels entre les sommets (protection / 0)
        dx_01 = max(abs(p1[0] - p0[0]), 1e-6)
        dx_02 = max(abs(p2[0] - p0[0]), 1e-6)
        dy_01 = max(abs(p1[1] - p0[1]), 1e-6)
        dy_02 = max(abs(p2[1] - p0[1]), 1e-6)

        # Dérivées approximées (variation UV / variation pixels)
        dU_dx = max(abs(u1 - u0) / dx_01, abs(u2 - u0) / dx_02)
        dV_dy = max(abs(v_1 - v_0) / dy_01, abs(v_2 - v_0) / dy_02)

        # Taille de la texture originale (niveau 0)
        if self.mips is not None:
            tex_size = max(self.mips[0].shape[0], self.mips[0].shape[1])
        else:
            tex_size = 1

        # LOD = log2( max(dU, dV) * taille_texture )
        # Si max_deriv < 1 → LOD négatif → on clamp à 0
        # (on ne peut pas avoir de niveau "plus grand" que 0)
        max_deriv = max(dU_dx, dV_dy)
        if max_deriv > 1e-8:
            lod = np.log2(max_deriv * tex_size)
        else:
            lod = 0.0
        lod = max(0.0, lod)   # clamp minimum à 0

        # ---- Boucle sur les pixels de la AABB ----------------
        for j in range(A[1], B[1] + 1):
            for i in range(A[0], B[0] + 1):

                # Centre du pixel en NDC
                x = (i + 0.5) / self.width  * 2.0 - 1.0
                y = (j + 0.5) / self.height * 2.0 - 1.0
                p = np.array([x, y])

                # Test point-dans-triangle (coordonnées barycentriques)
                a0 = edgeSide(p, v0, v1)
                a1 = edgeSide(p, v1, v2)
                a2 = edgeSide(p, v2, v0)

                if a0 >= 0 and a1 >= 0 and a2 >= 0:

                    # Coordonnées barycentriques 2D brutes
                    l0 = a1 / area
                    l1 = a2 / area
                    l2 = a0 / area

                    # Correction perspective :
                    # Les attributs (UV, normales) ne sont pas
                    # linéaires en espace écran → on pondère par 1/z.
                    iz0 = 1.0 / v0[2] if abs(v0[2]) > 1e-8 else 1.0
                    iz1 = 1.0 / v1[2] if abs(v1[2]) > 1e-8 else 1.0
                    iz2 = 1.0 / v2[2] if abs(v2[2]) > 1e-8 else 1.0

                    # Profondeur interpolée (linéaire en NDC)
                    z = l0 * v0[2] + l1 * v1[2] + l2 * v2[2]

                    # Poids perspective-corrects
                    denom = l0*iz0 + l1*iz1 + l2*iz2
                    if abs(denom) < 1e-12:
                        continue
                    w0 = (l0 * iz0) / denom
                    w1 = (l1 * iz1) / denom
                    w2 = (l2 * iz2) / denom

                    # Interpolation des attributs [3:14]
                    # (normale, V, L, UV, NDC xy)
                    n = v0.shape[0]
                    interp = v0[3:n] * w0 + v1[3:n] * w1 + v2[3:n] * w2

                    fragments.append(Fragment(i, j, z, interp, lod))

        return fragments

    # ==============================================================
    #  ÉTAPE 3 — FRAGMENT SHADER (Phong + mipmap trilinéaire)
    # ==============================================================

    def fragmentShader(self, fragment, data):
        """
        Calcule la couleur finale avec le modèle de Phong + texture.

        Modèle de Phong :
          couleur = (ka·ambiant + kd·diffus + ks·spéculaire) × texture

          N = normale de surface (normalisée)
          L = direction vers la lumière (normalisée)
          V = direction vers la caméra (normalisée)
          R = rayon réfléchi = 2(N·L)N − L

          diffus   = max(N·L, 0)
          spéculaire = max(R·V, 0)^shininess

        Toon shading : quantification en niveaux discrets pour un
          effet "cartoon / cel-shading".

        Texture : sampling trilinéaire avec le LOD calculé par
          le rasterizer (exploite la pyramide MIP).
        """

        # --- Vecteurs d'éclairage (interpolés, puis normalisés) ---
        N = fragment.interpolated_data[0:3]
        n_len = np.linalg.norm(N)
        if n_len < 1e-8:
            fragment.output = np.zeros(3)
            return
        N = N / n_len

        V = fragment.interpolated_data[3:6]
        v_len = np.linalg.norm(V)
        if v_len < 1e-8:
            fragment.output = np.zeros(3)
            return
        V = V / v_len

        L = fragment.interpolated_data[6:9]
        l_len = np.linalg.norm(L)
        if l_len < 1e-8:
            fragment.output = np.zeros(3)
            return
        L = L / l_len

        # --- Modèle de Phong ------------------------------------
        NdotL = np.dot(N, L)

        # Vecteur de réflexion R = 2(N·L)N − L
        R = 2.0 * NdotL * N - L

        ambient  = 1.0
        diffuse  = max(NdotL, 0.0)
        specular = pow(max(np.dot(R, V), 0.0), 64)   # shininess = 64

        ka = 0.1   # coefficient ambiant
        kd = 0.9   # coefficient diffus
        ks = 0.3   # coefficient spéculaire

        phong = ka * ambient + kd * diffuse + ks * specular

        # Toon shading : quantification en 5 niveaux
        phong = np.ceil(phong * 4 + 1) / 6.0

        # --- Sampling texture avec mipmapping trilinéaire --------
        u = fragment.interpolated_data[9]    # coord U
        v = fragment.interpolated_data[10]   # coord V

        # sample_trilinear choisit et mélange deux niveaux MIP
        # selon le LOD calculé lors de la rasterization.
        tex_color = sample_trilinear(self.mips, u, v, fragment.lod)

        # --- Couleur finale : éclairage × texture ----------------
        fragment.output = np.array([phong, phong, phong]) * tex_color

    # ==============================================================
    #  BOUCLE PRINCIPALE
    # ==============================================================

    def draw(self, vertices, triangles, data):
        """
        Exécute le pipeline complet.

        Ordre :
          1. Construction de la pyramide MIP (pré-calcul une seule fois)
          2. Vertex Shader sur tous les sommets
          3. Rasterization de chaque triangle
          4. Fragment Shader + Depth Test sur chaque fragment
        """

        # ---- 1. Construction de la pyramide MIP -----------------
        # On construit les mipmaps ici, une seule fois avant le rendu,
        # pour ne pas les recalculer à chaque fragment.
        print("[INFO] Construction de la pyramide MIP...")
        self.mips = build_mipmaps(data['texture'])
        print(f"[INFO] {len(self.mips)} niveaux MIP générés "
              f"(de {self.mips[0].shape[1]}×{self.mips[0].shape[0]} "
              f"à {self.mips[-1].shape[1]}×{self.mips[-1].shape[0]})")

        nb_vertices = vertices.shape[0]

        # ---- 2. Vertex Shader ------------------------------------
        self.newVertices = np.zeros((nb_vertices, 16), dtype=float)
        for i in range(nb_vertices):
            self.newVertices[i] = self.VertexShader(vertices[i], data)

        # ---- 3. Rasterization ------------------------------------
        all_fragments = []
        for tri in triangles:
            v0 = self.newVertices[tri[0]]
            v1 = self.newVertices[tri[1]]
            v2 = self.newVertices[tri[2]]
            frags = self.Rasterizer(v0, v1, v2)
            all_fragments.extend(frags)

        print(f"[INFO] {len(all_fragments)} fragments générés")

        # ---- 4. Fragment Shader + Depth Test ---------------------
        for f in all_fragments:
            self.fragmentShader(f, data)
            # Depth test : on ne garde que le fragment le plus proche
            if self.depthBuffer[f.y, f.x] > f.depth:
                self.depthBuffer[f.y, f.x] = f.depth
                self.image[f.y, f.x]       = f.output
