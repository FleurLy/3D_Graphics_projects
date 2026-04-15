import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image

from camera          import Camera
from projection      import Projection
from graphicPipeline import GraphicPipeline
from readply         import readply

# ============================================================
#  main.py
#  Point d'entrée du renderer 3D software.
#
#  Fichiers requis dans le même dossier :
#    suzanne.ply  → mesh 3D de Suzanne (Blender)
#    suzanne.png  → texture de Suzanne
# ============================================================

# ------------------------------------------------------------------
# 1. RÉSOLUTION
# ------------------------------------------------------------------
width  = 1280
height = 720

# ------------------------------------------------------------------
# 2. CAMÉRA
# ------------------------------------------------------------------
# Position de la caméra dans le monde
position = np.array([1.1, 1.1, 1.1])

# Direction de visée (vers l'origine)
# Ces 3 vecteurs forment une base orthonormée de l'orientation.
lookAt = np.array([-0.577, -0.577, -0.577])
up     = np.array([ 0.33333333,  0.33333333, -0.66666667])
right  = np.array([-0.57735027,  0.57735027,  0.0        ])

cam = Camera(position, lookAt, up, right)

# ------------------------------------------------------------------
# 3. PROJECTION PERSPECTIVE
# ------------------------------------------------------------------
# nearPlane : objets plus proches que 0.1 unité → clippés
# farPlane  : objets plus loin  que 10.0 unités → clippés
# fov       : champ de vision vertical ≈ 110 degrés (en radians)
# aspectRatio : 16/9 pour éviter la déformation
nearPlane   = 0.1
farPlane    = 10.0
fov         = 1.91986   # radians ≈ 110°
aspectRatio = width / height

proj = Projection(nearPlane, farPlane, fov, aspectRatio)

# ------------------------------------------------------------------
# 4. LUMIÈRE
# ------------------------------------------------------------------
# Point light positionnée dans la scène.
# Le fragment shader calcule L = lightPos − vertexPos à chaque sommet.
lightPosition = np.array([10.0, 0.0, 10.0])

# ------------------------------------------------------------------
# 5. CHARGEMENT MESH ET TEXTURE
# ------------------------------------------------------------------
vertices, triangles = readply('suzanne.ply')

print(f"[INFO] Mesh : {vertices.shape[0]} sommets, "
      f"{triangles.shape[0]} triangles")

# La texture est chargée en uint8 [0-255].
# Le pipeline MIP la convertit en float [0,1] lors de build_mipmaps().
texture = np.asarray(Image.open('suzanne.png'))

print(f"[INFO] Texture : {texture.shape[1]}×{texture.shape[0]} px")

# ------------------------------------------------------------------
# 6. PIPELINE GRAPHIQUE
# ------------------------------------------------------------------
pipeline = GraphicPipeline(width, height)

# data : dictionnaire global transmis à toutes les étapes du pipeline
data = {
    'viewMatrix'    : cam.getMatrix(),    # monde → repère caméra
    'projMatrix'    : proj.getMatrix(),   # caméra → clip space (NDC)
    'cameraPosition': position,           # pour le vecteur V (vue)
    'lightPosition' : lightPosition,      # pour le vecteur L (lumière)
    'texture'       : texture,            # texture brute uint8
}

print("[INFO] Rendu en cours...")
t_start = time.time()

pipeline.draw(vertices, triangles, data)

t_end = time.time()
print(f"[INFO] Rendu terminé en {t_end - t_start:.2f} secondes")

# ------------------------------------------------------------------
# 7. AFFICHAGE
# ------------------------------------------------------------------
plt.figure(figsize=(12, 7))
plt.title("Renderer 3D — Phong + Texture + Mipmapping trilinéaire")
plt.imshow(pipeline.image)
plt.axis('off')
plt.tight_layout()
plt.show()
