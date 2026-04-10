import cv2
import os
import shutil
import sys
import numpy as np
import generate_mimap as gm


def generate_mipmaps_func(img, mode, maskSize, lambda_val, iterations):
    if mode == "filtre":
        levels = gm.filtre(img, lambda_val, iterations)
    elif mode == "moy":
        levels = gm.moy(img, maskSize)
    elif mode == "med":
        levels = gm.med(img, maskSize)
    else:
        print("Erreur: Choix de mode inconnu, doit etre dans [med, moy, filtre]")
        sys.exit(1)
    return levels   


def main():
    if len(sys.argv) < 3:
        print("Usage: python test_script.py <chemin_vers_image> <mode de filtrage (med, moy ou filtre)>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Erreur : L'image {image_path} n'existe pas.")
        sys.exit(1)
        
    print(f"Lecture de l'image : {image_path}")
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Erreur : Impossible de lire l'image {image_path}.")
        sys.exit(1)
        
    print(f"Dimension de l'image originale : {img.shape}")
    
    if sys.argv[2] not in ["med", "moy", "filtre"]:
        print("Erreur: Choix de mode inconnu, doit etre dans [med, moy, filtre]")
        sys.exit(1)
    mode = sys.argv[2]

    # arg pour generer les mipmap
    # pour "filtre"
    lambda_val = 0.05
    iterations = 50
    # pour "med" et "moy"
    maskSize = 3
    
    print("Génération des mipmaps avec filtrage FGP-TV en cours...")

    levels = generate_mipmaps_func(img, mode, maskSize, lambda_val, iterations)
    
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mipmaps_output")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nEnregistrement des niveaux de mipmap dans le dossier '{output_dir}' :")
    dir_name =  os.path.basename(image_path)
    dir_path = os.path.join(output_dir, dir_name, mode)

    if os.path.exists(dir_path):
        shutil.rmtree(dir_path) 
    os.makedirs(dir_path)

    for i, level in enumerate(levels):
        out_path = os.path.join(dir_path, f"level_{i}.png")
        cv2.imwrite(out_path, level)
        print(f" - Enregistré : {out_path} (taille : {level.shape})")

if __name__ == "__main__":
    main()

