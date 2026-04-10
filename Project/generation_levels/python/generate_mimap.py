import numpy as np
import cv2
import convCPyth


def filtre(image_initiale, lambda_val, iterations):
    # 1. Charger la fonction C
    fgp_tv_func = convCPyth.filtrePyth()
    
    current_img = np.ascontiguousarray(image_initiale.astype(np.float32))
    levels = [current_img]
    
    while current_img.shape[0] > 1 and current_img.shape[1] > 1:
        
        current_img = np.ascontiguousarray(current_img)
        cleaned_img = np.empty_like(current_img)
        
        dimX, dimY = current_img.shape[0], current_img.shape[1]
        dimZ = 1 
        
        fgp_tv_func(current_img, lambda_val, iterations, 1e-4, 0, 0, 0, dimX, dimY, dimZ, cleaned_img)
        
        # B. RÉDUCTION (Changement de dimension)
        # On réduit de moitié (ex: 512x512 -> 256x256)
        new_dim = (dimY // 2, dimX // 2)
        # Utilisation de l'interpolation bilinéaire ou aire
        current_img = cv2.resize(cleaned_img, new_dim, interpolation=cv2.INTER_AREA)
        
        levels.append(current_img)
        print(f"Niveau généré : {current_img.shape}")
        
    return levels

def moy(image_initiale, maskSize, padMode=0):
    # 1. Charger la fonction C
    moy_func = convCPyth.moyPyth()
    
    current_img = np.ascontiguousarray(image_initiale.astype(np.float32))
    levels = [current_img]
    
    while current_img.shape[0] > 1 and current_img.shape[1] > 1:
        
        current_img = np.ascontiguousarray(current_img)
        cleaned_img = np.empty_like(current_img)
        
        dimX, dimY = current_img.shape[0], current_img.shape[1]
        dimZ = 1
        
        moy_func(current_img, maskSize, dimX, dimY, dimZ, cleaned_img, padMode)
        
        # B. RÉDUCTION (Changement de dimension)
        # On réduit de moitié (ex: 512x512 -> 256x256)
        new_dim = (dimY // 2, dimX // 2)
        # Utilisation de l'interpolation bilinéaire ou aire
        current_img = cv2.resize(cleaned_img, new_dim, interpolation=cv2.INTER_AREA)
        
        levels.append(current_img)
        print(f"Niveau généré : {current_img.shape}")
        
    return levels


def med(image_initiale, maskSize, padMode=0):
    # 1. Charger la fonction C
    med_func = convCPyth.medPyth()
    
    current_img = np.ascontiguousarray(image_initiale.astype(np.float32))
    levels = [current_img]
    
    while current_img.shape[0] > 1 and current_img.shape[1] > 1:
        
        current_img = np.ascontiguousarray(current_img)
        cleaned_img = np.empty_like(current_img)
        
        dimX, dimY = current_img.shape[0], current_img.shape[1]
        dimZ = 1
        
        med_func(current_img, maskSize, dimX, dimY, dimZ, cleaned_img, padMode)
        
        # B. RÉDUCTION (Changement de dimension)
        # On réduit de moitié (ex: 512x512 -> 256x256)
        new_dim = (dimY // 2, dimX // 2)
        # Utilisation de l'interpolation bilinéaire ou aire
        current_img = cv2.resize(cleaned_img, new_dim, interpolation=cv2.INTER_AREA)
        
        levels.append(current_img)
        print(f"Niveau généré : {current_img.shape}")
        
    return levels