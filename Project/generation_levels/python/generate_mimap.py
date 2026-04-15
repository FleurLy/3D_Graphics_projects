import numpy as np
import cv2
import convCPyth


def filtre(image_initiale, lambda_val, iterations):
    if len(image_initiale.shape) == 3 and image_initiale.shape[2] in [3, 4]:
        levels_channels = [filtre(image_initiale[:, :, c], lambda_val, iterations) for c in range(image_initiale.shape[2])]
        return [np.dstack(tuple(levels_channels[c][i] for c in range(image_initiale.shape[2]))) for i in range(len(levels_channels[0]))]

    # 1. Charger la fonction C
    fgp_tv_func = convCPyth.filtrePyth()
    
    current_img = np.ascontiguousarray(image_initiale.astype(np.float32))
    levels = [current_img]
    
    while current_img.shape[0] > 1 and current_img.shape[1] > 1:
        
        current_img = np.ascontiguousarray(current_img)
        cleaned_img = np.empty_like(current_img)
        
        dimX, dimY = current_img.shape[0], current_img.shape[1]
            
        fgp_tv_func(current_img, lambda_val, iterations, 1e-4, 0, 0, 0, dimX, dimY, cleaned_img)
        
        # B. RÉDUCTION (Changement de dimension)
        # On réduit de moitié (ex: 512x512 -> 256x256)
        new_dim = (dimY // 2, dimX // 2)
        # Utilisation de l'interpolation bilinéaire ou aire
        current_img = cv2.resize(cleaned_img, new_dim, interpolation=cv2.INTER_AREA)
        
        levels.append(current_img)
        
    return levels

def moy(image_initiale, maskSize, padMode=0):
    if len(image_initiale.shape) == 3 and image_initiale.shape[2] in [3, 4]:
        levels_channels = [moy(image_initiale[:, :, c], maskSize, padMode) for c in range(image_initiale.shape[2])]
        return [np.dstack(tuple(levels_channels[c][i] for c in range(image_initiale.shape[2]))) for i in range(len(levels_channels[0]))]

    moy_func = convCPyth.moyPyth()
    
    current_img = np.ascontiguousarray(image_initiale.astype(np.float32))
    img_entier = np.clip(current_img, 0, 255).astype(np.uint8)
    levels = [img_entier]
    
    while current_img.shape[0] > 1 and current_img.shape[1] > 1:
        
        current_img = np.ascontiguousarray(current_img)
        dimX, dimY = current_img.shape[0], current_img.shape[1]
        cleaned_img = np.zeros((dimX // 2, dimY // 2), dtype=np.float32)        
        cleaned_img = np.ascontiguousarray(cleaned_img)  
        
        moy_func(current_img, maskSize, dimX, dimY, cleaned_img, padMode)
        
        img_entier = np.clip(cleaned_img, 0, 255).astype(np.uint8)
        levels.append(img_entier)

        current_img = cleaned_img
        
    return levels


def med(image_initiale, maskSize, padMode=0):
    if len(image_initiale.shape) == 3 and image_initiale.shape[2] in [3, 4]:
        levels_channels = [med(image_initiale[:, :, c], maskSize, padMode) for c in range(image_initiale.shape[2])]
        return [np.dstack(tuple(levels_channels[c][i] for c in range(image_initiale.shape[2]))) for i in range(len(levels_channels[0]))]

    med_func = convCPyth.medPyth()
    
    current_img = np.ascontiguousarray(image_initiale.astype(np.float32))
    img_entier = np.clip(current_img, 0, 255).astype(np.uint8)
    levels = [img_entier]
    
    while current_img.shape[0] > 1 and current_img.shape[1] > 1:
        
        current_img = np.ascontiguousarray(current_img)
        
        dimX, dimY = current_img.shape[0], current_img.shape[1]
        dimX, dimY = current_img.shape[0], current_img.shape[1]
        cleaned_img = np.zeros((dimX // 2, dimY // 2), dtype=np.float32)        
        cleaned_img = np.ascontiguousarray(cleaned_img)  

        
        med_func(current_img, maskSize, dimX, dimY, cleaned_img, padMode)
        
        img_entier = np.clip(cleaned_img, 0, 255).astype(np.uint8)
        levels.append(img_entier)

        current_img = cleaned_img
        
        
    return levels

def miNe(image_initiale, padMode=0):
    if len(image_initiale.shape) == 3 and image_initiale.shape[2] in [3, 4]:
        levels_channels = [miNe(image_initiale[:, :, c], padMode) for c in range(image_initiale.shape[2])]
        return [np.dstack(tuple(levels_channels[c][i] for c in range(image_initiale.shape[2]))) for i in range(len(levels_channels[0]))]

    miNe_func = convCPyth.miNePyth()
    
    current_img = np.ascontiguousarray(image_initiale.astype(np.float32))
    img_entier = np.clip(current_img, 0, 255).astype(np.uint8)
    levels = [img_entier]
    
    while current_img.shape[0] > 1 and current_img.shape[1] > 1:
        
        dimX, dimY = current_img.shape[0], current_img.shape[1]
        
        current_img = np.ascontiguousarray(current_img)
        
        dimX, dimY = current_img.shape[0], current_img.shape[1]
        dimX, dimY = current_img.shape[0], current_img.shape[1]
        cleaned_img = np.zeros((dimX // 2, dimY // 2), dtype=np.float32)        
        cleaned_img = np.ascontiguousarray(cleaned_img)  
        
        miNe_func(current_img, dimX, dimY, cleaned_img, padMode)
        
        img_entier = np.clip(cleaned_img, 0, 255).astype(np.uint8)
        levels.append(img_entier)

        current_img = cleaned_img
        
    return levels

def kaiser(image_initiale, maskSize, alpha, padMode=0):
    if len(image_initiale.shape) == 3 and image_initiale.shape[2] in [3, 4]:
        levels_channels = [kaiser(image_initiale[:, :, c], maskSize, alpha, padMode) for c in range(image_initiale.shape[2])]
        return [np.dstack(tuple(levels_channels[c][i] for c in range(image_initiale.shape[2]))) for i in range(len(levels_channels[0]))]

    kaiser_func = convCPyth.KaiserPyth()
    
    current_img = np.ascontiguousarray(image_initiale.astype(np.float32))
    img_entier = np.clip(current_img, 0, 255).astype(np.uint8)
    levels = [img_entier]
    
    while current_img.shape[0] > 1 and current_img.shape[1] > 1:
        
        dimX, dimY = current_img.shape[0], current_img.shape[1]
        
        current_img = np.ascontiguousarray(current_img)
        
        dimX, dimY = current_img.shape[0], current_img.shape[1]
        dimX, dimY = current_img.shape[0], current_img.shape[1]
        cleaned_img = np.zeros((dimX // 2, dimY // 2), dtype=np.float32)        
        cleaned_img = np.ascontiguousarray(cleaned_img)  
        
        kaiser_func(current_img, maskSize, dimX, dimY, cleaned_img, padMode, alpha)
        
        img_entier = np.clip(cleaned_img, 0, 255).astype(np.uint8)
        levels.append(img_entier)

        current_img = cleaned_img
        
    return levels