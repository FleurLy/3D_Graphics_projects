#include <stdlib.h>
#include <math.h>
#include <omp.h>

// Kernel de Mitchell-Netravali (avec B=1/3, C=1/3 comme recommandé dans leur rapport)
float kernel(float x) {
    float B = 1.0f/3.0f;
    float C = 1.0f/3.0f;
    x = fabsf(x);
    float x2 = x * x;
    float x3 = x * x * x;

    if (x < 1.0f) {
        return ((12.0f - 9.0f * B - 6.0f * C) * x3 + 
                (-18.0f + 12.0f * B + 6.0f * C) * x2 + 
                (6.0f - 2.0f * B)) / 6.0f;
    } else if (x < 2.0f) {
        return ((-B - 6.0f * C) * x3 + 
                (6.0f * B + 30.0f * C) * x2 + 
                (-12.0f * B - 48.0f * C) * x + 
                (8.0f * B + 24.0f * C)) / 6.0f;
    }
    return 0.0f;
}


void miNe(float *BaseImage, int dimX, int dimY, float *FinalImage,  int maskSizediv2Mode) {
    int maskSizediv2 = 4;   
    int dimX_T = dimX + 2 * maskSizediv2;
    int dimY_T = dimY + 2 * maskSizediv2;
    float* T = (float*) malloc(dimX_T * dimY_T * sizeof(float));        // T BaseImage "padé" de BaseImage


    // construction de T
    for (int i = 0; i < dimX_T; i++) {
        for (int j = 0; j < dimY_T; j++) {
            int orig_i = i - maskSizediv2;
            int orig_j = j - maskSizediv2;
            
            if (orig_i >= 0 && orig_i < dimX && orig_j >= 0 && orig_j < dimY) {
                T[i * dimY_T + j] = BaseImage[orig_i * dimY + orig_j];
            } else {
                if (maskSizediv2Mode == 1) {                                // Répétition des bords
                    if (orig_i < 0) orig_i = 0;
                    if (orig_i >= dimX) orig_i = dimX - 1;
                    if (orig_j < 0) orig_j = 0;
                    if (orig_j >= dimY) orig_j = dimY - 1;
                    T[i * dimY_T + j] = BaseImage[orig_i * dimY + orig_j];
                } else {
                    T[i * dimY_T + j] = 0.0f;
                }
            }
        }
    }

//--------------------------------------------------------------------------------------

    // filtrage et modification de BaseImage

    int newDimX = dimX / 2;
    int newDimY = dimY / 2;


    for (int i = 0; i < newDimX; i++) {
        for (int j = 0; j < newDimY; j++) {
            // Position centrale dans T
            float center_i = (float)i * 2. + maskSizediv2 + 0.5f;
            float center_j = (float)j * 2.  + maskSizediv2 + 0.5f;

            float val = 0.0f;
            float PoidsTot = 0.0f;

            for (int imar = -maskSizediv2; imar <= maskSizediv2; imar++) {
                for (int jmar = -maskSizediv2; jmar <= maskSizediv2; jmar++) {
                    int src_i = (int)floorf(center_i) + imar;
                    int src_j = (int)floorf(center_j) + jmar;

                    // Calcul de la distance par rapport au centre de reconstruction
                    float dist_i = fabsf((float)src_i - center_i);
                    float dist_j = fabsf((float)src_j - center_j);

                    // Poids combiné Mitchell
                    float Poids = kernel(dist_i) * kernel(dist_j);
                    
                    val += T[src_i * dimY_T + src_j] * Poids;
                    PoidsTot += Poids;
                }
            }
            FinalImage[i * newDimY + j] = val / PoidsTot;
        }
    }
    free(T);
}