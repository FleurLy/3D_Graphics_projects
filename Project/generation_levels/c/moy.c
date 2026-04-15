#include <stdlib.h>
#include <omp.h>

void moy(float *BaseImage, int maskSize, int dimX, int dimY, float *FinalImage, int padMode) {
    int maskSizediv2 = maskSize / 2;
    int dimX_T = dimX + 2 * maskSizediv2;
    int dimY_T = dimY + 2 * maskSizediv2;
    float* T = (float*) malloc(dimX_T * dimY_T * sizeof(float));

    // construction de T
    for(int i = 0; i < dimX_T; i++) {
        for(int j = 0; j < dimY_T; j++) {
            int orig_i = i - maskSizediv2;
            int orig_j = j - maskSizediv2;
            
            if (orig_i >= 0 && orig_i < dimX && orig_j >= 0 && orig_j < dimY) {
                T[i * dimY_T + j] = BaseImage[orig_i * dimY + orig_j];
            } else {
                if (padMode == 1) {
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

    // construction de FinalImage
    int newDimX = dimX / 2;
    int newDimY = dimY / 2;

    for (int i=0; i < newDimX; i++) {
        for (int j=0; j < newDimY; j++) {
            float val = 0.0;
            int centre_i = 2*i +  1;
            int centre_j = 2*j + 1;

            for (int imar = 0; imar < maskSize; imar++) {
                for (int jmar = 0; jmar < maskSize; jmar++) {
                    val += T[(centre_i + imar) * dimY_T + (centre_j + jmar)];
                }
            }
            FinalImage[i*newDimY + j] = val / ((maskSize)*(maskSize));
        }
    }
    free(T);
}