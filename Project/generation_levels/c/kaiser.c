#include <stdlib.h>
#include <math.h>
#include <omp.h>

// Fonction de Bessel modifiée de première espèce d'ordre 0
// Nécessaire pour le calcul de la fenêtre de Kaiser
double besselI0(double x) {
    double sum = 1.0;
    double term = 1.0;
    for (int k = 1; k < 25; k++) {
        term *= (x * x) / (4.0 * k * k);
        sum += term;
        if (term < sum * 1e-8) break;
    }
    return sum;
}

void kaiser(float *BaseImage, int maskSize, int dimX, int dimY, float *FinalImage, int padMode, float alpha) {
    int maskSizediv2 = maskSize / 2;
    int dimX_T = dimX + 2 * maskSizediv2;
    int dimY_T = dimY + 2 * maskSizediv2;
    float* T = (float*) malloc(dimX_T * dimY_T * sizeof(float));

    // 1. Construction de l'image avec Padding (Identique à ton code)
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < dimX_T; i++) {
        for(int j = 0; j < dimY_T; j++) {
            int orig_i = i - maskSizediv2;
            int orig_j = j - maskSizediv2;
            
            if (orig_i >= 0 && orig_i < dimX && orig_j >= 0 && orig_j < dimY) {
                T[i * dimY_T + j] = BaseImage[orig_i * dimY + orig_j];
            } else {
                if (padMode == 1) { // Clamp
                    if (orig_i < 0) orig_i = 0;
                    if (orig_i >= dimX) orig_i = dimX - 1;
                    if (orig_j < 0) orig_j = 0;
                    if (orig_j >= dimY) orig_j = dimY - 1;
                    T[i * dimY_T + j] = BaseImage[orig_i * dimY + orig_j];
                } else { // Zero pad
                    T[i * dimY_T + j] = 0.0f;
                }
            }
        }
    }

    // 2. Pré-calcul du masque de Kaiser (Fenêtre 2D)
    float* mask = (float*) malloc(maskSize * maskSize * sizeof(float));
    float sumWeights = 0.0f;
    double i0_alpha = besselI0(alpha);

    for (int m = 0; m < maskSize; m++) {
        for (int n = 0; n < maskSize; n++) {
            // Distance normalisée par rapport au centre du masque [-1, 1]
            double dx = (2.0 * m / (maskSize - 1)) - 1.0;
            double dy = (2.0 * n / (maskSize - 1)) - 1.0;
            double r = sqrt(dx*dx + dy*dy);
            
            if (r <= 1.0) {
                mask[m * maskSize + n] = (float)(besselI0(alpha * sqrt(1.0 - r*r)) / i0_alpha);
            } else {
                mask[m * maskSize + n] = 0.0f;
            }
            sumWeights += mask[m * maskSize + n];
        }
    }

    // Normalisation du masque pour conserver la luminosité
    for (int i = 0; i < maskSize * maskSize; i++) mask[i] /= sumWeights;

    // 3. Construction de FinalImage avec convolution pondérée
    int newDimX = dimX / 2;
    int newDimY = dimY / 2;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < newDimX; i++) {
        for (int j = 0; j < newDimY; j++) {
            float val = 0.0f;
            // On se place sur les pixels correspondants dans T (stride de 2 pour le mipmap)
            int start_i = 2 * i; 
            int start_j = 2 * j;

            for (int imar = 0; imar < maskSize; imar++) {
                for (int jmar = 0; jmar < maskSize; jmar++) {
                    val += T[(start_i + imar) * dimY_T + (start_j + jmar)] * mask[imar * maskSize + jmar];
                }
            }
            FinalImage[i * newDimY + j] = val;
        }
    }

    free(T);
    free(mask);
}