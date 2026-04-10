    #include <stdlib.h>
    #include <omp.h>

    void med(float *A, int maskSize, int dimX, int dimY, int dimZ, float *D, int padMode) 
    {
        int maskSizediv2 = maskSize / 2;
        int dimX_T = dimX + 2*maskSizediv2;
        int dimY_T = dimY + 2*maskSizediv2;

        float* T = (float*) malloc(dimX_T * dimY_T * sizeof(float));
        
        #pragma omp parallel for
        for (int i = 0; i < dimX_T; i++) {
            for (int j = 0; j < dimY_T; j++) {
                int orig_i = i - maskSizediv2;
                int orig_j = j - maskSizediv2;
                
                if (orig_i >= 0 && orig_i < dimX && orig_j >= 0 && orig_j < dimY) {
                    T[i * dimY_T + j] = A[orig_i * dimY + orig_j];
                } else {
                    if (padMode == 1) { // Répétition des bords
                        if (orig_i < 0) orig_i = 0;
                        if (orig_i >= dimX) orig_i = dimX - 1;
                        if (orig_j < 0) orig_j = 0;
                        if (orig_j >= dimY) orig_j = dimY - 1;
                        T[i * dimY_T + j] = A[orig_i * dimY + orig_j];
                    } else { // Remplissage par 0
                        T[i * dimY_T + j] = 0.0f;
                    }
                }
            }
        }

        int size = maskSize * maskSize;

        #pragma omp parallel for
        for (int i=0; i < dimX; i++) {
            // Allocate private array for sorting per thread to avoid race conditions
            float* tab = (float*) malloc(size * sizeof(float));
            
            for (int j=0; j < dimY; j++) {
                int i_selon_T = i + maskSizediv2;
                int j_selon_T = j + maskSizediv2;
                
                int idx = 0;
                for (int imar = -maskSizediv2; imar <= maskSizediv2; imar++) {
                    for (int jmar = -maskSizediv2; jmar <= maskSizediv2; jmar++) {
                        tab[idx++] = T[(i_selon_T + imar) * dimY_T + (j_selon_T + jmar)];
                    }
                }

                // Insertion sort (très rapide pour de petits tableaux comme 3x3 ou 5x5)
                for (int k = 1; k < size; k++) {
                    float key = tab[k];
                    int m = k - 1;
                    while (m >= 0 && tab[m] > key) {
                        tab[m + 1] = tab[m];
                        m = m - 1;
                    }
                    tab[m + 1] = key;
                }

                D[i*dimY + j] = tab[size / 2];
            }
            free(tab);
        }
        free(T);
    }