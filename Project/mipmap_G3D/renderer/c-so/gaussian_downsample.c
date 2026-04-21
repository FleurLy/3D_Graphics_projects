#include <stdlib.h>
#include <string.h>


void gaussian_downsample(float *img, int H, int W, float *out) {

    float Poids[4] = {0.0625f, 0.4375f, 0.4375f, 0.0625f}; // poids du filtre

    /* Convolution horizontale */
    float *imgFloueHoriz = (float *) malloc(H * W * 3 * sizeof(float));
    memset(imgFloueHoriz, 0, H * W * 3 * sizeof(float));
    for (int y = 0; y < H; y++) {
        for (int xd = 0; xd < W; xd++) {
            for (int indPoids = 0; indPoids < 4; indPoids++) {
                int x = xd + (indPoids - 1);
                if (x < 0 || x >= W) continue;  // bords
                int indSrc = (y * W + x)  * 3;
                int indDst = (y * W + xd) * 3;
                imgFloueHoriz[indDst] += Poids[indPoids] * img[indSrc];
                imgFloueHoriz[indDst+1] += Poids[indPoids] * img[indSrc+1];
                imgFloueHoriz[indDst+2] += Poids[indPoids] * img[indSrc+2];
            }
        }
    }

    /* Convolution verticale */
    float *imgFloue = (float *) malloc(H * W * 3 * sizeof(float));
    memset(imgFloue, 0, H * W * 3 * sizeof(float));

    for (int yd = 0; yd < H; yd++) {
        for (int x = 0; x < W; x++) {
            for (int indPoids = 0; indPoids < 4; indPoids++) {
                int y = yd + (indPoids - 1);
                if (y < 0 || y >= H) continue;  // bord
                int indSrc = (y  * W + x) * 3;
                int indDst = (yd * W + x) * 3;
                imgFloue[indDst] += Poids[indPoids] * imgFloueHoriz[indSrc];
                imgFloue[indDst+1] += Poids[indPoids] * imgFloueHoriz[indSrc+1];
                imgFloue[indDst+2] += Poids[indPoids] * imgFloueHoriz[indSrc+2];
            }
        }
    }
    free(imgFloueHoriz);

    // Sous-echantillonnage
    int newH = H / 2;
    int newW = W / 2;
    for (int j = 0; j < newH; j++) {
        for (int i = 0; i < newW; i++) {
            int indSrc   = (2*j * W + 2*i) * 3;
            int indOut = (j * newW + i)    * 3;
            out[indOut] = imgFloue[indSrc];
            out[indOut+1] = imgFloue[indSrc+1];
            out[indOut+2] = imgFloue[indSrc+2];
        }
    }
    free(imgFloue);
}
