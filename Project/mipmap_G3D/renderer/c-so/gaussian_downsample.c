#include <stdlib.h>
#include <string.h>

/*
 * gaussian_downsample : noyau separable 4-taps [0.0625,0.4375,0.4375,0.0625]
 * applique horizontalement puis verticalement, suivi d'un sous-echantillonnage x2.
 * img : H*W*3 float, out : (H/2)*(W/2)*3 float
 */
void gaussian_downsample(float *img, int H, int W, float *out) {
    static const float k[4] = {0.0625f, 0.4375f, 0.4375f, 0.0625f};
    /* offsets du noyau : k-1 pour k in {0,1,2,3} = {-1,0,+1,+2} */

    float *horiz = (float *) malloc(H * W * 3 * sizeof(float));
    memset(horiz, 0, H * W * 3 * sizeof(float));

    /* Convolution horizontale */
    for (int y = 0; y < H; y++) {
        for (int xd = 0; xd < W; xd++) {
            for (int ki = 0; ki < 4; ki++) {
                int x = xd + (ki - 1);
                if (x < 0 || x >= W) continue;
                int src = (y * W + x)  * 3;
                int dst = (y * W + xd) * 3;
                horiz[dst+0] += k[ki] * img[src+0];
                horiz[dst+1] += k[ki] * img[src+1];
                horiz[dst+2] += k[ki] * img[src+2];
            }
        }
    }

    /* Convolution verticale dans un second buffer */
    float *vert = (float *) malloc(H * W * 3 * sizeof(float));
    memset(vert, 0, H * W * 3 * sizeof(float));

    for (int yd = 0; yd < H; yd++) {
        for (int x = 0; x < W; x++) {
            for (int ki = 0; ki < 4; ki++) {
                int y = yd + (ki - 1);
                if (y < 0 || y >= H) continue;
                int src = (y  * W + x) * 3;
                int dst = (yd * W + x) * 3;
                vert[dst+0] += k[ki] * horiz[src+0];
                vert[dst+1] += k[ki] * horiz[src+1];
                vert[dst+2] += k[ki] * horiz[src+2];
            }
        }
    }
    free(horiz);

    /* Sous-echantillonnage x2 : prendre vert[2j, 2i] */
    int Ho = H / 2;
    int Wo = W / 2;
    for (int j = 0; j < Ho; j++) {
        for (int i = 0; i < Wo; i++) {
            int src   = (2*j * W + 2*i) * 3;
            int obase = (j * Wo + i)    * 3;
            out[obase+0] = vert[src+0];
            out[obase+1] = vert[src+1];
            out[obase+2] = vert[src+2];
        }
    }
    free(vert);
}
