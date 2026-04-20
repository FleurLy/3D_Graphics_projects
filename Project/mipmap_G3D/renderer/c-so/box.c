#include <stdlib.h>

/*
 * box_downsample : moyenne uniforme de blocs 2x2, image RGB float32.
 * img : H*W*3 float, out : (H/2)*(W/2)*3 float
 */
void box_downsample(float *img, int H, int W, float *out) {
    int H2 = H - (H % 2);
    int W2 = W - (W % 2);
    int Ho = H2 / 2;
    int Wo = W2 / 2;

    for (int j = 0; j < Ho; j++) {
        for (int i = 0; i < Wo; i++) {
            int base00 = ((2*j)   * W + (2*i)  ) * 3;
            int base01 = ((2*j)   * W + (2*i+1)) * 3;
            int base10 = ((2*j+1) * W + (2*i)  ) * 3;
            int base11 = ((2*j+1) * W + (2*i+1)) * 3;
            int obase  = (j * Wo + i) * 3;
            for (int c = 0; c < 3; c++) {
                out[obase + c] = (img[base00+c] + img[base01+c]
                                + img[base10+c] + img[base11+c]) * 0.25f;
            }
        }
    }
}
