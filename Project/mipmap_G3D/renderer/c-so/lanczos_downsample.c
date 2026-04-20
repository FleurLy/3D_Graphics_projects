#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static float lanczos_val(double x, int a) {
    double da = (double)a;
    if (x <= -da || x >= da) return 0.0f;
    if (x > -1e-10 && x < 1e-10) return 1.0f;
    double pi_x = M_PI * x;
    return (float)((sin(pi_x) / pi_x) * (sin(pi_x / da) / (pi_x / da)));
}

/*
 * lanczos_downsample : filtre Lanczos ordre a + sous-echantillonnage x2.
 * img : H*W*3 float [0..1], out : (H/2)*(W/2)*3 float [0..1]
 */
void lanczos_downsample(float *img, int H, int W, int a, float *out) {
    int Ho = H / 2;
    int Wo = W / 2;

    for (int j = 0; j < Ho; j++) {
        double cy  = 2.0 * j + 0.5;
        int iy     = (int)cy;
        int ys0    = iy - a + 1; if (ys0 < 0) ys0 = 0;
        int ys1    = iy + a + 1; if (ys1 > H) ys1 = H;
        int ny     = ys1 - ys0;

        float *wy = (float *) malloc(ny * sizeof(float));
        for (int k = 0; k < ny; k++)
            wy[k] = lanczos_val((ys0 + k) - cy, a);

        for (int i = 0; i < Wo; i++) {
            double cx = 2.0 * i + 0.5;
            int ix    = (int)cx;
            int xs0   = ix - a + 1; if (xs0 < 0) xs0 = 0;
            int xs1   = ix + a + 1; if (xs1 > W) xs1 = W;
            int nx    = xs1 - xs0;

            float *wx = (float *) malloc(nx * sizeof(float));
            for (int k = 0; k < nx; k++)
                wx[k] = lanczos_val((xs0 + k) - cx, a);

            /* somme des poids 2D */
            float w_sum = 0.0f;
            for (int ky = 0; ky < ny; ky++)
                for (int kx = 0; kx < nx; kx++)
                    w_sum += wy[ky] * wx[kx];

            float r = 0.0f, g = 0.0f, b = 0.0f;
            if (w_sum >= 1e-8f) {
                float inv = 1.0f / w_sum;
                for (int ky = 0; ky < ny; ky++) {
                    int y = ys0 + ky;
                    for (int kx = 0; kx < nx; kx++) {
                        int x   = xs0 + kx;
                        float w = wy[ky] * wx[kx] * inv;
                        int base = (y * W + x) * 3;
                        r += w * img[base+0];
                        g += w * img[base+1];
                        b += w * img[base+2];
                    }
                }
            }

            int obase = (j * Wo + i) * 3;
            out[obase+0] = r < 0.0f ? 0.0f : (r > 1.0f ? 1.0f : r);
            out[obase+1] = g < 0.0f ? 0.0f : (g > 1.0f ? 1.0f : g);
            out[obase+2] = b < 0.0f ? 0.0f : (b > 1.0f ? 1.0f : b);

            free(wx);
        }
        free(wy);
    }
}
