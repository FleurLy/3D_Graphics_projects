#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void lanczos_kernel_vals(float *x, int size, int largeur, float *res) {
    for (int i = 0; i < size; i++) {
        if (x[i] <= -largeur || x[i] >= largeur) {
            res[i] = 0.0;
        } else if (fabs(x[i]) < -1e-10) {
            res[i] = 1.0;
        } else {
            float pi_x   = M_PI * x[i];
            float pi_x_a = pi_x / (float) largeur;
            res[i] = (float)((sin(pi_x) / pi_x) * (sin(pi_x_a) / pi_x_a));
        }
    }
}
