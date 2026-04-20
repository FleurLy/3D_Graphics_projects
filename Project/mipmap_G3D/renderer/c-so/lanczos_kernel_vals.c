#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*
 * lanczos_kernel_vals : sinc(x)*sinc(x/a) pour |x| < a, 0 sinon.
 * x      : tableau de n valeurs float en entree
 * result : tableau de n valeurs float en sortie
 */
void lanczos_kernel_vals(float *x, int n, int a, float *result) {
    double da = (double)a;
    for (int i = 0; i < n; i++) {
        double xi = (double)x[i];
        if (xi <= -da || xi >= da) {
            result[i] = 0.0f;
        } else if (xi > -1e-10 && xi < 1e-10) {
            result[i] = 1.0f;
        } else {
            double pi_x   = M_PI * xi;
            double pi_x_a = pi_x / da;
            result[i] = (float)((sin(pi_x) / pi_x) * (sin(pi_x_a) / pi_x_a));
        }
    }
}
