#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

float lanczos_val(double x, int largeur) {
    double ax = fabs(x);
    if (ax >= (double)largeur) return 0.0f;
    if (ax < 1e-10) return 1.0f;
    double pi_x = M_PI * x;
    return (float)((sin(pi_x) / pi_x) * (sin(pi_x / largeur) / (pi_x / largeur)));
}

void lanczos_downsample(float *img, int H, int W, int a, float *out) {
    int newH = H / 2;
    int newW = W / 2;

    for (int j = 0; j < newH; j++) {
        double centreY = 2.0 * j + 0.5; 
        
        int borneInfY = (int)centreY - a + 1; if (borneInfY < 0) borneInfY = 0;
        int borneSupY = (int)centreY + a + 1; if (borneSupY > H) borneSupY = H;
        int HeightRect = borneSupY - borneInfY;

        float *PoidsVert = (float *) malloc(HeightRect * sizeof(float));
        for (int k = 0; k < HeightRect; k++)
            PoidsVert[k] = lanczos_val((borneInfY + k) - centreY, a);

        for (int i = 0; i < newW; i++) {
            double centreX = 2.0 * i + 0.5;            
            int borneInfX = (int)centreX - a + 1; if (borneInfX < 0) borneInfX = 0;
            int borneSupX = (int)centreX + a + 1; if (borneSupX > W) borneSupX = W;
            int WidthRect = borneSupX - borneInfX;

            float *PoidsHoriz = (float *) malloc(WidthRect * sizeof(float));
            for (int k = 0; k < WidthRect; k++)
                PoidsHoriz[k] = lanczos_val((borneInfX + k) - centreX, a);

            // Somme des poids pour normalisation
            float sommePoids = 0.0f;
            for (int yR = 0; yR < HeightRect; yR++) {
                for (int xR = 0; xR < WidthRect; xR++) {
                    sommePoids += PoidsVert[yR] * PoidsHoriz[xR];
                }
            }

            float rgb[3] = {0.0f, 0.0f, 0.0f};

            if (sommePoids > 1e-8f) {
                for (int yR = 0; yR < HeightRect; yR++) {
                    int y = borneInfY + yR;
                    for (int xR = 0; xR < WidthRect; xR++) {
                        int x = borneInfX + xR;
                        float poidsFinal = (PoidsVert[yR] * PoidsHoriz[xR]) / sommePoids;
                        
                        int indImg = (y * W + x) * 3;
                        rgb[0] += poidsFinal * img[indImg];
                        rgb[1] += poidsFinal * img[indImg + 1];
                        rgb[2] += poidsFinal * img[indImg + 2];
                    }
                }
            }

            // Remplissage de out avec clamping
            int indOut = (j * newW + i) * 3;
            for (int c = 0; c < 3; c++) {
                float val = rgb[c];
                if (val < 0.0f) val = 0.0f;
                if (val > 1.0f) val = 1.0f;
                out[indOut + c] = val;
            }

            free(PoidsHoriz);
        }
        free(PoidsVert);
    }
}