#include <stdlib.h>


void med_downsample(float *img, int height, int width, float *out) {
    int newH = height / 2;
    int newW = width / 2;

    for (int j = 0; j < newH; j++) {
        for (int i = 0; i < newW; i++) {
            int ind0 = ((2*j)   * width + (2*i)  ) * 3; // indice des pixels concernés 
            int ind1 = ((2*j)   * width + (2*i+1)) * 3;
            int ind2 = ((2*j+1) * width + (2*i)  ) * 3;
            int ind3 = ((2*j+1) * width + (2*i+1)) * 3; 
            int IndCour  = (j * newW + i) * 3;
            for (int rgb = 0; rgb < 3; rgb++) {     // on change tout les cannaux de couleurs
                float tmp;
                float tab[4] = {     // creation de tab et tri à bulles pour avoir la mediane
                    img[ind0 + rgb],
                    img[ind1 + rgb],
                    img[ind2 + rgb],
                    img[ind3 + rgb]
                };            
                for (int i = 0; i < 4; i++) {
                    for (int j = i + 1; j < 4; j++) {
                        if (tab[i] > tab[j]) {
                            tmp = tab[i];
                            tab[i] = tab[j];
                            tab[j] = tmp;
                        }
                    }
                }
                out[IndCour + rgb] = (tab[1]+tab[2])/2; // mediane (moyenne des points median car sinon ça tend vers le blanc ou le noir)
            }
        }
    }
}
