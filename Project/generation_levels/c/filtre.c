#include <math.h>
#include <stdlib.h>
#include <memory.h>
#include <stdio.h>
#include "omp.h"

/* C-OMP implementation of FGP-TV [1] denoising/regularization model (2D/3D case)
 *
 * Input Parameters:
 * 1. Noisy image/volume [REQUIRED]
 * 2. lambda - regularization parameter [REQUIRED]
 * 3. Number of iterations [OPTIONAL parameter]
 * 4. eplsilon: tolerance constant [OPTIONAL parameter]
 * 5. TV-type: 'iso' or 'l1' [OPTIONAL parameter]
 * 6. nonneg: 'nonnegativity (0 is OFF by default) [OPTIONAL parameter]
 * 7. print information: 0 (off) or 1 (on) [OPTIONAL parameter]
 * 8. P1 (dual variable from the previous outer iteration) [OPTIONAL parameter]
 * 9. P2 (dual variable from the previous outer iteration) [OPTIONAL parameter]
 *
 * Output:
 * [1] Filtered/regularized image
 * [2] last function value
 * [3] P1 (dual variable from the previous outer iteration) [if 8 is provided]
 * [4] P2 (dual variable from the previous outer iteration) [if 9 is provided]
 *
 * Example of image denoising:
 * figure;
 * Im = double(imread('lena_gray_256.tif'))/255;  % loading image
 * u0 = Im + .05*randn(size(Im)); % adding noise
 * u = FGP_TV(single(u0), 0.05, 100, 1e-04);
 *
 * to compile with OMP support: gcc -shared -Wall -std=c99 -Wl,-soname,FGP_TV -fopenmp -o FGP_TV.so -fPIC FGP_TV.c
 * This function is based on the Matlab's code and paper by
 * [1] Amir Beck and Marc Teboulle, "Fast Gradient-Based Algorithms for Constrained Total Variation Image Denoising and Deblurring Problems"
 *
 * D. Kazantsev, 2016-17
 *
 */

float copyIm(float *A, float *B, int dimX, int dimY, int dimZ);
float Obj_func2D(float *A, float *D, float *R1, float *R2, float lambda, int dimX, int dimY);
float Grad_func2D(float *P1, float *P2, float *D, float *R1, float *R2, float lambda, int dimX, int dimY);
float Proj_func2D(float *P1, float *P2, int methTV, int dimX, int dimY);
float Rupd_func2D(float *P1, float *P1_old, float *P2, float *P2_old, float *R1, float *R2, float tkp1, float tk, int dimX, int dimY);
float Obj_func_CALC2D(float *A, float *D, float *funcvalA, float lambda, int dimX, int dimY);


void FGP_TV(float *A, float lambda, int iter, float epsil, int methTV, int nonneg, int printM, int dimX, int dimY, int dimZ, float *D) 
{
        
    int ll, j, count;    
    float *D_old=NULL, *P1=NULL, *P2=NULL, *P1_old=NULL, *P2_old=NULL, *R1=NULL, *R2=NULL, tk, tkp1, re, re1;    
    
    //A  = (float *) mxGetData(prhs[0]); /*noisy image (2D/3D) */
    //lambda =  (float) mxGetScalar(prhs[1]); /* regularization parameters */
    //iter = 100; /* default iterations number */
    //epsil = 0.0001; /* default tolerance constant */
    //methTV = 0;  /* default isotropic TV penalty */
    //nonneg = 0;  /* nonnegativity (0 is OFF by default) */
    //printM = 0;  /* print information (0 is 0FF by default) */    
        
    
    /*output function value (last iteration) */
    // plhs[1] = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL);
    //float *funcvalA = (float *) mxGetData(plhs[1]);
    

    // if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) {mexErrMsgTxt("The input image must be in a single precision"); }
    
    /* Handling Matlab output data*/
	// dimX = dim_array[0]; dimY = dim_array[1]; dimZ = dim_array[2];
    
    tk = 1.0f;
    tkp1=1.0f;
    count = 0;
   // re_old = 0.0f;
    
		D_old = (float*) calloc (dimY*dimX,sizeof(float));      
		P1 = (float*) calloc (dimY*dimX,sizeof(float));      
		P2 = (float*) calloc (dimY*dimX,sizeof(float));      
		P1_old = (float*) calloc (dimY*dimX,sizeof(float));      
		P2_old = (float*) calloc (dimY*dimX,sizeof(float));  
		R1 = (float*) calloc (dimY*dimX,sizeof(float));      
		R2 = (float*) calloc (dimY*dimX,sizeof(float));      
        
        /* begin iterations */
        for(ll=0; ll<iter; ll++) {
            
            /* computing the gradient of the objective function */
            Obj_func2D(A, D, R1, R2, lambda, dimX, dimY);
            
            if (nonneg == 1) {
                /* apply nonnegativity */
                for(j=0; j<dimX*dimY*dimZ; j++)  {if (D[j] < 0.0f) D[j] = 0.0f;}
            }
            
            /*Taking a step towards minus of the gradient*/
            Grad_func2D(P1, P2, D, R1, R2, lambda, dimX, dimY);
            
            /* projection step */
            Proj_func2D(P1, P2, methTV, dimX, dimY);
            
            /*updating R and t*/
            tkp1 = (1.0f + sqrt(1.0f + 4.0f*tk*tk))*0.5f;
            Rupd_func2D(P1, P1_old, P2, P2_old, R1, R2, tkp1, tk, dimX, dimY);
            
            /* calculate norm */
            re = 0.0f; re1 = 0.0f;
            for(j=0; j<dimX*dimY*dimZ; j++)
            {
                re += pow(D[j] - D_old[j],2);
                re1 += pow(D[j],2);
            }
            re = sqrt(re)/sqrt(re1);
            if (re < epsil)  count++;
            if (count > 4) {
               // Obj_func_CALC2D(A, D, funcvalA, lambda, dimX, dimY);
                break; }
            
            /* check that the residual norm is decreasing */
//             if (ll > 2) {
//                 if (re > re_old) {
//                     Obj_func_CALC2D(A, D, funcvalA, lambda, dimX, dimY);
//                     break; }}
            //re_old = re;
            /*printf("%f %i %i \n", re, ll, count); */
            
            /*storing old values*/
            copyIm(D, D_old, dimX, dimY, dimZ);
            copyIm(P1, P1_old, dimX, dimY, dimZ);
            copyIm(P2, P2_old, dimX, dimY, dimZ);
            tk = tkp1;
            
            /* calculating the objective function value */
            //if (ll == (iter-1)) Obj_func_CALC2D(A, D, funcvalA, lambda, dimX, dimY);
        }
        if (nonneg == 1) {
            /* apply nonnegativity */
            for(j=0; j<dimX*dimY*dimZ; j++)  {if (D[j] < 0.0f) D[j] = 0.0f;}
        }
        
        // if (printM == 1) printf("FGP-TV iterations stopped at iteration %i with the function value %f \n", ll, funcvalA[0]);    
   
   free(D_old);free(P1);free(P2);free(R1);free(R2);free(P1_old);free(P2_old);
}

float Obj_func_CALC2D(float *A, float *D, float *funcvalA, float lambda, int dimX, int dimY)
{   
    int i,j;
    float f1, f2, val1, val2;
    
    /*data-related term */
    f1 = 0.0f;
    for(i=0; i<dimX*dimY; i++) f1 += pow(D[i] - A[i],2);    
    
    /*TV-related term */
    f2 = 0.0f;
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
            /* boundary conditions  */
            if (i == dimX-1) {val1 = 0.0f;} else {val1 = A[(i+1)*dimY + (j)] - A[(i)*dimY + (j)];}
            if (j == dimY-1) {val2 = 0.0f;} else {val2 = A[(i)*dimY + (j+1)] - A[(i)*dimY + (j)];}    
            f2 += sqrt(pow(val1,2) + pow(val2,2));
        }}  
    
    /* sum of two terms */
    funcvalA[0] = 0.5f*f1 + lambda*f2;     
    return *funcvalA;
}

float Obj_func2D(float *A, float *D, float *R1, float *R2, float lambda, int dimX, int dimY)
{
	float val1, val2;
	int i, j;
#pragma omp parallel for shared(A,D,R1,R2) private(i,j,val1,val2)
	for (i = 0; i<dimX; i++) {
		for (j = 0; j<dimY; j++) {
			/* boundary conditions  */
			if (i == 0) { val1 = 0.0f; }
			else { val1 = R1[(i - 1)*dimY + (j)]; }
			if (j == 0) { val2 = 0.0f; }
			else { val2 = R2[(i)*dimY + (j - 1)]; }
			D[(i)*dimY + (j)] = A[(i)*dimY + (j)] - lambda*(R1[(i)*dimY + (j)] + R2[(i)*dimY + (j)] - val1 - val2);
		}
	}
	return *D;
}
float Grad_func2D(float *P1, float *P2, float *D, float *R1, float *R2, float lambda, int dimX, int dimY)
{
	float val1, val2, multip;
	int i, j;
	multip = (1.0f / (8.0f*lambda));
#pragma omp parallel for shared(P1,P2,D,R1,R2,multip) private(i,j,val1,val2)
	for (i = 0; i<dimX; i++) {
		for (j = 0; j<dimY; j++) {
			/* boundary conditions */
			if (i == dimX - 1) val1 = 0.0f; else val1 = D[(i)*dimY + (j)] - D[(i + 1)*dimY + (j)];
			if (j == dimY - 1) val2 = 0.0f; else val2 = D[(i)*dimY + (j)] - D[(i)*dimY + (j + 1)];
			P1[(i)*dimY + (j)] = R1[(i)*dimY + (j)] + multip*val1;
			P2[(i)*dimY + (j)] = R2[(i)*dimY + (j)] + multip*val2;
		}
	}
	return 1;
}
float Proj_func2D(float *P1, float *P2, int methTV, int dimX, int dimY)
{
	float val1, val2, denom;
	int i, j;
	if (methTV == 0) {
		/* isotropic TV*/
#pragma omp parallel for shared(P1,P2) private(i,j,denom)
		for (i = 0; i<dimX; i++) {
			for (j = 0; j<dimY; j++) {
				denom = pow(P1[(i)*dimY + (j)], 2) + pow(P2[(i)*dimY + (j)], 2);
				if (denom > 1) {
					P1[(i)*dimY + (j)] = P1[(i)*dimY + (j)] / sqrt(denom);
					P2[(i)*dimY + (j)] = P2[(i)*dimY + (j)] / sqrt(denom);
				}
			}
		}
	}
	else {
		/* anisotropic TV*/
#pragma omp parallel for shared(P1,P2) private(i,j,val1,val2)
		for (i = 0; i<dimX; i++) {
			for (j = 0; j<dimY; j++) {
				val1 = fabs(P1[(i)*dimY + (j)]);
				val2 = fabs(P2[(i)*dimY + (j)]);
				if (val1 < 1.0f) { val1 = 1.0f; }
				if (val2 < 1.0f) { val2 = 1.0f; }
				P1[(i)*dimY + (j)] = P1[(i)*dimY + (j)] / val1;
				P2[(i)*dimY + (j)] = P2[(i)*dimY + (j)] / val2;
			}
		}
	}
	return 1;
}
float Rupd_func2D(float *P1, float *P1_old, float *P2, float *P2_old, float *R1, float *R2, float tkp1, float tk, int dimX, int dimY)
{
	int i, j;
	float multip;
	multip = ((tk - 1.0f) / tkp1);
#pragma omp parallel for shared(P1,P2,P1_old,P2_old,R1,R2,multip) private(i,j)
	for (i = 0; i<dimX; i++) {
		for (j = 0; j<dimY; j++) {
			R1[(i)*dimY + (j)] = P1[(i)*dimY + (j)] + multip*(P1[(i)*dimY + (j)] - P1_old[(i)*dimY + (j)]);
			R2[(i)*dimY + (j)] = P2[(i)*dimY + (j)] + multip*(P2[(i)*dimY + (j)] - P2_old[(i)*dimY + (j)]);
		}
	}
	return 1;
}

/* General Functions */
/*****************************************************************/
/* Copy Image */
float copyIm(float *A, float *B, int dimX, int dimY, int dimZ)
{
    int j;
#pragma omp parallel for shared(A, B) private(j)
    for(j=0; j<dimX*dimY*dimZ; j++)  B[j] = A[j];
    return *B;
}
