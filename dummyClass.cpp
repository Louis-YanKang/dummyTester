
#include<iostream>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
using namespace std;

class Dummy{

public:
    
    
    int n = 512;
    	
    double *A = (double*)malloc(sizeof(double)*n*n);
    double *B = (double*)malloc(sizeof(double)*n*n);
    double *C = (double*)malloc(sizeof(double)*n*n);

    void gemm_omp(double *A, double *B, double *C, int n) 
    {   
      
           int i, j, k;
	#pragma omp target teams distribute parallel for  map(to:n,B[:n*n],A[:n*n]) map(from:C[:n*n])
	//{
            for (i = 0; i < n; i++) { 
                for (j = 0; j < n; j++) {
                    double dot  = 0;
                    for (k = 0; k < n; k++) {
                        dot += A[i*n+k]*B[k*n+j];
                    } 
                    C[i*n+j ] = dot;
                }
            }
         //}   
       
    }

};


int main(){

    Dummy gemm1;

    double *A = gemm1.A;
    double *B = gemm1.B;
    double *C = gemm1.C;
    double dtime;

    int i;
    
    double n = gemm1.n;


    for(i=0; i<n*n; i++) { 
	A[i] = rand()/RAND_MAX; 
	B[i] = rand()/RAND_MAX;
    }


    dtime = omp_get_wtime();
    gemm1.gemm_omp(A,B,C, n);
    dtime = omp_get_wtime() - dtime;
    printf("%f\n", dtime);

    return 0;
}
