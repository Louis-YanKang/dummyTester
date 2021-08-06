#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
using namespace std;
class Dummy{
        public:
        Dummy(int n): n_(n)
        {
            A = (double*)malloc(sizeof(double)*n*n);
            B = (double*)malloc(sizeof(double)*n*n);
            C = (double*)malloc(sizeof(double)*n*n);
        }
        ~Dummy()
        {
            free(A);
            free(B);
            free(C);
        }
        Dummy(const Dummy &d)
        {
            n_ = d.n_;
            memcpy(A, d.A, sizeof(double)*n_*n_);
            memcpy(B, d.B, sizeof(double)*n_*n_);
            memcpy(C, d.C, sizeof(double)*n_*n_);
        }
        double *A, *B, *C;
        private:
            int n_;
};
int main(){
    int i, n;
    n=512;
    Dummy gemm1(n);
    for(i=0; i<n*n; i++) { gemm1.A[i] = rand()/RAND_MAX; gemm1.B[i] = rand()/RAND_MAX;}
       int j, k;
       #pragma omp target teams distribute parallel for \
       map(to:gemm1) map(to:gemm1.B[:n*n],gemm1.A[:n*n]) map(from:gemm1.C[:n*n])
       //{
            for (i = 0; i < n; i++) {
                for (j = 0; j < n; j++) {
                    double dot  = 0.0;
                    for (k = 0; k < n; k++) {
                        dot += gemm1.A[i*n+k]*gemm1.B[k*n+j];
                    }
                    gemm1.C[i*n+j ] = dot;
                }
            }
    //}
    printf("Yeah!!!");
    return 0;
 }
