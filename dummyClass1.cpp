
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
using namespace std;

#pragma omp declare target
class Dummy{

public:

/**
    double *initA(int n){
         double *A; 
         A = (double*)malloc(sizeof(double)*n*n);
         return A;
    }

    double *initB(int n){
         double *B; 
         B = (double*)malloc(sizeof(double)*n*n);
         return B;
    }

    double *initC(int n){
         double *C; 
         C = (double*)malloc(sizeof(double)*n*n);
         return C;
    }
};

*/


int ne=512;
int nv=513;
int tail=-1;

struct Edge
{
   int  indices[512];
   Edge edgelist[nv];
   
   //Edge():ne(512),nv(513),tail(-1){}
}



inline void nbrscan_edges()
{
       Edge e1;
       int edges[ne];

#pragma omp target teams distribute parallel for map(to:e1.indices[ne+1], e1.edgelist[nv]) \
        map(from:edges[nv])
#pragma omp parallel for
   for (int i = 0; i < ne; i++){
#pragma omp task
     for (int e = e1.indices[i]; e < e1.indices[i+1]; e++){
         Edge const& edge = e1.edgelist[e];
         edges[e] = tail;
         } 
      }
}

};

#pragma omp end declare target
int main(){

    Dummy gemm1;

 //   double *A, *B, *C;

//    int i, n;
    
//    n=512;
    

//   A = gemm1.initA(n);
//   B = gemm1.initB(n);
//   C = gemm1.initC(n);

/**    for(i=0; i<n*n; i++) { A[i] = rand()/RAND_MAX; B[i] = rand()/RAND_MAX;}
	
	
       int j, k;
       #pragma omp target teams distribute parallel for \
        map(to:B[:n*n],A[:n*n]) map(from:C[:n*n])
       //#pragma omp target teams map(to:n,B[:n*n],A[:n*n]) map(from:C[:n*n])
       //{
	//#pragma omp distribute parallel for
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
*/	//}

    gemm1.nbrscan_edges();	    
    printf("Yeah!!!");

    return 0;
 } 





                                                                                                                                                                                  
