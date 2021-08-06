#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <random>
#include <utility>
#include <cassert>
#include <omp.h>
using namespace std;
#ifndef NV
#define NV (200)
#endif
#ifndef NE
#define NE (5000)
#endif


struct Edge
{   
   long tail_;
   double weight_;
      
   Edge(): tail_(-1), weight_(0.0) {}
};
template<typename T, typename G = std::default_random_engine>
T genRandom(T lo, T hi)
{
    thread_local static G gen(std::random_device{}());
    using Dist = typename std::conditional
        <
        std::is_integral<T>::value
        , std::uniform_int_distribution<T>
        , std::uniform_real_distribution<T>
        >::type;
    
    thread_local static Dist utd {};
    return utd(gen, typename Dist::param_type{lo, hi});
}

class Dummy
{
  public:
  Dummy(long n, long m): n_(n), m_(m)
  {
    assert(m_ > n_); // #vertices < #edges
    elist_ = new Edge[m_];
    eidx_ = new long[n_+1];
    vdegree_ = new long[m_];
    long d = m_ / n_;
    long last = d + (m_ % n_);
    long idx = 0;
    for (long k = 0; k < n_+1; k++)
    {
      eidx_[k] = idx;
      if (k == (n_-1))
        idx += last;
      else
        idx += d;
    }
    eidx_[n_] = m_;
    for (long i = 0; i < n_; i++)
    {
      for (long j = eidx_[i]; j < eidx_[i+1]; j++)
      {
        Edge& edge = elist_[j];
        edge.tail_ = genRandom<long>(0,n_);
        edge.weight_ = genRandom<double>(0.0,1.0);
      }
    }
  }

 ~Dummy()
  {
   delete[] eidx_;
   delete[] elist_;
   delete[] vdegree_;
  }
  
 
 Dummy(const Dummy &d) 
{
   n_ = d.n_;
   m_ = d.m_;
   memcpy(elist_, d.elist_, sizeof(Edge)*m_); 
   memcpy(eidx_, d.eidx_, sizeof(long)*(n_+1)); 
}

void print_edges() const
{
   std::cout << "Edges (u ---- v) with weights:" << std::endl; 
   for (long i = 0; i < n_; i++)
   {
     for (long j = eidx_[i]; j < eidx_[i+1]; j++)
     {
       Edge const& edge = elist_[j];
       std::cout << i << "----" << edge.tail_ << " (" << edge.weight_ << ")" << std::endl;
     }
   }
}
    // offload kernel
inline void nbrmax()
{
#ifdef USE_OMP_ACCELERATOR
#pragma omp target teams distribute parallel for \
        map(to:eidx_[0:n_+1], elist_[0:m_]) \
        map(from:vdegree_[0:n_])
#else
#pragma omp parallel for
#endif
     
     for (long i = 0; i < n_; i++)
     {
     /**  for (long e = eidx_[i];e < eidx_[i+1]; e++)
       {
          Edge const& edge = elist_[e];
          celist_[e] = edge.tail_;
       }
     */

     double wmax = -1.0;                             
     for (long e = eidx_[i]; e < eidx_[i+1]; e++)                             
     {                                     
	     Edge const& edge = elist_[e];                                     
	     if (wmax < edge.weight_)                                             
	       wmax = edge.weight_;                             
     }                             
	     vdegree_[i] = wmax;
     }
     //std::cout << "Scan neighborhood and copy" << std::endl;

     
}

private:
  long n_, m_;
  Edge *elist_;
  long *eidx_, *vdegree_;
};

int main(int argc, char *argv[])
{
  long m, n;
  if (argc == 2)
  {
    n = atol(argv[1]);
    m = m*2;
  }
  else if (argc == 3)
  {
    n = atol(argv[1]);
    m = atol(argv[2]);
  }
  else
  {
    n = NV;
    m = NE;
  }

  std::cout << "#Vertices: " << n << ", #Edges: " << m << std::endl;
  Dummy d(n, m);
  //d.print_edges();
  d.nbrmax();
  
  return 0;
}









