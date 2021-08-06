#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <random>
#include <utility>
#include <cassert>
#include <omp.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <atomic>
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
    explicit Dummy(long n, long m): n_(n), m_(m)
    {
      assert(m_ > n_); // #vertices < #edges
      elist_ = new Edge[m_];
      eidx_ = new long[n_+1];
      celist_ = new long[m_];
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
      delete[] celist_;
    }
     
    long& operator[](long idx) 
    { return celist_[idx]; }

    // thwart default copy ctors
    Dummy( Dummy const& ) = delete;
    void operator=( Dummy const& ) = delete;
    
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
    inline void nbrscan()
    {
#ifdef USE_OMP_ACCELERATOR
#pragma omp target teams distribute parallel for \
      map(to:eidx_[0:n_+1], elist_[0:m_]) \
      map(tofrom:celist_[0:m_])
#else
#pragma omp parallel for
#endif
      for (long i = 0; i < n_; i++)
      {
        for (long e = eidx_[i]; e < eidx_[i+1]; e++)
        {
          Edge const& edge = elist_[e];
          celist_[e] = edge.tail_;
        }
      }
    }
    
    long get_n() const { return n_; }
    long get_m() const { return m_; }
    
    long n_, m_;
    Edge *elist_;
    long *eidx_, *celist_;
};

void nbrscan_free(Dummy* d)
{
  const long n_ = d->get_n();
  const long m_ = d->get_m();
#ifdef USE_OMP_ACCELERATOR
#pragma omp target teams distribute parallel for \
  map(to: d->eidx_[0:n_+1], d->elist_[0:m_]) \
  map(from: d->celist_[0:m_])
#else
#pragma omp parallel for
#endif
  for (long i = 0; i < n_; i++)
  {
    for (long e = d->eidx_[i]; e < d->eidx_[i+1]; e++)
    {
      Edge const& edge = d->elist_[e];
      d->operator[](e) = edge.tail_;
    }
  }
}

/**
 bool compare_and_swap(long* x, long old_val, long new_val) {
    if (x == old_val) {
      x = new_val;
      return true;
    }
    return false;
  }
*/


__device__ int myAtomicAdd(int *address, int incr)
{
    // Create an initial guess for the value stored at *address.
    int guess = *address;
    int oldValue = atomicCAS(address, guess, guess + incr);

    // Loop while the guess is incorrect.
    while (oldValue != guess)
    {
        guess = oldValue;
        oldValue = atomicCAS(address, guess, guess + incr);
    }

    return oldValue;
}

__global__ void kernel(int *sharedInteger)
{
    myAtomicAdd(sharedInteger, 1);
}


int atomicCAS(int* atarr,int old_val,int new_val);

void nbrscan_atomic(Dummy* d)
{
  const int n_ = d->get_n();
  const int m_ = d->get_m();
  int* atarr[n_]; 
    //= new int[n_];
  //int old_val=10;
  //int new_val=100;
  int p = 20;

#ifdef USE_OMP_ACCELERATOR
#pragma omp target teams distribute parallel for \
  map(to: d->eidx_[0:n_+1], d->elist_[0:m_]) \
  map(from: d->celist_[0:m_])
#else
#pragma omp parallel for
#endif
  for (int i = 0; i < n_; i++)
  {
    for (int e = d->eidx_[i]; e < d->eidx_[i+1]; e++)
    {
      Edge const& edge = d->elist_[e];
      
      kernel(atarr[i]);
      //myAtomicAdd(atarr[i], p);
     //atarr[i] += 1;

    }
  }
    std::cout <<"---value update-----" << atarr[n_-1] <<endl;
}


int main(int argc, char *argv[])
{
  int m, n;
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
  Dummy *d = new Dummy(n, m);
  
  //d.print_edges();
  // d.nbrscan();
 // nbrscan_free(d);
 nbrscan_atomic(d);

  std::cout << "Scan neighborhood and copy" << std::endl;
  
  return 0;
}
