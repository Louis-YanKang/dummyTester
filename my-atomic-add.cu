#include "../common/common.h"
#include <stdio.h>
#include <stdlib.h>

/**
 * This example illustrates implementation of custom atomic operations using
 * CUDA's built-in atomicCAS function to implement atomic signed 32-bit integer
 * addition.
 **/

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
