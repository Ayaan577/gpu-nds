/**
 * Naive GPU Non-Dominated Sort — Baseline
 *
 * All N² pairs checked in parallel with no tiling or pruning.
 * Compile: nvcc -O3 naive_gpu.cu -o naive_gpu
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include "utils.cuh"

__global__ void naive_dominance_check(
    const float* __restrict__ objectives,
    int* __restrict__ dom_count,
    const int N, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= N || j >= N || i == j) return;

    bool all_leq = true, any_lt = false;
    for (int m = 0; m < M; m++) {
        float ai = objectives[i * M + m];
        float bj = objectives[j * M + m];
        if (ai > bj) { all_leq = false; break; }
        if (ai < bj) any_lt = true;
    }
    if (all_leq && any_lt)
        atomicAdd(&dom_count[j], 1);
}
