#include "crowding_distance.cuh"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

#define THREADS_PER_BLOCK 128

extern "C" __global__ void crowding_distance_kernel(
    const float*, const int*, float*, int, int, int);
extern "C" __global__ void max_rank_kernel(const int*, int*, int);

void launch_crowding_distance(
    const float* d_F,
    const int*   d_rank,
    float*       d_cd,
    int          N,
    int          M,
    cudaStream_t stream
) {
    int* d_max_rank;
    cudaMalloc(&d_max_rank, sizeof(int));
    cudaMemset(d_max_rank, 0, sizeof(int));

    max_rank_kernel<<<1, THREADS_PER_BLOCK, 0, stream>>>(
        d_rank, d_max_rank, N);

    int h_max_rank = 0;
    cudaMemcpyAsync(&h_max_rank, d_max_rank, sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_max_rank);

    int n_fronts = h_max_rank + 1;
    cudaMemsetAsync(d_cd, 0, N * sizeof(float), stream);

    crowding_distance_kernel<<<
        n_fronts, THREADS_PER_BLOCK, 0, stream>>>(
        d_F, d_rank, d_cd, N, M, n_fronts);

    cudaStreamSynchronize(stream);
}

/* Host wrapper for tests: allocates device memory, runs kernel,
   copies result back. */
void compute_crowding_distance(
    const float* h_F,
    const int*   h_rank,
    float*       h_cd,
    int          N,
    int          M
) {
    float* d_F;
    int*   d_rank;
    float* d_cd;

    cudaMalloc(&d_F,    N * M * sizeof(float));
    cudaMalloc(&d_rank, N     * sizeof(int));
    cudaMalloc(&d_cd,   N     * sizeof(float));

    cudaMemcpy(d_F,    h_F,    N*M*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rank, h_rank, N*sizeof(int),     cudaMemcpyHostToDevice);

    launch_crowding_distance(d_F, d_rank, d_cd, N, M, 0);

    cudaMemcpy(h_cd, d_cd, N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_F);
    cudaFree(d_rank);
    cudaFree(d_cd);
}
