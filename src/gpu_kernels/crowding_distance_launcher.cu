#include "crowding_distance.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

#define THREADS_PER_BLOCK 128

/* Forward declarations */
extern "C" {
__global__ void crowding_distance_kernel(
    const float*, const int*, float*, int, int, int);
__global__ void max_rank_kernel(
    const int*, int*, int);
}

void launch_crowding_distance(
    const float* d_F,
    const int*   d_rank,
    float*       d_cd,
    int          N,
    int          M,
    cudaStream_t stream
) {
    /* Step 1: Find n_fronts = max(rank) + 1 */
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

    /* Step 2: Zero-initialise output */
    cudaMemsetAsync(d_cd, 0, N * sizeof(float), stream);

    /* Step 3: Launch one block per front */
    crowding_distance_kernel<<<
        n_fronts, THREADS_PER_BLOCK, 0, stream>>>(
        d_F, d_rank, d_cd, N, M, n_fronts);

    cudaStreamSynchronize(stream);
}
