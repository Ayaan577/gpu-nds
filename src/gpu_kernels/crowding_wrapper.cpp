/* Thin C wrapper so Python ctypes can call it */

#include "crowding_distance.cuh"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef _WIN32
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API
#endif

extern "C" {

/*
 * Full pipeline: allocate device memory, run kernel, 
 * copy result back.
 *
 * Args (all host pointers):
 *   h_F    : float32 array (N x M), row-major
 *   h_rank : int32   array (N,)
 *   h_cd   : float32 array (N,) OUTPUT — caller allocates
 *   N, M   : ints
 */
EXPORT_API void compute_crowding_distance(
    const float* h_F,
    const int*   h_rank,
    float*       h_cd,
    int          N,
    int          M
) {
    float* d_F;
    int*   d_rank;
    float* d_cd;

    cudaError_t err = cudaMalloc(&d_F, N * M * sizeof(float));
    if (err != cudaSuccess) {
        printf("cudaMalloc error: %d\n", (int)err);
    }
    cudaMalloc(&d_rank, N     * sizeof(int));
    cudaMalloc(&d_cd,   N     * sizeof(float));

    cudaMemcpy(d_F,    h_F,    N*M*sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_rank, h_rank, N*sizeof(int),
               cudaMemcpyHostToDevice);

    launch_crowding_distance(d_F, d_rank, d_cd, N, M, 0);

    cudaMemcpy(h_cd, d_cd, N*sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaFree(d_F);
    cudaFree(d_rank);
    cudaFree(d_cd);
}

} // extern "C"
