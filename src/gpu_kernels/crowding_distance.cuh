#pragma once
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

void launch_crowding_distance(
    const float* d_F,
    const int*   d_rank,
    float*       d_cd,
    int          N,
    int          M,
    cudaStream_t stream
);

void compute_crowding_distance(
    const float* h_F,
    const int*   h_rank,
    float*       h_cd,
    int          N,
    int          M
);

#ifdef __cplusplus
}
#endif
