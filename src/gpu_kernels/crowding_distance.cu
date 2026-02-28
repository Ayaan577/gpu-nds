#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <stdio.h>

#define MAX_FRONT_SIZE 1024
#define THREADS_PER_BLOCK 128

__device__ void block_sort(
    float* s_obj, int* s_idx, int fs, int tid
) {
    for (int phase = 0; phase < fs; phase++) {
        int parity = phase & 1;
        for (int t = tid; t < fs - 1; t += blockDim.x) {
            if ((t & 1) == parity) {
                if (s_obj[t] > s_obj[t + 1]) {
                    float tmp_f  = s_obj[t];
                    s_obj[t]     = s_obj[t + 1];
                    s_obj[t + 1] = tmp_f;
                    int tmp_i    = s_idx[t];
                    s_idx[t]     = s_idx[t + 1];
                    s_idx[t + 1] = tmp_i;
                }
            }
        }
        __syncthreads();
    }
}

/*
 * Sort s_idx ascending by global index value.
 * Makes atomicAdd collection deterministic (matches CPU 0..N-1 scan).
 */
__device__ void sort_idx_ascending(
    int* s_idx, int fs, int tid
) {
    for (int phase = 0; phase < fs; phase++) {
        int parity = phase & 1;
        for (int t = tid; t < fs - 1; t += blockDim.x) {
            if ((t & 1) == parity) {
                if (s_idx[t] > s_idx[t + 1]) {
                    int tmp = s_idx[t];
                    s_idx[t]     = s_idx[t + 1];
                    s_idx[t + 1] = tmp;
                }
            }
        }
        __syncthreads();
    }
}

extern "C" __global__
void crowding_distance_kernel(
    const float* __restrict__ F,
    const int*   __restrict__ rank,
    float*                    cd,
    const int                 N,
    const int                 M,
    const int                 n_fronts
) {
    const int front_id = blockIdx.x;
    const int tid      = threadIdx.x;
    if (front_id >= n_fronts) return;

    __shared__ int   s_idx[MAX_FRONT_SIZE];
    __shared__ float s_obj[MAX_FRONT_SIZE];
    __shared__ int   s_size;

    if (tid == 0) s_size = 0;
    __syncthreads();

    for (int i = tid; i < N; i += blockDim.x) {
        if (rank[i] == front_id) {
            int pos = atomicAdd(&s_size, 1);
            if (pos < MAX_FRONT_SIZE)
                s_idx[pos] = i;
        }
    }
    __syncthreads();

    int fs = (s_size < MAX_FRONT_SIZE)
             ? s_size : MAX_FRONT_SIZE;

    /* Deterministic ordering — matches CPU reference */
    sort_idx_ascending(s_idx, fs, tid);

    if (fs <= 2) {
        for (int t = tid; t < fs; t += blockDim.x)
            cd[s_idx[t]] = FLT_MAX;
        return;
    }

    for (int t = tid; t < fs; t += blockDim.x)
        cd[s_idx[t]] = 0.0f;
    __syncthreads();

    for (int m = 0; m < M; m++) {
        for (int t = tid; t < fs; t += blockDim.x)
            s_obj[t] = F[s_idx[t] * M + m];
        __syncthreads();

        block_sort(s_obj, s_idx, fs, tid);

        if (tid == 0) {
            cd[s_idx[0]]      = FLT_MAX;
            cd[s_idx[fs - 1]] = FLT_MAX;
        }
        __syncthreads();

        float f_range = s_obj[fs - 1] - s_obj[0];
        if (f_range < 1e-10f) { __syncthreads(); continue; }

        for (int t = tid + 1; t < fs - 1; t += blockDim.x) {
            int   gi   = s_idx[t];
            float prev = s_obj[t - 1];
            float next = s_obj[t + 1];
            if (cd[gi] < FLT_MAX) {
                float contrib = (next - prev) / f_range;
                atomicAdd(&cd[gi], contrib);
            }
        }
        __syncthreads();
    }
}

extern "C" __global__
void max_rank_kernel(
    const int* rank, int* result, int N
) {
    __shared__ int s_max;
    if (threadIdx.x == 0) s_max = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        atomicMax(&s_max, rank[i]);
    __syncthreads();
    if (threadIdx.x == 0) atomicMax(result, s_max);
}
