/**
 * GPU-NDS CUDA Kernel Headers
 */

#ifndef GPU_NDS_KERNELS_CUH
#define GPU_NDS_KERNELS_CUH

/**
 * KERNEL 1: Compute sum of objectives for each solution.
 * Launch: N threads (1D grid).
 */
__global__ void compute_objective_sums(
    const float* __restrict__ objectives,
    float* __restrict__ sums,
    const int N, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float s = 0.0f;
    for (int m = 0; m < M; m++)
        s += objectives[i * M + m];
    sums[i] = s;
}

/**
 * KERNEL 2: Tiled dominance check with shared memory.
 * Launch: 2D grid of (TILE, TILE) blocks.
 */
__global__ void tiled_dominance_check_shared(
    const float* __restrict__ objectives,
    const float* __restrict__ sums,
    int* __restrict__ dom_count,
    const int N, const int M)
{
    extern __shared__ float smem[];
    const int TILE = blockDim.x;
    int tile_row = blockIdx.x;
    int tile_col = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float* tile_A = smem;
    float* tile_B = smem + TILE * M;

    int gi = tile_row * TILE + tx;
    int gj = tile_col * TILE + ty;

    if (gi < N)
        for (int m = ty; m < M; m += TILE)
            tile_A[tx * M + m] = objectives[gi * M + m];
    if (gj < N)
        for (int m = tx; m < M; m += TILE)
            tile_B[ty * M + m] = objectives[gj * M + m];
    __syncthreads();

    if (gi >= N || gj >= N || gi == gj) return;
    if (sums[gi] > sums[gj]) return;

    bool all_leq = true, any_lt = false;
    for (int m = 0; m < M; m++) {
        float ai = tile_A[tx * M + m];
        float bj = tile_B[ty * M + m];
        if (ai > bj) { all_leq = false; break; }
        if (ai < bj) any_lt = true;
    }
    if (all_leq && any_lt)
        atomicAdd(&dom_count[gj], 1);
}

/**
 * KERNEL 3: Front assignment round.
 */
__global__ void front_assign(
    const int* __restrict__ dom_count,
    int* __restrict__ front_rank,
    int* __restrict__ newly_assigned,
    const int N, const int current_front)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (front_rank[i] == -1 && dom_count[i] == 0) {
        front_rank[i] = current_front;
        newly_assigned[i] = 1;
    }
}

/**
 * KERNEL 4: Decrement domination count for remaining solutions.
 */
__global__ void decrement_dom_count(
    const float* __restrict__ objectives,
    const float* __restrict__ sums,
    const int* __restrict__ newly_assigned,
    const int* __restrict__ front_rank,
    int* __restrict__ dom_count,
    const int N, const int M)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N || front_rank[j] != -1) return;

    int dec = 0;
    for (int i = 0; i < N; i++) {
        if (newly_assigned[i] == 0) continue;
        if (sums[i] > sums[j]) continue;
        bool all_leq = true, any_lt = false;
        for (int m = 0; m < M; m++) {
            float ai = objectives[i * M + m];
            float bj = objectives[j * M + m];
            if (ai > bj) { all_leq = false; break; }
            if (ai < bj) any_lt = true;
        }
        if (all_leq && any_lt) dec++;
    }
    if (dec > 0) atomicSub(&dom_count[j], dec);
}

#endif // GPU_NDS_KERNELS_CUH
