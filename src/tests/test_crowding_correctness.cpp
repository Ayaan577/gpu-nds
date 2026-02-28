#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "../gpu_kernels/crowding_distance.cuh"

/* Reference CPU crowding distance */
void reference_crowding_cpu(
    const float* F, const int* rank,
    float* cd, int N, int M
) {
    for (int i = 0; i < N; i++) cd[i] = 0.0f;

    int max_rank = 0;
    for (int i = 0; i < N; i++)
        if (rank[i] > max_rank) max_rank = rank[i];

    int* front_idx = (int*)malloc(N * sizeof(int));
    float* obj_col = (float*)malloc(N * sizeof(float));

    for (int k = 0; k <= max_rank; k++) {
        int fs = 0;
        for (int i = 0; i < N; i++)
            if (rank[i] == k) front_idx[fs++] = i;

        if (fs <= 2) {
            for (int j = 0; j < fs; j++)
                cd[front_idx[j]] = FLT_MAX;
            continue;
        }

        for (int m = 0; m < M; m++) {
            /* Load objective m */
            for (int j = 0; j < fs; j++)
                obj_col[j] = F[front_idx[j] * M + m];

            /* Insertion sort (correct, simple) */
            for (int a = 1; a < fs; a++) {
                float kv = obj_col[a];
                int   ki = front_idx[a];
                int b = a - 1;
                while (b >= 0 && obj_col[b] > kv) {
                    obj_col[b+1]    = obj_col[b];
                    front_idx[b+1]  = front_idx[b];
                    b--;
                }
                obj_col[b+1]   = kv;
                front_idx[b+1] = ki;
            }

            cd[front_idx[0]]      = FLT_MAX;
            cd[front_idx[fs - 1]] = FLT_MAX;

            float f_range = obj_col[fs-1] - obj_col[0];
            if (f_range < 1e-10f) continue;

            for (int j = 1; j < fs - 1; j++) {
                if (cd[front_idx[j]] < FLT_MAX)
                    cd[front_idx[j]] +=
                        (obj_col[j+1] - obj_col[j-1]) 
                        / f_range;
            }
        }
    }
    free(front_idx);
    free(obj_col);
}

// Ensure the wrapper is visible
extern "C" void compute_crowding_distance(
    const float* h_F,
    const int*   h_rank,
    float*       h_cd,
    int          N,
    int          M
);

int run_test(int N, int M, int seed) {
    srand(seed);
    float* F    = (float*)malloc(N * M * sizeof(float));
    int*   rank = (int*)  malloc(N     * sizeof(int));
    float* cd_ref = (float*)malloc(N   * sizeof(float));
    float* cd_gpu = (float*)malloc(N   * sizeof(float));

    /* Random objectives in [0,1] */
    for (int i = 0; i < N * M; i++)
        F[i] = (float)rand() / RAND_MAX;

    /* Assign random ranks 0..sqrt(N) */
    int max_r = (int)sqrt((double)N);
    for (int i = 0; i < N; i++)
        rank[i] = rand() % (max_r + 1);

    /* Reference CPU */
    reference_crowding_cpu(F, rank, cd_ref, N, M);

    /* GPU kernel via wrapper */
    compute_crowding_distance(F, rank, cd_gpu, N, M);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error after compute_crowding_distance: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    /* Compare */
    int pass = 1;
    float max_err = 0.0f;
    for (int i = 0; i < N; i++) {
        int ref_inf = (cd_ref[i] >= FLT_MAX / 2);
        int gpu_inf = (cd_gpu[i] >= FLT_MAX / 2);
        if (ref_inf != gpu_inf) { 
            printf("Error at i=%d: ref_inf=%d, gpu_inf=%d, cd_ref=%f, cd_gpu=%f\n", 
                   i, ref_inf, gpu_inf, cd_ref[i], cd_gpu[i]);
            pass = 0; 
            break; 
        }
        if (!ref_inf) {
            float err = fabsf(cd_gpu[i] - cd_ref[i])
                        / (fabsf(cd_ref[i]) + 1e-10f);
            if (err > max_err) max_err = err;
            if (err > 0.01f) { 
                printf("Error at i=%d: err=%f, cd_ref=%f, cd_gpu=%f\n", 
                       i, err, cd_ref[i], cd_gpu[i]);
                pass = 0; 
                break; 
            }
        }
    }

    printf("N=%5d M=%2d seed=%3d: %s (max_rel_err=%.5f)\n",
           N, M, seed, pass ? "PASS" : "FAIL", max_err);

    free(F); free(rank); free(cd_ref); free(cd_gpu);
    return pass;
}

int main() {
    printf("Crowding distance correctness tests\n");
    printf("=====================================\n");
    int all_pass = 1;
    int configs[][2] = {
        {50,2},{200,3},{500,3},{1000,5},
        {2000,5},{2000,10},{5000,3}
    };
    int n_configs = sizeof(configs) / sizeof(configs[0]);
    for (int c = 0; c < n_configs; c++)
        for (int seed = 0; seed < 3; seed++)
            all_pass &= run_test(
                configs[c][0], configs[c][1], seed);
    printf("=====================================\n");
    printf("%s\n", all_pass ? "ALL TESTS PASSED" 
                             : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
