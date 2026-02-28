#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "../gpu_kernels/crowding_distance.cuh"

void reference_crowding_cpu(
    const float* F, const int* rank,
    float* cd, int N, int M
) {
    for (int i = 0; i < N; i++) cd[i] = 0.0f;
    int max_rank = 0;
    for (int i = 0; i < N; i++)
        if (rank[i] > max_rank) max_rank = rank[i];

    int*   fidx = (int*)  malloc(N * sizeof(int));
    float* fobj = (float*)malloc(N * sizeof(float));

    for (int k = 0; k <= max_rank; k++) {
        int fs = 0;
        for (int i = 0; i < N; i++)
            if (rank[i] == k) fidx[fs++] = i;
        if (fs <= 2) {
            for (int j = 0; j < fs; j++)
                cd[fidx[j]] = FLT_MAX;
            continue;
        }
        for (int m = 0; m < M; m++) {
            for (int j = 0; j < fs; j++)
                fobj[j] = F[fidx[j] * M + m];
            // insertion sort
            for (int a = 1; a < fs; a++) {
                float kv = fobj[a];
                int   ki = fidx[a];
                int   b  = a - 1;
                while (b >= 0 && fobj[b] > kv) {
                    fobj[b+1] = fobj[b];
                    fidx[b+1] = fidx[b];
                    b--;
                }
                fobj[b+1] = kv;
                fidx[b+1] = ki;
            }
            cd[fidx[0]]      = FLT_MAX;
            cd[fidx[fs - 1]] = FLT_MAX;
            float fr = fobj[fs-1] - fobj[0];
            if (fr < 1e-10f) continue;
            for (int j = 1; j < fs - 1; j++)
                if (cd[fidx[j]] < FLT_MAX)
                    cd[fidx[j]] +=
                        (fobj[j+1] - fobj[j-1]) / fr;
        }
    }
    free(fidx); free(fobj);
}

int run_test(int N, int M, int seed) {
    srand(seed);
    float* F     = (float*)malloc(N*M*sizeof(float));
    int*   rank  = (int*)  malloc(N  *sizeof(int));
    float* cd_ref = (float*)malloc(N *sizeof(float));
    float* cd_gpu = (float*)malloc(N *sizeof(float));

    for (int i = 0; i < N*M; i++)
        F[i] = (float)rand()/RAND_MAX;
    int max_r = (int)sqrt((double)N);
    for (int i = 0; i < N; i++)
        rank[i] = rand() % (max_r + 1);

    reference_crowding_cpu(F, rank, cd_ref, N, M);
    compute_crowding_distance(F, rank, cd_gpu, N, M);

    int   pass    = 1;
    float max_err = 0.0f;
    for (int i = 0; i < N; i++) {
        int ri = (cd_ref[i] >= FLT_MAX/2);
        int gi = (cd_gpu[i] >= FLT_MAX/2);
        if (ri != gi) { pass = 0; break; }
        if (!ri) {
            float err = fabsf(cd_gpu[i]-cd_ref[i])
                        / (fabsf(cd_ref[i])+1e-10f);
            if (err > max_err) max_err = err;
            if (err > 0.01f)  { pass = 0; break; }
        }
    }
    printf("N=%5d M=%2d seed=%d: %s (max_err=%.5f)\n",
           N, M, seed,
           pass ? "PASS" : "FAIL", max_err);
    free(F); free(rank); free(cd_ref); free(cd_gpu);
    return pass;
}

int main() {
    printf("Crowding distance correctness\n");
    printf("=====================================\n");
    int cfgs[][2] = {
        {50,2},{200,3},{500,3},{1000,5},
        {2000,5},{2000,10},{5000,3}
    };
    int nc = sizeof(cfgs)/sizeof(cfgs[0]);
    int ok = 1;
    for (int c = 0; c < nc; c++)
        for (int s = 0; s < 3; s++)
            ok &= run_test(cfgs[c][0], cfgs[c][1], s);
    printf("=====================================\n");
    printf("%s\n", ok ? "ALL PASSED" : "SOME FAILED");
    return ok ? 0 : 1;
}
