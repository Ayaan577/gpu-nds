#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../gpu_kernels/crowding_distance.cuh"

int main() {
    int cfgs[][2] = {
        {500,3},{1000,3},{2000,3},{5000,3},
        {2000,5},{2000,10},{10000,5}
    };
    int nc = sizeof(cfgs)/sizeof(cfgs[0]);

    printf("%-8s %-4s %-12s\n","N","M","GPU_ms");
    printf("------------------------\n");

    for (int c = 0; c < nc; c++) {
        int N = cfgs[c][0], M = cfgs[c][1];
        srand(42);
        float* h_F    = (float*)malloc(N*M*sizeof(float));
        int*   h_rank = (int*)  malloc(N  *sizeof(int));
        float* h_cd   = (float*)malloc(N  *sizeof(float));
        for (int i = 0; i < N*M; i++)
            h_F[i] = (float)rand()/RAND_MAX;
        int mr = (int)sqrt((double)N);
        for (int i = 0; i < N; i++)
            h_rank[i] = rand() % (mr+1);

        float *d_F, *d_cd; int *d_rank;
        cudaMalloc(&d_F,    N*M*sizeof(float));
        cudaMalloc(&d_rank, N  *sizeof(int));
        cudaMalloc(&d_cd,   N  *sizeof(float));
        cudaMemcpy(d_F,    h_F,    N*M*sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_rank, h_rank, N  *sizeof(int),
                   cudaMemcpyHostToDevice);

        // Warmup
        for (int r=0;r<3;r++)
            launch_crowding_distance(
                d_F,d_rank,d_cd,N,M,0);
        cudaDeviceSynchronize();

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        int RUNS = 20;
        cudaEventRecord(t0);
        for (int r=0;r<RUNS;r++)
            launch_crowding_distance(
                d_F,d_rank,d_cd,N,M,0);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms=0;
        cudaEventElapsedTime(&ms,t0,t1);
        printf("%-8d %-4d %-12.3f\n",N,M,ms/RUNS);

        cudaFree(d_F);cudaFree(d_rank);cudaFree(d_cd);
        free(h_F);free(h_rank);free(h_cd);
    }
    return 0;
}
