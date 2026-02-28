/**
 * CUDA Utility Functions
 */
#ifndef UTILS_CUH
#define UTILS_CUH

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

/**
 * Print GPU device information.
 */
inline void print_device_info() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s\n", prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Global memory: %.1f GB\n", prop.totalGlobalMem / 1e9);
    printf("  Shared memory per block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("  Memory bandwidth: %.1f GB/s\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);
    printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
}

/**
 * Timer using CUDA events for accurate GPU timing.
 */
struct CudaTimer {
    cudaEvent_t start, stop;

    CudaTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void tic() { cudaEventRecord(start); }
    float toc_ms() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

#endif // UTILS_CUH
