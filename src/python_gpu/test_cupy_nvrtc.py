import ctypes
import os
import cupy as cp
import numpy as np

try:
    from src.python_gpu.gpu_kernels.crowding_distance import gpu_crowding_distance
    print("Module loaded.")
except Exception as e:
    print(f"Error loading module: {e}")
    exit(1)

# Test arrays on GPU
N, M = 10, 2
# Random data
h_F = np.random.rand(N, M).astype(np.float32)
h_rank = np.zeros(N, dtype=np.int32)

print("Arrays created.")

# Copy to device using CuPy
d_F = cp.asarray(h_F)
d_rank = cp.asarray(h_rank)

print("Calling gpu_crowding_distance...")
d_cd = gpu_crowding_distance(d_F, d_rank, N, M)

print("CD Array:")
print(d_cd)
