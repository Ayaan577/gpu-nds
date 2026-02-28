import ctypes
import os
import cupy as cp
import numpy as np

# Load the built DLL
# Try loading from the current directory, or absolute path
dll_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gpu_kernels', 'libcrowding.dll'))
print(f"Loading {dll_path}...")
lib = ctypes.CDLL(dll_path)

# Set the function signature
# EXPORT_API void compute_crowding_distance(const float* h_F, const int* h_rank, float* h_cd, int N, int M)
lib.compute_crowding_distance.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int
]
lib.compute_crowding_distance.restype = None

# Test arrays
N, M = 10, 2
h_F = np.random.rand(N, M).astype(np.float32)
h_rank = np.zeros(N, dtype=np.int32)
h_cd = np.zeros(N, dtype=np.float32)

F_ptr = h_F.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
rank_ptr = h_rank.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
cd_ptr = h_cd.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

print("Invoking DLL from python...")
lib.compute_crowding_distance(F_ptr, rank_ptr, cd_ptr, N, M)

print("CD Array:")
print(h_cd)
