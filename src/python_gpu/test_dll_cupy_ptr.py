import ctypes
import os
import cupy as cp
import numpy as np

# Load the built DLL
dll_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gpu_kernels', 'libcrowding.dll'))
print(f"Loading {dll_path}...")
lib = ctypes.CDLL(dll_path)

# EXPORT_API void launch_crowding_distance(const float* d_F, const int* d_rank, float* d_cd, int N, int M, void* stream)
lib.launch_crowding_distance.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p
]
lib.launch_crowding_distance.restype = None

# Test arrays on GPU
N, M = 10, 2
h_F = np.random.rand(N, M).astype(np.float32)
h_rank = np.zeros(N, dtype=np.int32)
h_cd = np.zeros(N, dtype=np.float32)

# Copy to device using CuPy
d_F = cp.asarray(h_F)
d_rank = cp.asarray(h_rank)
d_cd = cp.asarray(h_cd)

stream_ptr = cp.cuda.get_current_stream().ptr

print("Invoking DLL from python using CuPy memory pointers...")
lib.launch_crowding_distance(d_F.data.ptr, d_rank.data.ptr, d_cd.data.ptr, N, M, stream_ptr)

print("CD Array:")
print(d_cd)
