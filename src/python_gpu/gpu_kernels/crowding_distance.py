import cupy as cp
import os

KERNEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'gpu_kernels'))

# Read the standalone C++ kernel we created
with open(os.path.join(KERNEL_DIR, 'crowding_distance.cu'), 'r') as f:
    cuda_source = f.read()
with open(os.path.join(KERNEL_DIR, 'crowding_distance_launcher.cu'), 'r') as f:
    cuda_source_launcher = f.read()

# NVRTC does not support <cuda_runtime.h>, <device_launch_parameters.h>, etc.
# We strip includes to let NVRTC compile the raw C++ code.
def strip_includes(source):
    lines = source.split('\n')
    return '\n'.join([l for l in lines if not l.startswith('#include') and not l.startswith('#pragma')])

# Combine source and add required definitions
# Replace FLT_MAX with its literal value to avoid conflict with CuPy's limits header.
combined_source = (
    "#define min(a,b) ((a)<(b)?(a):(b))\n"
    + strip_includes(cuda_source).replace("FLT_MAX", "3.402823466e+38F")
)

# Compile via CuPy NVRTC (uses driver's built-in compiler, always compatible)
_module = cp.RawModule(
    code=combined_source, 
    options=('-std=c++14', '--use_fast_math')
)
_crowding_kernel = _module.get_function('crowding_distance_kernel')
_max_rank_kernel = _module.get_function('max_rank_kernel')

def gpu_crowding_distance(F_gpu, rank_gpu, N, M):
    """
    Python wrapper that launches the native C++ standalone kernels compiled dynamically.
    This replaces the pure-cupy Python implementation.
    """
    assert F_gpu.dtype == cp.float32, "F must be float32"
    assert rank_gpu.dtype == cp.int32, "rank must be int32"
    
    # 1. Get max rank using our kernel
    d_max_rank = cp.zeros(1, dtype=cp.int32)
    _max_rank_kernel((1,), (128,), (rank_gpu, d_max_rank, N))
    
    n_fronts = int(d_max_rank.get()[0]) + 1
    
    # 2. Zero-init output
    cd_gpu = cp.zeros(N, dtype=cp.float32)
    
    # 3. Launch
    _crowding_kernel(
        (n_fronts,), (128,),
        (F_gpu, rank_gpu, cd_gpu, N, M, n_fronts)
    )
    
    return cd_gpu
