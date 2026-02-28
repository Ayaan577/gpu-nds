import cupy as cp
import os

# Get path to the kernel folder
KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(KERNEL_DIR, 'crowding_distance.cu'), 'r') as f:
    cuda_source = f.read()
with open(os.path.join(KERNEL_DIR, 'crowding_distance_launcher.cu'), 'r') as f:
    cuda_source_launcher = f.read()

# We need to combine the source since NVRTC compiles a single module
# Extract just the kernel definitions
combined_source = cuda_source + "\n\n" + cuda_source_launcher

try:
    _module = cp.RawModule(code=combined_source, options=('-std=c++17', '--use_fast_math'))
    _crowding_kernel = _module.get_function('crowding_distance_kernel')
    _max_rank_kernel = _module.get_function('max_rank_kernel')
except Exception as e:
    print(f"Failed to compile raw kernel: {e}")

def gpu_crowding_distance_standalone(F_gpu, rank_gpu, N, M):
    """
    Python launcher corresponding to Step 3.
    """
    assert F_gpu.dtype == cp.float32, "F must be float32"
    assert rank_gpu.dtype == cp.int32, "rank must be int32"
    
    # 1. Get max rank using our kernel just to use everything we built
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
