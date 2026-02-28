import cupy as cp
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.python_gpu.gpu_kernels.crowding_distance import gpu_crowding_distance

def reference_crowding(F_np, ranks_np):
    N, M = F_np.shape
    cd = np.zeros(N)
    n_fronts = int(ranks_np.max()) + 1
    for k in range(n_fronts):
        idx = np.where(ranks_np == k)[0]
        if len(idx) <= 2:
            cd[idx] = np.inf
            continue
        for m in range(M):
            sorted_order = np.argsort(F_np[idx, m])
            sorted_idx   = idx[sorted_order]
            cd[sorted_idx[0]]  = np.inf
            cd[sorted_idx[-1]] = np.inf
            f_range = (F_np[sorted_idx[-1], m] - F_np[sorted_idx[0],  m])
            if f_range < 1e-10:
                continue
            for j in range(1, len(sorted_idx) - 1):
                cd[sorted_idx[j]] += ((F_np[sorted_idx[j+1], m] - F_np[sorted_idx[j-1], m]) / f_range)
    return cd

N, M = 10, 2
rng = np.random.RandomState(42)
F_np = rng.rand(N, M).astype(np.float32)
ranks_np = np.zeros(N, dtype=np.int32) # All in front 0

F_gpu = cp.array(F_np)
rank_gpu = cp.array(ranks_np)

cd_gpu = cp.asnumpy(gpu_crowding_distance(F_gpu, rank_gpu, N, M))
cd_ref = reference_crowding(F_np, ranks_np)

print("OBJ (first objective F[:,0]):", F_np[:,0])
print("OBJ (second objective F[:,1]):", F_np[:,1])
print("REF:", cd_ref)
print("GPU:", cd_gpu)
