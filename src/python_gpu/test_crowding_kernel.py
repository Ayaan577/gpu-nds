import cupy as cp
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.python_gpu.gpu_kernels.crowding_distance import gpu_crowding_distance

def reference_crowding(F_np, ranks_np):
    """Pure numpy reference implementation."""
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
            f_range = (F_np[sorted_idx[-1], m] - 
                       F_np[sorted_idx[0],  m])
            if f_range < 1e-10:
                continue
            for j in range(1, len(sorted_idx) - 1):
                cd[sorted_idx[j]] += (
                    (F_np[sorted_idx[j+1], m] - 
                     F_np[sorted_idx[j-1], m]) / f_range
                )
    return cd

def test_crowding(N, M, seed=42):
    rng = np.random.RandomState(seed)
    F_np = rng.rand(N, M).astype(np.float32)
    
    # Get ranks via your existing GPU NDS
    from src.python_gpu.gpu_nds_cupy import GPU_NDS
    nds_sorter = GPU_NDS()
    
    F_gpu    = cp.array(F_np)
    rank_np, _ = nds_sorter.sort(F_np)
    rank_gpu = cp.array(rank_np, dtype=cp.int32)
    
    # GPU crowding
    cd_gpu = gpu_crowding_distance(F_gpu, rank_gpu, N, M)
    cd_gpu_np = cp.asnumpy(cd_gpu)
    
    # Reference crowding
    cd_ref = reference_crowding(F_np, rank_np)
    
    # Compare (INF should match, finite values within 1%)
    inf_mask_ref = np.isinf(cd_ref)
    inf_mask_gpu = cd_gpu_np > 1e29
    assert np.all(inf_mask_ref == inf_mask_gpu), "INF boundary mismatch"
    inf_mask = inf_mask_ref
    
    finite_mask = ~inf_mask
    if finite_mask.sum() > 0:
        rel_err = np.abs(
            cd_gpu_np[finite_mask] - cd_ref[finite_mask]
        ) / (np.abs(cd_ref[finite_mask]) + 1e-10)
        max_err = rel_err.max()
        assert max_err < 0.01, \
            f"Max relative error {max_err:.4f} exceeds 1%"
        print(f"  N={N:5d}, M={M}: PASS "
              f"(max_rel_err={max_err:.5f})")
    else:
        print(f"  N={N:5d}, M={M}: PASS (all boundary)")

if __name__ == '__main__':
    print("Crowding distance kernel correctness tests:")
    for N in [50, 200, 500, 1000, 2000]:
        for M in [2, 3, 5, 8, 10]:
            test_crowding(N, M)
    print("ALL TESTS PASSED")
