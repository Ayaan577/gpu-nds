"""
Quick C++ vs GPU benchmark across key N and M values.
Saves results to experiments/results/cpp_vs_gpu.csv
"""
import sys, os, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.cpu_cpp.cpu_nds_wrapper import nsga2_sort as cpp_nsga2, dcns_sort as cpp_dcns, bos_sort as cpp_bos
from src.python_gpu.gpu_nds_cupy import GPU_NDS, get_backend
from src.benchmarks.generate_problems import generate_dtlz

print(f"GPU Backend: {get_backend()}")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'experiments', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def bench(func, obj, repeats=10):
    """Run func(obj) repeats times and return (mean_ms, std_ms)."""
    times = []
    for _ in range(repeats):
        _, t = func(obj)
        times.append(t)
    return np.mean(times), np.std(times)

def bench_gpu(obj, repeats=10):
    """Run GPU-NDS repeats times and return (mean_ms, std_ms)."""
    sorter = GPU_NDS(tile_size=32, use_sum_presort=True)
    times = []
    for _ in range(repeats):
        _, t = sorter.sort(obj)
        times.append(t)
    return np.mean(times), np.std(times)

# ===== Experiment: C++ vs GPU across N and M =====
rows = []
N_vals = [100, 500, 1000, 2000, 5000, 10000]
M_vals = [2, 3, 5, 8, 10]
repeats = 10

print("\n" + "=" * 70)
print("C++ CPU vs GPU-NDS Benchmark")
print("=" * 70)

for M in M_vals:
    print(f"\n--- M={M} ---")
    for N in N_vals:
        obj = generate_dtlz(2, N, M, seed=42)  # DTLZ2

        # Skip very slow configs
        if N > 5000 and M > 5:
            # Skip CPP-NSGA2 for very large N (still O(MN^2))
            pass

        # C++ baselines
        algos = {}
        if N <= 5000:
            m, s = bench(cpp_nsga2, obj, repeats)
            algos['CPP-NSGA2'] = (m, s)
        
        m, s = bench(cpp_dcns, obj, repeats)
        algos['CPP-DCNS'] = (m, s)
        
        m, s = bench(cpp_bos, obj, repeats)
        algos['CPP-BOS'] = (m, s)
        
        # GPU
        m_gpu, s_gpu = bench_gpu(obj, repeats)
        algos['GPU-NDS'] = (m_gpu, s_gpu)
        
        # Print summary
        parts = []
        for name, (m, s) in algos.items():
            parts.append(f"{name}={m:.1f}±{s:.1f}ms")
        print(f"  N={N:5d}: {' | '.join(parts)}")
        
        # Compute speedups
        for name, (m, s) in algos.items():
            if name == 'GPU-NDS':
                continue
            rows.append({
                'N': N, 'M': M,
                'algorithm': name,
                'mean_ms': m, 'std_ms': s,
                'gpu_ms': m_gpu, 'gpu_std': s_gpu,
                'speedup': m / m_gpu if m_gpu > 0 else np.nan,
            })
        # Also store GPU row
        rows.append({
            'N': N, 'M': M,
            'algorithm': 'GPU-NDS',
            'mean_ms': m_gpu, 'std_ms': s_gpu,
            'gpu_ms': m_gpu, 'gpu_std': s_gpu,
            'speedup': 1.0,
        })

df = pd.DataFrame(rows)
path = os.path.join(RESULTS_DIR, 'cpp_vs_gpu.csv')
df.to_csv(path, index=False)
print(f"\nResults saved to {path}")

# Print key speedup summary
print("\n" + "=" * 70)
print("KEY SPEEDUPS (C++ vs GPU-NDS)")
print("=" * 70)
for M in M_vals:
    sub = df[(df['M'] == M) & (df['algorithm'] == 'CPP-DCNS')]
    if len(sub) > 0:
        for _, row in sub.iterrows():
            print(f"  N={int(row['N']):5d}, M={M:2d}: CPP-DCNS={row['mean_ms']:10.1f}ms → GPU={row['gpu_ms']:6.1f}ms  speedup={row['speedup']:8.1f}x")

print("\nDONE")
