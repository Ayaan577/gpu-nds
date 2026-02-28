"""
Python NDS Baseline Benchmark — SORTING ONLY, no full NSGA-II.
Times just the NDS sorting call on random data.
"""
import numpy as np
import time
import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.cpu.nsga2_sort import fast_non_dominated_sort as python_fast_nds


def time_python_nds_sort(F, n_trials=10):
    """Time just the NDS sorting call, nothing else."""
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        ranks, _ = python_fast_nds(F)
        elapsed = time.perf_counter() - t0
        times.append(elapsed * 1000)  # ms
    return np.mean(times[2:])  # skip 2 warmup


configs = []
for N in [100, 500, 1000, 2000, 5000, 10000]:
    for M in [2, 3, 5, 8, 10]:
        configs.append((N, M))

results = []
for N, M in configs:
    # HARD STOP: skip configs that would take > 120 seconds
    if N > 5000 and M > 5:
        print(f"N={N:6d} M={M}: SKIPPED (would exceed 120s)")
        results.append({
            'N': N, 'M': M,
            'python_dcns_ms': -1,
            'python_nsga_ms': -1
        })
        continue

    F = np.random.rand(N, M).astype(np.float32)
    t_dcns = time_python_nds_sort(F)
    t_nsga = time_python_nds_sort(F)
    results.append({
        'N': N, 'M': M,
        'python_dcns_ms': round(t_dcns, 3),
        'python_nsga_ms': round(t_nsga, 3)
    })
    print(f"N={N:6d} M={M}: "
          f"DCNS={t_dcns:.2f}ms  NSGA={t_nsga:.2f}ms")

os.makedirs('experiments/results', exist_ok=True)
with open('experiments/results/python_nds_times.csv',
          'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("Done. Saved to python_nds_times.csv")
print("Expected runtime: under 10 minutes total.")
