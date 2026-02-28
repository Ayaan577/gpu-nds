"""
Master Benchmark Runner — runs all 6 experiments and saves CSV results.

Usage:
    python -m src.benchmarks.run_benchmarks [--all | --smoke | --exp N]

Experiments:
    1. Scalability with N (population size)
    2. Scalability with M (number of objectives)
    3. GPU Speedup Curves (derived from 1 & 2)
    4. Dominance Comparisons Count
    5. Ablation Study (TILE size, presort)
    6. Real-World Task (sklearn hyperparameter optimisation)
"""

import sys
import os
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.cpu.nsga2_sort import fast_non_dominated_sort
from src.cpu.dcns import dcns_sort, DCNSSorter
from src.cpu.bos import bos_sort
from src.python_gpu.gpu_nds_cupy import GPU_NDS, NaiveGPU_NDS, get_backend
from src.benchmarks.generate_problems import (
    generate_dtlz, generate_wfg, generate_random_uniform,
)

# C++ baselines (via ctypes)
try:
    from src.cpu_cpp.cpu_nds_wrapper import (
        nsga2_sort as cpp_nsga2_sort,
        dcns_sort as cpp_dcns_sort,
        bos_sort as cpp_bos_sort,
    )
    _CPP_AVAILABLE = True
except (ImportError, FileNotFoundError):
    _CPP_AVAILABLE = False
    print("WARNING: C++ baselines not available, skipping CPP-* algorithms")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'experiments', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# =====================================================================
# Helper: run an algorithm and return timing
# =====================================================================

def _run_algo(name, objectives, repeats=10, **kwargs):
    """Run an algorithm *repeats* times and return (mean_ms, std_ms, ranks).

    Parameters
    ----------
    name : str
        One of 'CPU-NSGA2', 'CPU-DCNS', 'CPU-BOS', 'GPU-NDS', 'Naive-GPU'.
    objectives : np.ndarray (N, M)
    repeats : int
    **kwargs : forwarded to sorter constructors

    Returns
    -------
    dict with keys: mean_ms, std_ms, ranks (from last run)
    """
    times = []
    ranks = None

    for _ in range(repeats):
        if name == 'CPU-NSGA2':
            r, t = fast_non_dominated_sort(objectives)
        elif name == 'CPU-DCNS':
            use_presort = kwargs.get('use_sum_presort', True)
            r, t = dcns_sort(objectives, use_sum_presort=use_presort)
        elif name == 'CPU-BOS':
            r, t = bos_sort(objectives)
        elif name == 'CPP-NSGA2':
            r, t = cpp_nsga2_sort(objectives)
        elif name == 'CPP-DCNS':
            use_presort = kwargs.get('use_sum_presort', True)
            r, t = cpp_dcns_sort(objectives, use_presort=use_presort)
        elif name == 'CPP-BOS':
            r, t = cpp_bos_sort(objectives)
        elif name == 'GPU-NDS':
            tile = kwargs.get('tile_size', 32)
            presort = kwargs.get('use_sum_presort', True)
            sorter = GPU_NDS(tile_size=tile, use_sum_presort=presort)
            r, t = sorter.sort(objectives)
        elif name == 'Naive-GPU':
            sorter = NaiveGPU_NDS()
            r, t = sorter.sort(objectives)
        else:
            raise ValueError(f"Unknown algorithm: {name}")
        times.append(t)
        ranks = r

    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'ranks': ranks,
    }


# =====================================================================
# Experiment 1: Scalability with N
# =====================================================================

def exp1_scalability_N(smoke=False):
    """Runtime vs population size N."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Scalability with N")
    print("=" * 60)

    N_vals = [100, 500, 1000, 2000, 5000] if not smoke else [100, 500]
    if not smoke:
        N_vals += [10000]
    M = 5
    problems = {
        'DTLZ2': lambda N: generate_dtlz(2, N, M, seed=42),
        'DTLZ7': lambda N: generate_dtlz(7, N, M, seed=42),
        'WFG4': lambda N: generate_wfg(4, N, M, seed=42),
    }
    algos = ['CPU-NSGA2', 'CPU-DCNS', 'CPU-BOS', 'GPU-NDS', 'Naive-GPU']
    repeats = 3 if smoke else 10

    rows = []
    for prob_name, gen in problems.items():
        for N in tqdm(N_vals, desc=prob_name):
            obj = gen(N)
            for algo in algos:
                # Skip slow algos for very large N
                if N > 5000 and algo in ('CPU-NSGA2', 'CPU-BOS', 'Naive-GPU'):
                    rows.append({
                        'problem': prob_name, 'N': N, 'M': M,
                        'algorithm': algo, 'mean_ms': np.nan, 'std_ms': np.nan,
                    })
                    continue
                res = _run_algo(algo, obj, repeats=repeats)
                rows.append({
                    'problem': prob_name, 'N': N, 'M': M,
                    'algorithm': algo,
                    'mean_ms': res['mean_ms'], 'std_ms': res['std_ms'],
                })

    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, 'exp1_scalability_N.csv')
    df.to_csv(path, index=False)
    print(f"  Saved to {path}")
    return df


# =====================================================================
# Experiment 2: Scalability with M
# =====================================================================

def exp2_scalability_M(smoke=False):
    """Runtime vs number of objectives M."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Scalability with M")
    print("=" * 60)

    N = 2000 if not smoke else 500
    M_vals = [2, 3, 5, 8, 10, 15] if not smoke else [2, 3, 5]
    problems = {
        'DTLZ2': lambda M: generate_dtlz(2, N, M, seed=42),
        'DTLZ6': lambda M: generate_dtlz(6, N, M, seed=42),
    }
    algos = ['CPU-NSGA2', 'CPU-DCNS', 'CPU-BOS', 'GPU-NDS', 'Naive-GPU']
    repeats = 3 if smoke else 10

    rows = []
    for prob_name, gen in problems.items():
        for M in tqdm(M_vals, desc=prob_name):
            obj = gen(M)
            for algo in algos:
                res = _run_algo(algo, obj, repeats=repeats)
                rows.append({
                    'problem': prob_name, 'N': N, 'M': M,
                    'algorithm': algo,
                    'mean_ms': res['mean_ms'], 'std_ms': res['std_ms'],
                })

    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, 'exp2_scalability_M.csv')
    df.to_csv(path, index=False)
    print(f"  Saved to {path}")
    return df


# =====================================================================
# Experiment 3: GPU Speedup Curves
# =====================================================================

def exp3_speedup(smoke=False):
    """Compute speedup = CPU-DCNS time / GPU-NDS time."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: GPU Speedup")
    print("=" * 60)

    N_vals = [100, 500, 1000, 2000, 5000] if not smoke else [100, 500]
    M_vals = [2, 3, 5, 8, 10] if not smoke else [2, 3, 5]
    repeats = 3 if smoke else 10

    rows = []
    for N in tqdm(N_vals, desc='Speedup'):
        for M in M_vals:
            obj = generate_dtlz(2, N, M, seed=42)
            res_cpu = _run_algo('CPU-DCNS', obj, repeats=repeats)
            res_gpu = _run_algo('GPU-NDS', obj, repeats=repeats)
            speedup = res_cpu['mean_ms'] / res_gpu['mean_ms'] if res_gpu['mean_ms'] > 0 else 0
            rows.append({
                'N': N, 'M': M,
                'cpu_dcns_ms': res_cpu['mean_ms'],
                'gpu_nds_ms': res_gpu['mean_ms'],
                'speedup': speedup,
            })

    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, 'exp3_speedup.csv')
    df.to_csv(path, index=False)
    print(f"  Saved to {path}")
    return df


# =====================================================================
# Experiment 4: Dominance Comparison Count
# =====================================================================

def exp4_comparisons(smoke=False):
    """Count actual dominance comparisons."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Dominance Comparisons")
    print("=" * 60)

    N_vals = [100, 500, 1000, 5000] if not smoke else [100, 500]
    M_vals = [3, 5, 10] if not smoke else [3, 5]
    problems = {
        'DTLZ1': 1, 'DTLZ2': 2, 'DTLZ5': 5,
    }

    rows = []
    for prob_name, prob_id in problems.items():
        for N in tqdm(N_vals, desc=prob_name):
            for M in M_vals:
                obj = generate_dtlz(prob_id, N, M, seed=42)

                # CPU-DCNS no presort
                _, _, comp_no = dcns_sort(obj, use_sum_presort=False, count_comparisons=True)
                # CPU-DCNS with presort
                _, _, comp_yes = dcns_sort(obj, use_sum_presort=True, count_comparisons=True)
                # GPU-NDS (approximate)
                sorter = GPU_NDS(use_sum_presort=True)
                sorter.sort(obj, count_comparisons=True)
                comp_gpu = sorter.comparisons

                rows.append({
                    'problem': prob_name, 'N': N, 'M': M,
                    'DCNS_no_presort': comp_no,
                    'DCNS_presort': comp_yes,
                    'GPU_NDS': comp_gpu,
                })

    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, 'exp4_comparisons.csv')
    df.to_csv(path, index=False)
    print(f"  Saved to {path}")
    return df


# =====================================================================
# Experiment 5: Ablation Study
# =====================================================================

def exp5_ablation(smoke=False):
    """TILE size and presort ablation."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Ablation Study")
    print("=" * 60)

    N = 2000 if not smoke else 500
    M = 10 if not smoke else 5
    obj = generate_dtlz(2, N, M, seed=42)
    repeats = 3 if smoke else 10

    rows = []
    # TILE size ablation
    for tile in [16, 32, 64]:
        res = _run_algo('GPU-NDS', obj, repeats=repeats, tile_size=tile)
        rows.append({
            'variant': f'TILE={tile}', 'N': N, 'M': M,
            'mean_ms': res['mean_ms'], 'std_ms': res['std_ms'],
        })

    # Presort ablation
    for presort in [True, False]:
        res = _run_algo('GPU-NDS', obj, repeats=repeats, use_sum_presort=presort)
        rows.append({
            'variant': f'presort={"ON" if presort else "OFF"}',
            'N': N, 'M': M,
            'mean_ms': res['mean_ms'], 'std_ms': res['std_ms'],
        })

    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, 'exp5_ablation.csv')
    df.to_csv(path, index=False)
    print(f"  Saved to {path}")
    return df


# =====================================================================
# Experiment 6: Real-World Task
# =====================================================================

def exp6_real_world(smoke=False):
    """Multi-objective hyperparameter optimisation with sklearn."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 6: Real-World Task")
    print("=" * 60)

    from sklearn.datasets import load_digits
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import cross_val_score

    digits = load_digits()
    X, y = digits.data, digits.target

    rng = np.random.default_rng(42)
    n_configs = 50 if not smoke else 15
    configs = []
    for _ in range(n_configs):
        hidden = tuple(rng.integers(10, 100, size=rng.integers(1, 4)))
        lr = 10 ** rng.uniform(-4, -1)
        alpha = 10 ** rng.uniform(-5, -2)
        configs.append({'hidden_layer_sizes': hidden,
                        'learning_rate_init': lr, 'alpha': alpha})

    objectives = []
    print("  Training MLPs...")
    for cfg in tqdm(configs, desc='MLP'):
        t0 = time.perf_counter()
        clf = MLPClassifier(max_iter=100, random_state=42, **cfg)
        try:
            scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
            acc = scores.mean()
        except Exception:
            acc = 0.0
        t1 = time.perf_counter()
        n_params = sum(np.prod(c.shape) for c in clf.coefs_) if hasattr(clf, 'coefs_') else 0
        objectives.append([1.0 - acc, (t1 - t0) * 1000, n_params])

    obj = np.array(objectives, dtype=np.float64)

    algos = ['CPU-NSGA2', 'CPU-DCNS', 'CPU-BOS', 'GPU-NDS']
    repeats = 5

    rows = []
    for algo in algos:
        res = _run_algo(algo, obj, repeats=repeats)
        rows.append({
            'algorithm': algo, 'N': len(obj), 'M': 3,
            'mean_ms': res['mean_ms'], 'std_ms': res['std_ms'],
        })
        print(f"  {algo}: {res['mean_ms']:.2f} ± {res['std_ms']:.2f} ms")

    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, 'exp6_real_world.csv')
    df.to_csv(path, index=False)
    print(f"  Saved to {path}")
    return df


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description='GPU-NDS Benchmark Runner')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--smoke', action='store_true', help='Quick smoke test (small sizes)')
    parser.add_argument('--exp', type=int, nargs='+', help='Run specific experiment(s)')
    args = parser.parse_args()

    print(f"GPU Backend: {get_backend()}")
    print(f"Results dir: {os.path.abspath(RESULTS_DIR)}")

    smoke = args.smoke

    experiments = {
        1: exp1_scalability_N,
        2: exp2_scalability_M,
        3: exp3_speedup,
        4: exp4_comparisons,
        5: exp5_ablation,
        6: exp6_real_world,
    }

    to_run = list(experiments.keys()) if (args.all or args.smoke) else (args.exp or [])

    if not to_run:
        print("No experiments specified. Use --all, --smoke, or --exp N")
        return

    for exp_id in sorted(to_run):
        try:
            experiments[exp_id](smoke=smoke)
        except Exception as e:
            print(f"\n[ERROR] Experiment {exp_id} FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("ALL DONE")
    print("=" * 60)


if __name__ == '__main__':
    main()
