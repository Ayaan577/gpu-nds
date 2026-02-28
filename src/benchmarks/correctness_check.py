"""
Correctness Verification for GPU-NDS

Compares front assignments from:
  - CPU-NSGA2 (Deb 2002)
  - CPU-DCNS  (Mishra 2019)
  - GPU-NDS   (our implementation)

For every (problem, N, M, seed) configuration, asserts that all three
algorithms produce identical front assignments.

Usage:
    python -m src.benchmarks.correctness_check
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.cpu.nsga2_sort import fast_non_dominated_sort
from src.cpu.dcns import dcns_sort
from src.python_gpu.gpu_nds_cupy import gpu_nds


def check_ranks_equivalent(ranks_a, ranks_b, name_a, name_b):
    """Check that two rank assignments define the same front partition.

    Two rank arrays are equivalent if for every pair (i, j),
    ranks_a[i] == ranks_a[j]  <=>  ranks_b[i] == ranks_b[j]
    AND the relative ordering of fronts is preserved.

    In practice we normalise both to 0-indexed consecutive integers
    and compare directly.

    Parameters
    ----------
    ranks_a, ranks_b : np.ndarray of int
    name_a, name_b : str

    Returns
    -------
    bool
    """
    # Normalise: map to 0-indexed consecutive ranks
    def normalise(r):
        unique = np.unique(r)
        mapping = {v: i for i, v in enumerate(sorted(unique))}
        return np.array([mapping[v] for v in r], dtype=np.int32)

    ra = normalise(ranks_a)
    rb = normalise(ranks_b)

    if np.array_equal(ra, rb):
        return True
    else:
        diff_idx = np.where(ra != rb)[0]
        print(f"  MISMATCH {name_a} vs {name_b}: "
              f"{len(diff_idx)} solutions differ (first 5: {diff_idx[:5]})")
        for idx in diff_idx[:5]:
            print(f"    sol {idx}: {name_a}={ra[idx]}, {name_b}={rb[idx]}")
        return False


def run_correctness_check(verbose=True):
    """Run correctness checks across multiple configurations.

    Returns
    -------
    all_pass : bool
    results : list of dict
    """
    configs = []
    # N × M × problem × seed
    for N in [100, 500, 1000]:
        for M in [2, 3, 5, 10]:
            for prob in ['dtlz1', 'dtlz2', 'dtlz3']:
                for seed in [0, 1, 2]:
                    configs.append({'N': N, 'M': M, 'problem': prob, 'seed': seed})

    results = []
    pass_count = 0
    fail_count = 0

    from src.benchmarks.generate_problems import generate_dtlz

    if verbose:
        print(f"{'Problem':<10} {'N':>6} {'M':>3} {'Seed':>4}  "
              f"{'NSGA2-DCNS':>11} {'NSGA2-GPU':>10} {'DCNS-GPU':>10}  {'Status'}")
        print("-" * 75)

    for cfg in configs:
        prob_id = int(cfg['problem'].replace('dtlz', ''))
        obj = generate_dtlz(prob_id, cfg['N'], cfg['M'], seed=cfg['seed'])

        ranks_nsga2, _ = fast_non_dominated_sort(obj)
        ranks_dcns, _ = dcns_sort(obj, use_sum_presort=True)
        ranks_gpu, _ = gpu_nds(obj)

        ok_nd = check_ranks_equivalent(ranks_nsga2, ranks_dcns, 'NSGA2', 'DCNS')
        ok_ng = check_ranks_equivalent(ranks_nsga2, ranks_gpu, 'NSGA2', 'GPU')
        ok_dg = check_ranks_equivalent(ranks_dcns, ranks_gpu, 'DCNS', 'GPU')

        all_ok = ok_nd and ok_ng and ok_dg
        status = 'PASS' if all_ok else 'FAIL'

        if all_ok:
            pass_count += 1
        else:
            fail_count += 1

        results.append({**cfg, 'pass': all_ok})

        if verbose:
            sym = lambda ok: 'Y' if ok else 'N'
            print(f"{cfg['problem']:<10} {cfg['N']:>6} {cfg['M']:>3} {cfg['seed']:>4}  "
                  f"{'':>4}{sym(ok_nd):<7} {'':>3}{sym(ok_ng):<7} {'':>3}{sym(ok_dg):<7}  {status}")

    if verbose:
        print("-" * 75)
        print(f"Total: {pass_count} PASS, {fail_count} FAIL out of {len(configs)}")

    if fail_count > 0:
        print("\n[FAIL] CORRECTNESS CHECK FAILED -- do NOT proceed to paper submission!")
        return False, results

    print("\n[OK] All correctness checks PASSED.")
    return True, results


if __name__ == '__main__':
    all_pass, _ = run_correctness_check(verbose=True)
    sys.exit(0 if all_pass else 1)
