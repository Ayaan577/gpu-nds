"""
Analytical Performance Model for GPU-NDS

1. Theoretical comparison count:
   C(N, M, F) ≈ N * (N/F) * p_dom(M)
   where F = expected fronts, p_dom = probability of comparability.

2. GPU memory transaction model:
   transactions(N, M, TILE) bytes read from global memory.

3. Validation: plot theoretical vs empirical comparison counts.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def expected_fronts(N, M):
    """Estimate expected number of Pareto fronts for random uniform data.

    For random points in [0,1]^M, the expected number of fronts is
    approximately O(N^(1 - 1/(M-1))) for M >= 2.

    Parameters
    ----------
    N : int
    M : int

    Returns
    -------
    float
    """
    if M <= 1:
        return 1.0
    exponent = 1.0 - 1.0 / max(M - 1, 1)
    return max(1.0, N ** exponent)


def p_dominance(M):
    """Probability that one random solution dominates another in M objectives.

    For uniformly random solutions, P(a dominates b) ≈ 1/M! * 2^M ... 
    A practical approximation: p_dom ≈ 2 / (M + 1)! which decreases rapidly.
    Simpler bound used in paper: M / (M+1) for the probability of comparability.

    Parameters
    ----------
    M : int

    Returns
    -------
    float
    """
    # Exact: P(dom) = 1/M! but approximate comparability = 2 * 1/M!
    from math import factorial
    return 2.0 / factorial(M)


def theoretical_comparisons(N, M, F_estimate=None):
    """Theoretical dominance comparison count for DCNS with presort.

    C ≈ N * (N / F) * p_dom(M)

    Parameters
    ----------
    N : int
    M : int
    F_estimate : float or None
        If None, use expected_fronts(N, M).

    Returns
    -------
    float
    """
    if F_estimate is None:
        F_estimate = expected_fronts(N, M)
    return N * (N / F_estimate) * p_dominance(M)


def empirical_comparisons(N, M, problem_id=2, seed=42):
    """Run CPU-DCNS with comparison counter and return actual count.

    Parameters
    ----------
    N, M : int
    problem_id : int
    seed : int

    Returns
    -------
    int
    """
    from src.benchmarks.generate_problems import generate_dtlz
    from src.cpu.dcns import dcns_sort
    obj = generate_dtlz(problem_id, N, M, seed=seed)
    _, _, comps = dcns_sort(obj, use_sum_presort=True, count_comparisons=True)
    return comps


def memory_transactions(N, M, TILE):
    """Estimate global memory bytes read in tiled dominance check.

    Each tile pair loads 2 * TILE * M floats from global memory.
    Total tile pairs = (N/TILE)^2.

    Parameters
    ----------
    N, M, TILE : int

    Returns
    -------
    int : bytes
    """
    n_tiles = (N + TILE - 1) // TILE
    return n_tiles * n_tiles * 2 * TILE * M * 4  # float32 = 4 bytes


def bandwidth_utilisation(N, M, TILE, bandwidth_GBps, time_ms):
    """Fraction of peak memory bandwidth utilised.

    Parameters
    ----------
    N, M, TILE : int
    bandwidth_GBps : float
        Peak bandwidth in GB/s.
    time_ms : float
        Measured kernel time.

    Returns
    -------
    float : fraction (0-1)
    """
    bytes_read = memory_transactions(N, M, TILE)
    achieved = bytes_read / (time_ms / 1000.0) / 1e9  # GB/s
    return achieved / bandwidth_GBps


def validate_model(save_dir=None, smoke=False):
    """Plot theoretical vs empirical comparisons.

    Saves plot to experiments/plots/fig7_model_validation.pdf

    Parameters
    ----------
    save_dir : str or None
    smoke : bool
    """
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(__file__), '..', '..',
                                'experiments', 'plots')
    os.makedirs(save_dir, exist_ok=True)

    N_vals = [100, 500, 1000, 2000] if not smoke else [100, 500]
    M_vals = [3, 5, 10] if not smoke else [3, 5]

    fig, axes = plt.subplots(1, len(M_vals), figsize=(5 * len(M_vals), 4),
                             sharey=False)
    if len(M_vals) == 1:
        axes = [axes]

    colors = plt.cm.Set2.colors

    for ax_idx, M in enumerate(M_vals):
        ax = axes[ax_idx]
        theo = []
        emp = []
        for N in N_vals:
            theo.append(theoretical_comparisons(N, M))
            emp.append(empirical_comparisons(N, M, problem_id=2, seed=42))

        ax.plot(N_vals, theo, 'o--', color=colors[0], label='Theoretical', linewidth=2)
        ax.plot(N_vals, emp, 's-', color=colors[1], label='Empirical (DCNS)', linewidth=2)
        ax.set_xlabel('N (population size)', fontsize=12)
        ax.set_ylabel('Dominance comparisons', fontsize=12)
        ax.set_title(f'M = {M}', fontsize=13)
        ax.legend(fontsize=10)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Performance Model Validation: Theoretical vs Empirical Comparisons',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(save_dir, f'fig7_model_validation.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved fig7_model_validation to {save_dir}")


if __name__ == '__main__':
    validate_model(smoke='--smoke' in sys.argv)
