"""
CPU Baseline: Fast Non-Dominated Sort (Deb et al. 2002, NSGA-II)

Implements the classic O(MN²) non-dominated sorting algorithm from:
    Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002).
    "A fast and elitist multiobjective genetic algorithm: NSGA-II."
    IEEE Transactions on Evolutionary Computation, 6(2), 182-197.

Input:  numpy array of shape (N, M)  — N solutions, M objectives (minimisation)
Output: numpy array of shape (N,)    — front rank for each solution (0-indexed)
"""

import time
import numpy as np


def dominates(a, b):
    """Check if solution *a* dominates solution *b* (all objectives minimised).

    Parameters
    ----------
    a, b : array-like of shape (M,)
        Objective vectors.

    Returns
    -------
    bool
        True iff a[j] <= b[j] for all j AND a[j] < b[j] for at least one j.

    Complexity
    ----------
    O(M) per call.
    """
    return bool(np.all(a <= b) and np.any(a < b))


def fast_non_dominated_sort(objectives, seed=None, count_comparisons=False):
    """Classic NSGA-II fast non-dominated sort.

    Parameters
    ----------
    objectives : np.ndarray, shape (N, M)
        Objective values for N solutions and M objectives (minimisation).
    seed : int or None
        Unused — present for API compatibility with other sorters.
    count_comparisons : bool
        If True, also return the total number of dominance comparisons performed.

    Returns
    -------
    ranks : np.ndarray of int, shape (N,)
        Front rank of each solution (0-indexed: front 0 is the Pareto front).
    time_ms : float
        Wall-clock time in milliseconds.
    comparisons : int  (only if count_comparisons=True)
        Number of dominance comparisons performed.

    Complexity
    ----------
    O(M * N²) time, O(N²) space (domination sets).
    """
    N = objectives.shape[0]
    ranks = np.full(N, -1, dtype=np.int32)

    # S[p] = set of solutions dominated by p
    S = [[] for _ in range(N)]
    # n[p] = number of solutions that dominate p
    n = np.zeros(N, dtype=np.int32)

    comparisons = 0
    t0 = time.perf_counter()

    # --- Phase 1: compute domination sets and counts ---
    for p in range(N):
        for q in range(p + 1, N):
            comparisons += 1
            if dominates(objectives[p], objectives[q]):
                S[p].append(q)
                n[q] += 1
            elif dominates(objectives[q], objectives[p]):
                S[q].append(p)
                n[p] += 1

    # --- Phase 2: iterative front peeling ---
    current_front = []
    for p in range(N):
        if n[p] == 0:
            ranks[p] = 0
            current_front.append(p)

    front_idx = 0
    while current_front:
        next_front = []
        for p in current_front:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    ranks[q] = front_idx + 1
                    next_front.append(q)
        front_idx += 1
        current_front = next_front

    t1 = time.perf_counter()
    time_ms = (t1 - t0) * 1000.0

    if count_comparisons:
        return ranks, time_ms, comparisons
    return ranks, time_ms
