"""
CPU Baseline: Best Order Sort (BOS)

Based on:
    Roy, P. C., Islam, Md. M., & Deb, K. (2016).
    "Best Order Sort: A new algorithm to non-dominated sorting for
     evolutionary multi-objective optimization."
    In Proc. GECCO 2016 Companion, pp. 1113-1120.

The idea: sort the population by each objective independently, then
process solutions in a "best-first" order, assigning each to the
lowest-numbered front where it is not dominated by any existing member.

Input:  numpy array (N, M) — objective values (minimisation)
Output: numpy array (N,)   — front ranks (0-indexed)
"""

import time
import numpy as np


def bos_sort(objectives, seed=None, count_comparisons=False):
    """Best Order Sort.

    Parameters
    ----------
    objectives : np.ndarray, shape (N, M)
        Objective values for N solutions and M objectives (minimisation).
    seed : int or None
        Unused — present for API compatibility.
    count_comparisons : bool
        If True, additionally return the comparison count.

    Returns
    -------
    ranks : np.ndarray of int, shape (N,)
        0-indexed front rank.
    time_ms : float
        Wall-clock time in milliseconds.
    comparisons : int  (only if *count_comparisons*)

    Complexity
    ----------
    Best-case O(MN√N), worst-case O(MN²).
    """
    N, M = objectives.shape
    comparisons = 0

    t0 = time.perf_counter()

    # --- Step 1: For each objective, compute sorted order ---
    sorted_by_obj = np.empty((M, N), dtype=np.int64)
    rank_in_obj = np.empty((N, M), dtype=np.int64)
    for m in range(M):
        order = np.argsort(objectives[:, m], kind='stable')
        sorted_by_obj[m] = order
        rank_in_obj[order, m] = np.arange(N)

    # --- Step 2: Priority = sum of ranks across all objectives ---
    priority = rank_in_obj.sum(axis=1)
    process_order = np.argsort(priority, kind='stable')

    # --- Step 3: Assign to fronts ---
    fronts = []  # list of lists, fronts[k] = list of solution indices in front k
    ranks = np.full(N, -1, dtype=np.int32)

    for idx in process_order:
        placed = False
        for f_idx, front in enumerate(fronts):
            dominated_by_any = False
            for member in front:
                comparisons += 1
                if _dominates_fast(objectives[member], objectives[idx]):
                    dominated_by_any = True
                    break
            if not dominated_by_any:
                front.append(idx)
                ranks[idx] = f_idx
                placed = True
                break
        if not placed:
            fronts.append([idx])
            ranks[idx] = len(fronts) - 1

    t1 = time.perf_counter()
    time_ms = (t1 - t0) * 1000.0

    if count_comparisons:
        return ranks, time_ms, comparisons
    return ranks, time_ms


def _dominates_fast(a, b):
    """Check if *a* dominates *b* (minimisation). O(M).

    Parameters
    ----------
    a, b : np.ndarray of shape (M,)

    Returns
    -------
    bool
    """
    return bool(np.all(a <= b) and np.any(a < b))
