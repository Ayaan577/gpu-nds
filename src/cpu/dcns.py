"""
CPU Baseline: Divide-and-Conquer Non-Dominated Sort (DCNS)

Based on:
    Mishra, S., Mondal, S., & Saha, S. (2019).
    "Fast implementation of steady-state NSGA-II."
    Swarm and Evolutionary Computation, 44, 46-60.

The algorithm:
  1. Sort by first objective
  2. Recursively split into halves
  3. Sort each half
  4. Merge: check dominance of right-half against left-half
  5. Assign ranks based on domination counts

Enhancement: sum-of-objectives pre-sort to prune cross-partition comparisons.

Input:  numpy array (N, M) — objective values (minimisation)
Output: numpy array (N,)   — front ranks (0-indexed)
"""

import time
import numpy as np


class DCNSSorter:
    """Divide-and-Conquer Non-Dominated Sort with optional sum-of-objectives presort.

    Parameters
    ----------
    use_sum_presort : bool, default True
        If True, pre-sort solutions by sum of objectives before the main sort.
        This enables the sum-bound pruning optimisation that skips dominance
        checks when sum(a) > sum(b) (since then a cannot dominate b when
        all objectives are being minimised).

    Attributes
    ----------
    comparisons : int
        Total dominance comparisons performed in the last call to ``sort``.
    """

    def __init__(self, use_sum_presort=True):
        self.use_sum_presort = use_sum_presort
        self.comparisons = 0

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def sort(self, objectives, seed=None, count_comparisons=False):
        """Run DCNS on *objectives* and return front ranks.

        Parameters
        ----------
        objectives : np.ndarray, shape (N, M)
        seed : int or None
            Unused — present for API compatibility.
        count_comparisons : bool
            If True, additionally return the comparison count.

        Returns
        -------
        ranks : np.ndarray of int, shape (N,)
            0-indexed front rank for each solution (in original input order).
        time_ms : float
            Wall-clock time in milliseconds.
        comparisons : int  (only if *count_comparisons* is True)

        Complexity
        ----------
        Best-case O(N log N + MN), worst-case O(MN²).
        """
        N, M = objectives.shape
        self.comparisons = 0

        t0 = time.perf_counter()

        # Work on indexed copies so we can map back to original order.
        order = np.arange(N)
        obj = objectives.copy()

        # Optional sum-of-objectives pre-sort
        if self.use_sum_presort:
            sums = obj.sum(axis=1)
            presort_idx = np.argsort(sums)
            obj = obj[presort_idx]
            order = order[presort_idx]
        else:
            # Default: sort by first objective
            presort_idx = np.argsort(obj[:, 0])
            obj = obj[presort_idx]
            order = order[presort_idx]

        dom_count = np.zeros(N, dtype=np.int32)

        # Recursive divide-and-conquer
        self._dc_sort(obj, dom_count, 0, N)

        # Convert domination counts to ranks via iterative peeling
        ranks_local = self._counts_to_ranks(dom_count, obj, N)

        # Map back to original order
        ranks = np.empty(N, dtype=np.int32)
        ranks[order] = ranks_local

        t1 = time.perf_counter()
        time_ms = (t1 - t0) * 1000.0

        if count_comparisons:
            return ranks, time_ms, self.comparisons
        return ranks, time_ms

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    @staticmethod
    def _dominates(a, b):
        """Check if *a* dominates *b* (minimisation)."""
        return bool(np.all(a <= b) and np.any(a < b))

    def _dc_sort(self, obj, dom_count, lo, hi):
        """Recursively split [lo, hi) and merge dominance information.

        Parameters
        ----------
        obj : np.ndarray, shape (N, M)   — sorted slice
        dom_count : np.ndarray of int32   — domination counter (mutated)
        lo, hi : int
            Half-open interval into *obj*.
        """
        if hi - lo <= 1:
            return

        mid = (lo + hi) // 2
        self._dc_sort(obj, dom_count, lo, mid)
        self._dc_sort(obj, dom_count, mid, hi)

        # Merge: for every point in right half, check dominance from left half
        self._merge(obj, dom_count, lo, mid, hi)

    def _merge(self, obj, dom_count, lo, mid, hi):
        """Check if any left-half solution dominates a right-half solution.

        Uses the sum-of-objectives bound for pruning when enabled.
        """
        M = obj.shape[1]
        use_sum = self.use_sum_presort

        if use_sum:
            sums = obj.sum(axis=1)

        for j in range(mid, hi):
            for i in range(lo, mid):
                # Sum-bound pruning: if sum(left[i]) > sum(right[j]),
                # left[i] cannot dominate right[j].
                if use_sum and sums[i] > sums[j]:
                    continue
                self.comparisons += 1
                if self._dominates(obj[i], obj[j]):
                    dom_count[j] += 1

    @staticmethod
    def _counts_to_ranks(dom_count, obj, N):
        """Convert domination counts to 0-indexed front ranks by iterative peeling."""
        ranks = np.full(N, -1, dtype=np.int32)
        remaining = set(range(N))
        front = 0
        while remaining:
            current = [i for i in remaining if dom_count[i] == 0]
            if not current:
                # Safety: assign remaining to next front (shouldn't happen on valid input)
                for i in remaining:
                    ranks[i] = front
                break
            for i in current:
                ranks[i] = front
                remaining.discard(i)
            # Decrement dom_count for solutions dominated by current front members
            # (We don't track S[i] explicitly in DCNS, so we re-check.)
            for i in current:
                for j in list(remaining):
                    if DCNSSorter._dominates(obj[i], obj[j]):
                        dom_count[j] -= 1
            front += 1
        return ranks


# Module-level convenience wrapper
def dcns_sort(objectives, seed=None, use_sum_presort=True, count_comparisons=False):
    """Divide-and-Conquer Non-Dominated Sort (convenience wrapper).

    Parameters
    ----------
    objectives : np.ndarray, shape (N, M)
    seed : int or None
    use_sum_presort : bool
    count_comparisons : bool

    Returns
    -------
    ranks : np.ndarray of int, shape (N,)
    time_ms : float
    comparisons : int  (only if *count_comparisons*)
    """
    sorter = DCNSSorter(use_sum_presort=use_sum_presort)
    return sorter.sort(objectives, seed=seed, count_comparisons=count_comparisons)
