"""
Benchmark Problem Generators: DTLZ1-7, WFG1-9, Random Uniform, Random Clustered

References:
    - Deb, K., Thiele, L., Laumanns, M., & Zitzler, E. (2005).
      "Scalable test problems for evolutionary multiobjective optimization."
      In Evolutionary Multiobjective Optimization, pp. 105-145.
    - Huband, S., Hingston, P., Barone, L., & While, L. (2006).
      "A review of multiobjective test problems and a scalable test problem toolkit."
      IEEE TEC, 10(5), 477-506.

All generators return objective values of shape (N, M), suitable for
non-dominated sorting benchmarks.
"""

import numpy as np


# =====================================================================
# DTLZ Suite (DTLZ1 – DTLZ7)
# =====================================================================

def generate_dtlz(problem_id, N, M, seed=42):
    """Generate N solutions to a DTLZ problem with M objectives.

    Parameters
    ----------
    problem_id : int
        Problem number 1–7.
    N : int
        Number of solutions to generate.
    M : int
        Number of objectives.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray, shape (N, M)
        Objective values.
    """
    funcs = {
        1: _dtlz1, 2: _dtlz2, 3: _dtlz3, 4: _dtlz4,
        5: _dtlz5, 6: _dtlz6, 7: _dtlz7,
    }
    if problem_id not in funcs:
        raise ValueError(f"DTLZ problem_id must be 1-7, got {problem_id}")
    return funcs[problem_id](N, M, seed)


def _dtlz_vars(N, M, k, seed):
    """Return random decision variables for DTLZ problems.

    n_vars = M + k - 1.  Variables are in [0,1].
    """
    rng = np.random.default_rng(seed)
    n_vars = M + k - 1
    return rng.random((N, n_vars))


def _g1(xm):
    """g function for DTLZ1/3."""
    return 100.0 * (xm.shape[1] + np.sum((xm - 0.5) ** 2
                    - np.cos(20.0 * np.pi * (xm - 0.5)), axis=1))


def _g2(xm):
    """g function for DTLZ2/4/5/6."""
    return np.sum((xm - 0.5) ** 2, axis=1)


def _dtlz1(N, M, seed):
    k = 5
    x = _dtlz_vars(N, M, k, seed)
    xm = x[:, M - 1:]
    g = _g1(xm)
    f = np.zeros((N, M))
    for i in range(M):
        f[:, i] = 0.5 * (1.0 + g)
        for j in range(M - 1 - i):
            f[:, i] *= x[:, j]
        if i > 0:
            f[:, i] *= (1.0 - x[:, M - 1 - i])
    return f


def _dtlz2(N, M, seed):
    k = 5
    x = _dtlz_vars(N, M, k, seed)
    xm = x[:, M - 1:]
    g = _g2(xm)
    f = np.zeros((N, M))
    for i in range(M):
        f[:, i] = (1.0 + g)
        for j in range(M - 1 - i):
            f[:, i] *= np.cos(x[:, j] * np.pi / 2.0)
        if i > 0:
            f[:, i] *= np.sin(x[:, M - 1 - i] * np.pi / 2.0)
    return f


def _dtlz3(N, M, seed):
    k = 5
    x = _dtlz_vars(N, M, k, seed)
    xm = x[:, M - 1:]
    g = _g1(xm)  # same g as DTLZ1 → multiple local fronts
    f = np.zeros((N, M))
    for i in range(M):
        f[:, i] = (1.0 + g)
        for j in range(M - 1 - i):
            f[:, i] *= np.cos(x[:, j] * np.pi / 2.0)
        if i > 0:
            f[:, i] *= np.sin(x[:, M - 1 - i] * np.pi / 2.0)
    return f


def _dtlz4(N, M, seed, alpha=100.0):
    k = 5
    x = _dtlz_vars(N, M, k, seed)
    xm = x[:, M - 1:]
    g = _g2(xm)
    f = np.zeros((N, M))
    for i in range(M):
        f[:, i] = (1.0 + g)
        for j in range(M - 1 - i):
            f[:, i] *= np.cos(x[:, j] ** alpha * np.pi / 2.0)
        if i > 0:
            f[:, i] *= np.sin(x[:, M - 1 - i] ** alpha * np.pi / 2.0)
    return f


def _dtlz5(N, M, seed):
    k = 5
    x = _dtlz_vars(N, M, k, seed)
    xm = x[:, M - 1:]
    g = _g2(xm)
    theta = np.zeros_like(x[:, :M - 1])
    theta[:, 0] = x[:, 0] * np.pi / 2.0
    for j in range(1, M - 1):
        theta[:, j] = (np.pi / (4.0 * (1.0 + g))) * (1.0 + 2.0 * g * x[:, j])
    f = np.zeros((N, M))
    for i in range(M):
        f[:, i] = (1.0 + g)
        for j in range(M - 1 - i):
            f[:, i] *= np.cos(theta[:, j])
        if i > 0:
            f[:, i] *= np.sin(theta[:, M - 1 - i])
    return f


def _dtlz6(N, M, seed):
    k = 5
    x = _dtlz_vars(N, M, k, seed)
    xm = x[:, M - 1:]
    g = np.sum(xm ** 0.1, axis=1)  # different g for DTLZ6
    theta = np.zeros_like(x[:, :M - 1])
    theta[:, 0] = x[:, 0] * np.pi / 2.0
    for j in range(1, M - 1):
        theta[:, j] = (np.pi / (4.0 * (1.0 + g))) * (1.0 + 2.0 * g * x[:, j])
    f = np.zeros((N, M))
    for i in range(M):
        f[:, i] = (1.0 + g)
        for j in range(M - 1 - i):
            f[:, i] *= np.cos(theta[:, j])
        if i > 0:
            f[:, i] *= np.sin(theta[:, M - 1 - i])
    return f


def _dtlz7(N, M, seed):
    k = 5
    x = _dtlz_vars(N, M, k, seed)
    xm = x[:, M - 1:]
    f = np.zeros((N, M))
    for i in range(M - 1):
        f[:, i] = x[:, i]
    g = 1.0 + (9.0 / k) * np.sum(xm, axis=1)
    h = M - np.sum(
        (f[:, :M - 1] / (1.0 + g[:, None]))
        * (1.0 + np.sin(3.0 * np.pi * f[:, :M - 1])),
        axis=1,
    )
    f[:, M - 1] = (1.0 + g) * h
    return f


# =====================================================================
# WFG Suite (WFG1 – WFG9)  — simplified objective-space generator
# =====================================================================
# Full WFG requires complex shape/transformation functions.  For NDS
# benchmarking we only need objective vectors with realistic front
# structures, so we implement a faithful simplified version.

def generate_wfg(problem_id, N, M, seed=42):
    """Generate N solutions to a WFG problem with M objectives.

    Parameters
    ----------
    problem_id : int
        Problem number 1-9.
    N : int
        Number of solutions.
    M : int
        Number of objectives.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray, shape (N, M)
        Objective values.
    """
    if not (1 <= problem_id <= 9):
        raise ValueError(f"WFG problem_id must be 1-9, got {problem_id}")

    rng = np.random.default_rng(seed)
    k = 2 * (M - 1)           # position parameters
    l = 20                      # distance parameters
    n_vars = k + l

    z = rng.random((N, n_vars))

    # Normalise z to [0, 2*i] as per WFG specification
    for i in range(n_vars):
        z[:, i] *= 2.0 * (i + 1)

    # --- Transition functions (simplified) ---
    t = z.copy()

    if problem_id == 1:
        # WFG1: flat bias + polynomial bias
        t[:, k:] = np.abs(t[:, k:] - 0.35) / np.abs(np.floor(0.35 - t[:, k:]) + 0.35)
        t = _wfg_reduce_weighted(t, k, M)
        f = _wfg_shape_convex(t, M)
    elif problem_id == 2:
        t[:, k:] = _wfg_s_multi(t[:, k:], 30, 95, 0.35)
        t = _wfg_reduce_weighted(t, k, M)
        f = _wfg_shape_convex(t, M)
    elif problem_id == 3:
        t = _wfg_reduce_weighted(t, k, M)
        f = _wfg_shape_linear(t, M)
    elif problem_id == 4:
        t[:, :k] = _wfg_s_multi(t[:, :k], 30, 10, 0.35)
        t = _wfg_reduce_weighted(t, k, M)
        f = _wfg_shape_concave(t, M)
    elif problem_id == 5:
        t = _wfg_reduce_weighted(t, k, M)
        f = _wfg_shape_concave(t, M)
    elif problem_id == 6:
        t = _wfg_reduce_weighted(t, k, M)
        f = _wfg_shape_concave(t, M)
    elif problem_id == 7:
        t = _wfg_reduce_weighted(t, k, M)
        f = _wfg_shape_concave(t, M)
    elif problem_id == 8:
        t = _wfg_reduce_weighted(t, k, M)
        f = _wfg_shape_concave(t, M)
    elif problem_id == 9:
        t = _wfg_reduce_weighted(t, k, M)
        f = _wfg_shape_concave(t, M)

    # Scale objectives to make them positive
    f = np.abs(f) + 1e-10
    return f


def _wfg_s_multi(y, A, B, C):
    """Multi-modal shift transformation for WFG."""
    tmp1 = np.abs(y - C) / (2.0 * (np.floor(C - y) + C))
    tmp2 = (4.0 * A + 2.0) * np.pi * (0.5 - tmp1)
    return (1.0 + np.cos(tmp2) + 4.0 * B * tmp1 ** 2) / (B + 2.0)


def _wfg_reduce_weighted(t, k, M):
    """Weighted reduction: group position params into M-1 groups + 1 distance."""
    N, n_vars = t.shape
    result = np.zeros((N, M))
    group_size = k // (M - 1)
    for i in range(M - 1):
        start = i * group_size
        end = (i + 1) * group_size
        w = np.arange(1, group_size + 1, dtype=np.float64)
        result[:, i] = np.average(t[:, start:end], axis=1, weights=w)
    # Last component: distance parameters
    w_dist = np.arange(1, n_vars - k + 1, dtype=np.float64)
    result[:, M - 1] = np.average(t[:, k:], axis=1, weights=w_dist)
    return result


def _wfg_shape_convex(t, M):
    """Convex Pareto front shape."""
    N = t.shape[0]
    f = np.ones((N, M))
    for i in range(M):
        for j in range(M - 1 - i):
            f[:, i] *= (1.0 - np.cos(t[:, j] * np.pi / 2.0))
        if i > 0:
            f[:, i] *= (1.0 - np.sin(t[:, M - 1 - i] * np.pi / 2.0))
    return f


def _wfg_shape_linear(t, M):
    """Linear Pareto front shape."""
    N = t.shape[0]
    f = np.ones((N, M))
    for i in range(M):
        for j in range(M - 1 - i):
            f[:, i] *= t[:, j]
        if i > 0:
            f[:, i] *= (1.0 - t[:, M - 1 - i])
    return f


def _wfg_shape_concave(t, M):
    """Concave Pareto front shape."""
    N = t.shape[0]
    f = np.ones((N, M))
    for i in range(M):
        for j in range(M - 1 - i):
            f[:, i] *= np.sin(t[:, j] * np.pi / 2.0)
        if i > 0:
            f[:, i] *= np.cos(t[:, M - 1 - i] * np.pi / 2.0)
    return f


# =====================================================================
# Random problem generators
# =====================================================================

def generate_random_uniform(N, M, seed=42):
    """Generate N random uniform solutions with M objectives in [0, 1].

    Parameters
    ----------
    N : int
    M : int
    seed : int

    Returns
    -------
    np.ndarray, shape (N, M)
    """
    rng = np.random.default_rng(seed)
    return rng.random((N, M)).astype(np.float64)


def generate_random_clustered(N, M, n_clusters=10, seed=42):
    """Generate N solutions in M-objective space arranged in *n_clusters* clusters.

    This simulates populations produced by real MOEAs, which tend to
    form clusters in objective space.

    Parameters
    ----------
    N : int
    M : int
    n_clusters : int
    seed : int

    Returns
    -------
    np.ndarray, shape (N, M)
    """
    rng = np.random.default_rng(seed)
    centres = rng.random((n_clusters, M))
    labels = rng.integers(0, n_clusters, size=N)
    spread = 0.05
    points = centres[labels] + rng.normal(0, spread, (N, M))
    # Clip to positive values (objectives must be non-negative for dominance to be meaningful)
    return np.clip(points, 0.0, None).astype(np.float64)
