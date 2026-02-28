"""
CPU-NSGA-II baseline for end-to-end comparison with GPU-NSGA-II.
Uses C++ NDS kernel + NumPy/Python for other operators.
"""
import time
import numpy as np
from src.cpu_cpp.cpu_nds_wrapper import nsga2_sort as cpp_nsga2_sort


def _dtlz2_evaluate(x, n_obj):
    """Evaluate DTLZ2 objectives on CPU."""
    N, n_var = x.shape
    k = n_var - n_obj + 1
    g = np.sum((x[:, n_obj-1:] - 0.5)**2, axis=1)  # (N,)

    f = np.zeros((N, n_obj), dtype=np.float32)
    for m in range(n_obj):
        fm = 1.0 + g
        for j in range(n_obj - 1 - m):
            fm *= np.cos(x[:, j] * np.pi / 2)
        if m > 0:
            fm *= np.sin(x[:, n_obj - 1 - m] * np.pi / 2)
        f[:, m] = fm
    return f


def _crowding_distance(obj, ranks):
    """Compute crowding distance."""
    N = obj.shape[0]
    M = obj.shape[1]
    cd = np.zeros(N, dtype=np.float64)

    max_rank = ranks.max()
    for front in range(max_rank + 1):
        mask = ranks == front
        n_front = mask.sum()
        if n_front <= 2:
            cd[mask] = np.inf
            continue

        front_idx = np.where(mask)[0]
        front_obj = obj[front_idx]

        for m in range(M):
            order = np.argsort(front_obj[:, m])
            sorted_vals = front_obj[order, m]
            f_range = sorted_vals[-1] - sorted_vals[0]
            if f_range < 1e-12:
                continue
            local_cd = np.zeros(n_front)
            local_cd[0] = np.inf
            local_cd[-1] = np.inf
            if n_front > 2:
                local_cd[1:-1] = (sorted_vals[2:] - sorted_vals[:-2]) / f_range
            cd[front_idx[order]] += local_cd

    return cd


def _tournament_selection(ranks, cd, N, rng):
    """Binary tournament selection."""
    idx1 = rng.permutation(N)
    idx2 = rng.permutation(N)

    r1, r2 = ranks[idx1], ranks[idx2]
    c1, c2 = cd[idx1], cd[idx2]

    select_1 = (r1 < r2) | ((r1 == r2) & (c1 >= c2))
    winners = np.where(select_1, idx1, idx2)
    return winners


def _sbx_crossover(parents, eta_c, prob_cross, rng):
    """SBX crossover."""
    N = parents.shape[0]
    n_var = parents.shape[1]
    n_pairs = N // 2

    p1 = parents[:n_pairs]
    p2 = parents[n_pairs:]
    c1 = np.copy(p1)
    c2 = np.copy(p2)

    u = rng.random((n_pairs, n_var))
    cross_mask = rng.random((n_pairs, n_var)) < prob_cross

    beta = np.where(u <= 0.5,
                    (2.0 * u) ** (1.0 / (eta_c + 1.0)),
                    (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1.0)))

    c1[cross_mask] = (0.5 * ((1 + beta) * p1 + (1 - beta) * p2))[cross_mask]
    c2[cross_mask] = (0.5 * ((1 - beta) * p1 + (1 + beta) * p2))[cross_mask]

    c1 = np.clip(c1, 0.0, 1.0)
    c2 = np.clip(c2, 0.0, 1.0)

    return np.concatenate([c1, c2], axis=0)


def _polynomial_mutation(pop, eta_m, prob_mut, rng):
    """Polynomial mutation."""
    N, n_var = pop.shape
    mask = rng.random((N, n_var)) < prob_mut
    u = rng.random((N, n_var))

    delta = np.zeros_like(pop)
    norm = pop  # already in [0,1]

    low_mask = u < 0.5
    high_mask = ~low_mask

    xy_l = 1.0 - norm
    val_l = 2.0 * u + (1.0 - 2.0 * u) * (xy_l ** (eta_m + 1.0))
    delta_l = val_l ** (1.0 / (eta_m + 1.0)) - 1.0

    xy_h = norm
    val_h = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy_h ** (eta_m + 1.0))
    delta_h = 1.0 - val_h ** (1.0 / (eta_m + 1.0))

    delta = np.where(low_mask, delta_l, delta_h)

    pop[mask] = pop[mask] + delta[mask]
    pop = np.clip(pop, 0.0, 1.0)
    return pop


class CPU_NSGA2:
    """CPU-based NSGA-II using C++ NDS + NumPy operators."""

    def __init__(self, pop_size=200, n_gen=100, problem='DTLZ2', n_obj=3,
                 n_var=None, eta_c=20.0, eta_m=20.0, prob_cross=0.9,
                 seed=42):
        assert pop_size % 2 == 0
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.problem = problem
        self.n_obj = n_obj
        self.n_var = n_var or (n_obj + 9)
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.prob_cross = prob_cross
        self.prob_mut = 1.0 / self.n_var
        self.seed = seed

    def _evaluate(self, pop):
        if self.problem == 'DTLZ2':
            return _dtlz2_evaluate(pop, self.n_obj)
        raise NotImplementedError(f"Problem {self.problem}")

    def run(self):
        """Run CPU-NSGA-II. Returns dict with results."""
        rng = np.random.RandomState(self.seed)
        t0 = time.perf_counter()

        # Initialize
        pop = rng.random((self.pop_size, self.n_var)).astype(np.float32)
        obj = self._evaluate(pop)

        for gen in range(self.n_gen):
            # NDS (C++)
            ranks, _ = cpp_nsga2_sort(obj)

            # Crowding distance
            cd = _crowding_distance(obj, ranks)

            # Selection
            parents_idx = _tournament_selection(ranks, cd, self.pop_size, rng)
            parents = pop[parents_idx].copy()

            # Crossover
            offspring = _sbx_crossover(parents, self.eta_c, self.prob_cross, rng)

            # Mutation
            offspring = _polynomial_mutation(offspring, self.eta_m, self.prob_mut, rng)

            # Evaluate
            off_obj = self._evaluate(offspring)

            # Combine
            combined_pop = np.concatenate([pop, offspring], axis=0)
            combined_obj = np.concatenate([obj, off_obj], axis=0)

            # NDS on combined
            combined_ranks, _ = cpp_nsga2_sort(combined_obj)

            # Crowding distance on combined
            combined_cd = _crowding_distance(combined_obj, combined_ranks)

            # Environmental selection
            selected = _environmental_selection(
                combined_ranks, combined_cd, self.pop_size)
            pop = combined_pop[selected]
            obj = combined_obj[selected]

        t1 = time.perf_counter()
        total_ms = (t1 - t0) * 1000

        final_ranks, _ = cpp_nsga2_sort(obj)
        pf_mask = final_ranks == 0

        return {
            'pareto_front': obj[pf_mask],
            'pareto_set': pop[pf_mask],
            'all_objectives': obj,
            'all_ranks': final_ranks,
            'total_ms': total_ms,
            'n_gen': self.n_gen,
            'pop_size': self.pop_size,
            'n_pareto': int(pf_mask.sum()),
        }


def _environmental_selection(ranks, cd, N_target):
    """Select top N_target indices using NSGA-II rules."""
    max_rank = ranks.max()
    selected = []

    for front in range(max_rank + 1):
        front_idx = np.where(ranks == front)[0]
        remaining = N_target - len(selected)
        if remaining <= 0:
            break
        if len(front_idx) <= remaining:
            selected.extend(front_idx.tolist())
        else:
            front_cd = cd[front_idx]
            order = np.argsort(-front_cd)
            selected.extend(front_idx[order[:remaining]].tolist())

    return np.array(selected, dtype=np.int64)
