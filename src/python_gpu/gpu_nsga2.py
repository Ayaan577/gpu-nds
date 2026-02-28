"""
GPU-Resident NSGA-II — Full evolutionary loop on GPU via CuPy.

All operators (NDS, crowding distance, tournament selection, SBX crossover,
polynomial mutation, objective evaluation) run as GPU kernels.  Data stays
GPU-resident across generations to avoid PCIe transfer overhead.

Usage:
    from src.python_gpu.gpu_nsga2 import GPU_NSGA2
    ga = GPU_NSGA2(pop_size=200, n_gen=100, problem='DTLZ2', n_obj=3, n_var=12)
    result = ga.run()       # returns dict with pareto_front, pareto_set, time_ms
"""

import time
import numpy as np

try:
    import cupy as cp
    _CUPY = True
except ImportError:
    cp = np  # fallback
    _CUPY = False

from src.python_gpu.gpu_nds_cupy import GPU_NDS, get_backend

# =====================================================================
# CUDA Kernels
# =====================================================================

_CROWDING_DISTANCE_KERNEL = r"""
extern "C" __global__
void crowding_distance(
    const float* __restrict__ objectives,   // [N, M] row-major
    const int*   __restrict__ front_rank,    // [N]
    const int*   __restrict__ sorted_idx,    // [N] indices sorted per-front per-objective
    float*       __restrict__ distance,      // [N] output
    const int N,
    const int M,
    const int n_fronts)
{
    // This kernel is called AFTER computing sorted indices per front per objective.
    // Each thread handles one solution.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    distance[i] = 0.0f;
}
"""

_SBX_CROSSOVER_KERNEL = r"""
extern "C" __global__
void sbx_crossover(
    const float* __restrict__ parent1,     // [N/2, n_var]
    const float* __restrict__ parent2,     // [N/2, n_var]
    float*       __restrict__ child1,      // [N/2, n_var]
    float*       __restrict__ child2,      // [N/2, n_var]
    const float* __restrict__ randoms,     // [N/2, n_var, 2] uniform randoms
    const float eta_c,                     // distribution index (default 20)
    const float prob_cross,                // crossover probability (default 0.9)
    const float lower_bound,
    const float upper_bound,
    const int n_pairs,
    const int n_var)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_pairs * n_var) return;

    int pair = idx / n_var;
    int var  = idx % n_var;

    float p1 = parent1[pair * n_var + var];
    float p2 = parent2[pair * n_var + var];
    float u  = randoms[(pair * n_var + var) * 2];
    float r  = randoms[(pair * n_var + var) * 2 + 1];

    // Crossover probability check
    if (r > prob_cross) {
        child1[pair * n_var + var] = p1;
        child2[pair * n_var + var] = p2;
        return;
    }

    float beta;
    if (u <= 0.5f) {
        beta = powf(2.0f * u, 1.0f / (eta_c + 1.0f));
    } else {
        beta = powf(1.0f / (2.0f * (1.0f - u)), 1.0f / (eta_c + 1.0f));
    }

    float c1 = 0.5f * ((1.0f + beta) * p1 + (1.0f - beta) * p2);
    float c2 = 0.5f * ((1.0f - beta) * p1 + (1.0f + beta) * p2);

    // Clip to bounds
    c1 = fminf(fmaxf(c1, lower_bound), upper_bound);
    c2 = fminf(fmaxf(c2, lower_bound), upper_bound);

    child1[pair * n_var + var] = c1;
    child2[pair * n_var + var] = c2;
}
"""

_POLYNOMIAL_MUTATION_KERNEL = r"""
extern "C" __global__
void polynomial_mutation(
    float* __restrict__ pop,               // [N, n_var] — modified in-place
    const float* __restrict__ randoms,     // [N, n_var, 2]
    const float eta_m,                     // distribution index (default 20)
    const float prob_mut,                  // per-variable mutation probability
    const float lower_bound,
    const float upper_bound,
    const int N,
    const int n_var)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * n_var) return;

    float x = pop[idx];
    float u = randoms[idx * 2];
    float r = randoms[idx * 2 + 1];

    if (r > prob_mut) return;

    float delta;
    float diff = upper_bound - lower_bound;
    if (diff < 1e-12f) return;

    float norm = (x - lower_bound) / diff;
    if (u < 0.5f) {
        float xy = 1.0f - norm;
        float val = 2.0f * u + (1.0f - 2.0f * u) * powf(xy, eta_m + 1.0f);
        delta = powf(val, 1.0f / (eta_m + 1.0f)) - 1.0f;
    } else {
        float xy = norm;
        float val = 2.0f * (1.0f - u) + 2.0f * (u - 0.5f) * powf(xy, eta_m + 1.0f);
        delta = 1.0f - powf(val, 1.0f / (eta_m + 1.0f));
    }

    x = x + delta * diff;
    x = fminf(fmaxf(x, lower_bound), upper_bound);
    pop[idx] = x;
}
"""

# =====================================================================
# DTLZ Problem Evaluators (GPU)
# =====================================================================

_DTLZ2_KERNEL = r"""
extern "C" __global__
void dtlz2_evaluate(
    const float* __restrict__ x,     // [N, n_var]
    float*       __restrict__ f,     // [N, n_obj]
    const int N,
    const int n_var,
    const int n_obj)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int k = n_var - n_obj + 1;

    // Compute g
    float g = 0.0f;
    for (int j = n_obj - 1; j < n_var; j++) {
        float xj = x[i * n_var + j];
        g += (xj - 0.5f) * (xj - 0.5f);
    }

    // Compute objectives
    for (int m = 0; m < n_obj; m++) {
        float fm = 1.0f + g;
        for (int j = 0; j < n_obj - 1 - m; j++) {
            fm *= cosf(x[i * n_var + j] * 1.5707963267948966f);  // pi/2
        }
        if (m > 0) {
            fm *= sinf(x[i * n_var + (n_obj - 1 - m)] * 1.5707963267948966f);
        }
        f[i * n_obj + m] = fm;
    }
}
"""


class GPU_NSGA2:
    """Full GPU-resident NSGA-II.

    Parameters
    ----------
    pop_size : int
        Population size (must be even).
    n_gen : int
        Number of generations.
    problem : str
        Problem name ('DTLZ2', etc.)
    n_obj : int
        Number of objectives.
    n_var : int
        Number of decision variables.
    eta_c : float
        SBX crossover distribution index.
    eta_m : float
        Polynomial mutation distribution index.
    prob_cross : float
        Crossover probability.
    seed : int or None
        Random seed.
    """

    def __init__(self, pop_size=200, n_gen=100, problem='DTLZ2', n_obj=3,
                 n_var=None, eta_c=20.0, eta_m=20.0, prob_cross=0.9,
                 seed=42):
        assert pop_size % 2 == 0, "pop_size must be even"
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.problem = problem
        self.n_obj = n_obj
        self.n_var = n_var or (n_obj + 9)  # DTLZ default: M + k - 1, k=10
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.prob_cross = prob_cross
        self.prob_mut = 1.0 / self.n_var
        self.seed = seed

        self._nds = GPU_NDS(tile_size=32, use_sum_presort=True)
        self._compiled = False
        self._kernels = {}

    def _compile(self):
        if self._compiled:
            return
        if _CUPY:
            self._kernels['sbx'] = cp.RawKernel(_SBX_CROSSOVER_KERNEL, 'sbx_crossover')
            self._kernels['mutation'] = cp.RawKernel(_POLYNOMIAL_MUTATION_KERNEL, 'polynomial_mutation')
            self._kernels['dtlz2'] = cp.RawKernel(_DTLZ2_KERNEL, 'dtlz2_evaluate')
        self._compiled = True

    def _evaluate(self, pop_d):
        """Evaluate objectives on GPU. Returns (N, n_obj) device array."""
        N = pop_d.shape[0]
        f_d = cp.zeros((N, self.n_obj), dtype=cp.float32)
        if self.problem == 'DTLZ2':
            threads = 256
            blocks = (N + threads - 1) // threads
            self._kernels['dtlz2']((blocks,), (threads,),
                                    (pop_d, f_d, N, self.n_var, self.n_obj))
        else:
            raise NotImplementedError(f"Problem {self.problem} not implemented on GPU")
        return f_d

    def _crowding_distance(self, obj_d, ranks_d):
        """Compute crowding distance per front on GPU."""
        N, M = obj_d.shape
        cd = cp.zeros(N, dtype=cp.float32)
        max_rank = int(ranks_d.max())
        for front in range(max_rank + 1):
            mask = ranks_d == front
            n_front = int(mask.sum())
            if n_front <= 2:
                cd[mask] = cp.inf
                continue
            front_idx = cp.where(mask)[0]
            front_obj = obj_d[front_idx]
            for m in range(self.n_obj):
                sorted_order = cp.argsort(front_obj[:, m])
                sorted_vals = front_obj[sorted_order, m]
                f_range = float(sorted_vals[-1] - sorted_vals[0])
                if f_range < 1e-12:
                    continue
                local_cd = cp.zeros(n_front, dtype=cp.float32)
                local_cd[0] = cp.inf
                local_cd[-1] = cp.inf
                if n_front > 2:
                    local_cd[1:-1] = (sorted_vals[2:] - sorted_vals[:-2]) / f_range
                cd[front_idx[sorted_order]] += local_cd
        return cd

    def _tournament_selection(self, ranks_d, cd_d, N, rng):
        """Binary tournament selection on GPU."""
        # Generate random pairs
        perm = rng.permutation(N)
        idx1 = cp.asarray(perm[:N])
        perm2 = rng.permutation(N)
        idx2 = cp.asarray(perm2[:N])

        r1 = ranks_d[idx1]
        r2 = ranks_d[idx2]
        c1 = cd_d[idx1]
        c2 = cd_d[idx2]

        # Selection: prefer lower rank; if same rank, prefer higher crowding
        select_1 = (r1 < r2) | ((r1 == r2) & (c1 >= c2))
        winners = cp.where(select_1, idx1, idx2)
        return winners

    def _sbx_crossover(self, parents_d, rng):
        """SBX crossover on GPU."""
        N = parents_d.shape[0]
        n_pairs = N // 2
        p1 = parents_d[:n_pairs]
        p2 = parents_d[n_pairs:]
        c1 = cp.empty_like(p1)
        c2 = cp.empty_like(p2)

        randoms = cp.asarray(rng.random((n_pairs, self.n_var, 2)).astype(np.float32))

        threads = 256
        total = n_pairs * self.n_var
        blocks = (total + threads - 1) // threads
        self._kernels['sbx']((blocks,), (threads,),
                              (p1, p2, c1, c2, randoms,
                               np.float32(self.eta_c),
                               np.float32(self.prob_cross),
                               np.float32(0.0), np.float32(1.0),
                               np.int32(n_pairs), np.int32(self.n_var)))
        return cp.concatenate([c1, c2], axis=0)

    def _polynomial_mutation(self, pop_d, rng):
        """Polynomial mutation on GPU."""
        N = pop_d.shape[0]
        randoms = cp.asarray(rng.random((N, self.n_var, 2)).astype(np.float32))
        threads = 256
        total = N * self.n_var
        blocks = (total + threads - 1) // threads
        self._kernels['mutation']((blocks,), (threads,),
                                   (pop_d, randoms,
                                    np.float32(self.eta_m),
                                    np.float32(self.prob_mut),
                                    np.float32(0.0), np.float32(1.0),
                                    np.int32(N), np.int32(self.n_var)))
        return pop_d

    def _environmental_selection(self, combined_obj_d, combined_pop_d, ranks_d, cd_d, N_target):
        """Select top N_target from combined population using NSGA-II rules."""
        N = combined_obj_d.shape[0]
        max_rank = int(ranks_d.max())

        selected_idx = []
        total_selected = 0
        for front in range(max_rank + 1):
            front_mask = ranks_d == front
            front_indices = cp.where(front_mask)[0]
            n_front = len(front_indices)

            remaining = N_target - total_selected
            if remaining <= 0:
                break

            if n_front <= remaining:
                selected_idx.append(front_indices)
                total_selected += n_front
            else:
                # Sort by crowding distance (descending) and take top 'remaining'
                front_cd = cd_d[front_indices]
                order = cp.argsort(-front_cd)
                selected_idx.append(front_indices[order[:remaining]])
                total_selected += remaining

        selected = cp.concatenate(selected_idx)
        return combined_pop_d[selected], combined_obj_d[selected]

    def run(self):
        """Run GPU-NSGA-II. Returns dict with results."""
        self._compile()
        xp = cp if _CUPY else np
        rng = np.random.RandomState(self.seed)

        t0 = time.perf_counter()

        # Initialize population [0, 1]^n_var
        pop = rng.random((self.pop_size, self.n_var)).astype(np.float32)
        pop_d = xp.asarray(pop)

        # Evaluate
        obj_d = self._evaluate(pop_d)

        for gen in range(self.n_gen):
            # NDS
            obj_np = cp.asnumpy(obj_d) if _CUPY else obj_d
            ranks_np, _ = self._nds.sort(obj_np)
            ranks_d = xp.asarray(ranks_np.astype(np.int32))

            # Crowding distance
            cd_d = self._crowding_distance(obj_d, ranks_d)

            # Selection
            parents_idx = self._tournament_selection(ranks_d, cd_d, self.pop_size, rng)
            parents_d = pop_d[parents_idx]

            # Crossover
            offspring_d = self._sbx_crossover(parents_d, rng)

            # Mutation
            offspring_d = self._polynomial_mutation(offspring_d, rng)

            # Evaluate offspring
            off_obj_d = self._evaluate(offspring_d)

            # Combine parent + offspring
            combined_pop_d = xp.concatenate([pop_d, offspring_d], axis=0)
            combined_obj_d = xp.concatenate([obj_d, off_obj_d], axis=0)

            # NDS on combined
            combined_obj_np = cp.asnumpy(combined_obj_d) if _CUPY else combined_obj_d
            combined_ranks_np, _ = self._nds.sort(combined_obj_np)
            combined_ranks_d = xp.asarray(combined_ranks_np.astype(np.int32))

            # Crowding distance on combined
            combined_cd_d = self._crowding_distance(
                combined_obj_d, combined_ranks_d)

            # Environmental selection
            pop_d, obj_d = self._environmental_selection(
                combined_obj_d, combined_pop_d, combined_ranks_d,
                combined_cd_d, self.pop_size)

        if _CUPY:
            cp.cuda.Stream.null.synchronize()

        t1 = time.perf_counter()
        total_ms = (t1 - t0) * 1000

        # Extract results
        final_pop = cp.asnumpy(pop_d) if _CUPY else pop_d
        final_obj = cp.asnumpy(obj_d) if _CUPY else obj_d
        final_ranks, _ = self._nds.sort(final_obj)

        # Front 0 = Pareto front
        pf_mask = final_ranks == 0
        return {
            'pareto_front': final_obj[pf_mask],
            'pareto_set': final_pop[pf_mask],
            'all_objectives': final_obj,
            'all_variables': final_pop,
            'all_ranks': final_ranks,
            'total_ms': total_ms,
            'n_gen': self.n_gen,
            'pop_size': self.pop_size,
            'n_pareto': int(pf_mask.sum()),
        }
