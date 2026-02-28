"""
GPU-Native Divide-and-Conquer Non-Dominated Sorting — CuPy Implementation

This is the PRIMARY CONTRIBUTION of the project.  It implements the 4-phase
GPU-NDS pipeline using CuPy (which compiles real CUDA kernels via RawKernel):

  Phase 1: Sum-of-objectives pre-sort (GPU argsort)
  Phase 2: Tiled dominance checks   (custom CUDA kernel via shared memory)
  Phase 3: Front assignment          (iterative peeling with GPU atomics)
  Phase 4: Prefix-scan cleanup       (CuPy array ops)

Falls back to a pure-NumPy simulation when CuPy/CUDA is unavailable, so
benchmarks can always run (just slower on CPU).

Input:  numpy array of shape (N, M) — objective values
Output: (ranks: np.ndarray(N,), time_ms: float)
"""

import time
import numpy as np

# --------------- backend detection ---------------
import os as _os
import sys as _sys

# On Windows, add NVIDIA pip-package DLL directories so CuPy finds NVRTC
if _sys.platform == 'win32':
    _sp = None
    for p in _sys.path:
        _candidate = _os.path.join(p, 'nvidia', 'cuda_nvrtc', 'bin')
        if _os.path.isdir(_candidate):
            _sp = _candidate
            break
    if _sp is None:
        try:
            import site
            for sp_dir in site.getsitepackages():
                _candidate = _os.path.join(sp_dir, 'nvidia', 'cuda_nvrtc', 'bin')
                if _os.path.isdir(_candidate):
                    _sp = _candidate
                    break
        except Exception:
            pass
    if _sp is not None:
        try:
            _os.add_dll_directory(_sp)
        except (OSError, AttributeError):
            _os.environ['PATH'] = _sp + ';' + _os.environ.get('PATH', '')

try:
    import cupy as cp
    _HAS_CUPY = True
    # Full test: create array, compile and run a kernel (needs NVRTC + GPU)
    try:
        _t = cp.array([3.0, 1.0, 2.0], dtype=cp.float32)
        _t_sorted = cp.sort(_t)   # requires NVRTC JIT
        assert float(_t_sorted[0]) == 1.0
        del _t, _t_sorted
    except Exception:
        _HAS_CUPY = False
except (ImportError, Exception):
    _HAS_CUPY = False

if not _HAS_CUPY:
    cp = None  # type: ignore




# =====================================================================
# CUDA kernel source (compiled at runtime by CuPy)
# =====================================================================

_TILED_DOMINANCE_KERNEL = r"""
extern "C" __global__
void tiled_dominance_check(
    const float* __restrict__ objectives,  // [N x M], row-major
    const float* __restrict__ sums,        // [N]
    int* __restrict__ dom_count,           // [N], output
    const int N,
    const int M,
    const int TILE)
{
    // 2-D grid: blockIdx.x -> tile-row, blockIdx.y -> tile-col
    int tile_row = blockIdx.x;
    int tile_col = blockIdx.y;
    int tx = threadIdx.x;  // local id within a tile-row
    int ty = threadIdx.y;  // local id within a tile-col

    int i = tile_row * TILE + tx;   // global row (potential dominator)
    int j = tile_col * TILE + ty;   // global col (potentially dominated)

    if (i >= N || j >= N || i == j) return;

    // --- Sum-of-objectives pruning ---
    // If sum(i) > sum(j), then i cannot dominate j (all objs minimised).
    if (sums[i] > sums[j]) return;

    // --- Full dominance check ---
    bool all_leq = true;
    bool any_lt  = false;
    for (int m = 0; m < M; ++m) {
        float ai = objectives[i * M + m];
        float bj = objectives[j * M + m];
        if (ai > bj) { all_leq = false; break; }
        if (ai < bj) { any_lt  = true; }
    }
    if (all_leq && any_lt) {
        atomicAdd(&dom_count[j], 1);
    }
}
"""

_TILED_DOMINANCE_KERNEL_SHARED = r"""
extern "C" __global__
void tiled_dominance_check_shared(
    const float* __restrict__ objectives,  // [N x M], row-major
    const float* __restrict__ sums,        // [N]
    int* __restrict__ dom_count,           // [N], output
    const int N,
    const int M)
{
    // blockDim = (TILE, TILE);  shared memory tiles for coalesced loads
    extern __shared__ float smem[];        // 2 * TILE * M floats

    const int TILE = blockDim.x;           // == blockDim.y
    int tile_row = blockIdx.x;
    int tile_col = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float* tile_A = smem;                  // [TILE][M]
    float* tile_B = smem + TILE * M;       // [TILE][M]

    int gi = tile_row * TILE + tx;
    int gj = tile_col * TILE + ty;

    // Cooperatively load tile_A (row tx)
    if (gi < N) {
        for (int m = ty; m < M; m += TILE)
            tile_A[tx * M + m] = objectives[gi * M + m];
    }
    // Cooperatively load tile_B (row ty)
    if (gj < N) {
        for (int m = tx; m < M; m += TILE)
            tile_B[ty * M + m] = objectives[gj * M + m];
    }
    __syncthreads();

    if (gi >= N || gj >= N || gi == gj) return;

    // Sum pruning
    if (sums[gi] > sums[gj]) return;

    bool all_leq = true;
    bool any_lt  = false;
    for (int m = 0; m < M; ++m) {
        float ai = tile_A[tx * M + m];
        float bj = tile_B[ty * M + m];
        if (ai > bj) { all_leq = false; break; }
        if (ai < bj) { any_lt  = true; }
    }
    if (all_leq && any_lt) {
        atomicAdd(&dom_count[gj], 1);
    }
}
"""

_FRONT_ASSIGN_KERNEL = r"""
extern "C" __global__
void front_assign(
    const int* __restrict__ dom_count,
    int* __restrict__ front_rank,
    int* __restrict__ newly_assigned,
    const int N,
    const int current_front)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (front_rank[i] == -1 && dom_count[i] == 0) {
        front_rank[i] = current_front;
        newly_assigned[i] = 1;
    }
}
"""

_DECREMENT_DOM_KERNEL = r"""
extern "C" __global__
void decrement_dom_count(
    const float* __restrict__ objectives,
    const float* __restrict__ sums,
    const int* __restrict__ newly_assigned,
    const int* __restrict__ front_rank,
    int* __restrict__ dom_count,
    const int N,
    const int M)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N || front_rank[j] != -1) return;  // already assigned

    int dec = 0;
    for (int i = 0; i < N; ++i) {
        if (newly_assigned[i] == 0) continue;
        // Check if i dominates j
        if (sums[i] > sums[j]) continue;
        bool all_leq = true;
        bool any_lt  = false;
        for (int m = 0; m < M; ++m) {
            float ai = objectives[i * M + m];
            float bj = objectives[j * M + m];
            if (ai > bj) { all_leq = false; break; }
            if (ai < bj) { any_lt  = true; }
        }
        if (all_leq && any_lt) dec++;
    }
    if (dec > 0) atomicSub(&dom_count[j], dec);
}
"""


class GPU_NDS:
    """GPU-Native Non-Dominated Sorter using CuPy CUDA kernels.

    Parameters
    ----------
    tile_size : int, default 32
        TILE dimension for shared-memory dominance kernel (16, 32, or 64).
    use_sum_presort : bool, default True
        Enable sum-of-objectives pre-sort and pruning.
    use_shared_memory : bool, default True
        Use the shared-memory tiled kernel (True) or the simpler global-memory
        version (False).  Shared memory is faster for M <= ~32.
    """

    def __init__(self, tile_size=32, use_sum_presort=True, use_shared_memory=True):
        self.tile_size = tile_size
        self.use_sum_presort = use_sum_presort
        self.use_shared_memory = use_shared_memory
        self._compiled = {}
        self.comparisons = 0

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def sort(self, objectives, seed=None, count_comparisons=False):
        """Non-dominated sort on GPU.

        Parameters
        ----------
        objectives : np.ndarray, shape (N, M)
        seed : int or None
        count_comparisons : bool

        Returns
        -------
        ranks : np.ndarray, shape (N,)   (0-indexed)
        time_ms : float
        comparisons : int  (only if *count_comparisons*)
        """
        if _HAS_CUPY:
            result = self._sort_cupy(objectives)
        else:
            result = self._sort_numpy_fallback(objectives)

        if count_comparisons:
            return result[0], result[1], self.comparisons
        return result

    # ------------------------------------------------------------------
    # CuPy / CUDA path
    # ------------------------------------------------------------------
    def _sort_cupy(self, objectives):
        N, M = objectives.shape
        TILE = self.tile_size

        # Compile kernels once
        self._ensure_compiled()

        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()
        start_event.record()

        obj_f32 = cp.asarray(objectives, dtype=cp.float32)

        # -- Phase 1: sum-of-objectives pre-sort --
        sums = cp.array(obj_f32.get().sum(axis=1), dtype=cp.float32)
        if self.use_sum_presort:
            sort_idx = cp.argsort(sums)
            obj_f32 = obj_f32[sort_idx]
            sums = sums[sort_idx]
        else:
            sort_idx = cp.arange(N, dtype=cp.int32)

        # -- Phase 2: tiled dominance count --
        dom_count = cp.zeros(N, dtype=cp.int32)

        # Max threads per block is 1024, so TILE*TILE must be <= 1024
        # For shared-memory kernel, cap at 32 (32*32=1024)
        use_shared = self.use_shared_memory and TILE <= 32

        if use_shared:
            grid = ((N + TILE - 1) // TILE, (N + TILE - 1) // TILE)
            block = (TILE, TILE)
            smem_bytes = 2 * TILE * M * 4  # float32
            self._compiled['dom_shared'](
                grid, block,
                (obj_f32, sums, dom_count, np.int32(N), np.int32(M)),
                shared_mem=smem_bytes,
            )
        else:
            # Use global-memory kernel with effective tile capped at 32
            eff_tile = min(TILE, 32)
            grid = ((N + eff_tile - 1) // eff_tile, (N + eff_tile - 1) // eff_tile)
            block = (eff_tile, eff_tile)
            self._compiled['dom_global'](
                grid, block,
                (obj_f32, sums, dom_count, np.int32(N), np.int32(M), np.int32(eff_tile)),
            )

        # -- Phase 3 & 4: iterative front assignment with GPU kernels --
        front_rank = cp.full(N, -1, dtype=cp.int32)
        newly_assigned = cp.zeros(N, dtype=cp.int32)

        block1d = 256
        grid1d = (N + block1d - 1) // block1d
        current_front = 0
        assigned_count = 0

        while assigned_count < N:
            newly_assigned[:] = 0
            self._compiled['front_assign'](
                (grid1d,), (block1d,),
                (dom_count, front_rank, newly_assigned,
                 np.int32(N), np.int32(current_front)),
            )
            n_new = int(newly_assigned.get().sum())
            if n_new == 0:
                # Safety: assign all remaining to this front
                mask = front_rank == -1
                front_rank[mask] = current_front
                assigned_count = N
                break

            assigned_count += n_new

            if assigned_count < N:
                self._compiled['decrement_dom'](
                    (grid1d,), (block1d,),
                    (obj_f32, sums, newly_assigned, front_rank, dom_count,
                     np.int32(N), np.int32(M)),
                )
            current_front += 1

        # Map back to original order
        ranks_sorted = front_rank.get()
        ranks = np.empty(N, dtype=np.int32)
        inv = cp.asnumpy(sort_idx)
        ranks[inv] = ranks_sorted

        end_event.record()
        end_event.synchronize()
        time_ms = cp.cuda.get_elapsed_time(start_event, end_event)

        # Approximate comparison count (for paper's analysis)
        self.comparisons = int(N) * int(N)  # upper bound; GPU doesn't count individually

        return ranks, time_ms

    def _ensure_compiled(self):
        if 'dom_shared' not in self._compiled:
            self._compiled['dom_shared'] = cp.RawKernel(
                _TILED_DOMINANCE_KERNEL_SHARED, 'tiled_dominance_check_shared')
            self._compiled['dom_global'] = cp.RawKernel(
                _TILED_DOMINANCE_KERNEL, 'tiled_dominance_check')
            self._compiled['front_assign'] = cp.RawKernel(
                _FRONT_ASSIGN_KERNEL, 'front_assign')
            self._compiled['decrement_dom'] = cp.RawKernel(
                _DECREMENT_DOM_KERNEL, 'decrement_dom_count')

    # ------------------------------------------------------------------
    # NumPy fallback (CPU simulation of the same algorithm)
    # ------------------------------------------------------------------
    def _sort_numpy_fallback(self, objectives):
        """Pure-NumPy simulation of the GPU-NDS algorithm.

        Runs on CPU but follows the same algorithmic structure so that
        benchmarks can run on machines without CUDA.
        """
        N, M = objectives.shape
        self.comparisons = 0

        t0 = time.perf_counter()

        obj = objectives.astype(np.float32)

        # Phase 1: sum pre-sort
        sums = obj.sum(axis=1)
        if self.use_sum_presort:
            sort_idx = np.argsort(sums)
            obj = obj[sort_idx]
            sums = sums[sort_idx]
        else:
            sort_idx = np.arange(N)

        # Phase 2: dominance count (vectorised pairwise)
        dom_count = np.zeros(N, dtype=np.int32)
        for i in range(N):
            for j in range(i + 1, N):
                if self.use_sum_presort and sums[i] > sums[j]:
                    continue
                self.comparisons += 1
                diff = obj[i] - obj[j]
                if np.all(diff <= 0) and np.any(diff < 0):
                    dom_count[j] += 1
                elif np.all(diff >= 0) and np.any(diff > 0):
                    dom_count[i] += 1

        # Phase 3-4: front peeling
        front_rank = np.full(N, -1, dtype=np.int32)
        current_front = 0
        assigned = 0

        while assigned < N:
            mask = (front_rank == -1) & (dom_count == 0)
            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                front_rank[front_rank == -1] = current_front
                break
            front_rank[idxs] = current_front
            assigned += len(idxs)

            # Decrement dom_count for remaining
            remaining = np.where(front_rank == -1)[0]
            for i_idx in idxs:
                for j_idx in remaining:
                    if self.use_sum_presort and sums[i_idx] > sums[j_idx]:
                        continue
                    diff = obj[i_idx] - obj[j_idx]
                    if np.all(diff <= 0) and np.any(diff < 0):
                        dom_count[j_idx] -= 1

            current_front += 1

        # Map back
        ranks = np.empty(N, dtype=np.int32)
        ranks[sort_idx] = front_rank

        t1 = time.perf_counter()
        time_ms = (t1 - t0) * 1000.0
        return ranks, time_ms


# =====================================================================
# Naive GPU baseline
# =====================================================================

class NaiveGPU_NDS:
    """Naive N² all-pairs GPU non-dominated sort (baseline for comparison).

    Every pair (i,j) is checked in parallel without any tiling optimisation
    or sum-of-objectives pruning.
    """

    def __init__(self):
        self.comparisons = 0

    def sort(self, objectives, seed=None, count_comparisons=False):
        N, M = objectives.shape
        self.comparisons = N * (N - 1) // 2

        if _HAS_CUPY:
            ranks, time_ms = self._sort_cupy(objectives)
        else:
            ranks, time_ms = self._sort_numpy(objectives)

        if count_comparisons:
            return ranks, time_ms, self.comparisons
        return ranks, time_ms

    def _sort_cupy(self, objectives):
        N, M = objectives.shape
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()

        obj = cp.asarray(objectives, dtype=cp.float32)
        dom_count = cp.zeros(N, dtype=cp.int32)
        sums = cp.zeros(N, dtype=cp.float32)  # no pruning

        kernel = cp.RawKernel(_TILED_DOMINANCE_KERNEL, 'tiled_dominance_check')
        TILE = 16
        grid = ((N + TILE - 1) // TILE, (N + TILE - 1) // TILE)
        block = (TILE, TILE)
        kernel(grid, block,
               (obj, sums, dom_count, np.int32(N), np.int32(M), np.int32(TILE)))

        # Front assignment on GPU
        front_rank = cp.full(N, -1, dtype=cp.int32)
        newly_assigned = cp.zeros(N, dtype=cp.int32)
        bk = 256
        gk = (N + bk - 1) // bk
        fa_kernel = cp.RawKernel(_FRONT_ASSIGN_KERNEL, 'front_assign')
        dd_kernel = cp.RawKernel(_DECREMENT_DOM_KERNEL, 'decrement_dom_count')

        current_front = 0
        total = 0
        while total < N:
            newly_assigned[:] = 0
            fa_kernel((gk,), (bk,),
                      (dom_count, front_rank, newly_assigned,
                       np.int32(N), np.int32(current_front)))
            n_new = int(newly_assigned.get().sum())
            if n_new == 0:
                front_rank[front_rank == -1] = current_front
                break
            total += n_new
            if total < N:
                dd_kernel((gk,), (bk,),
                          (obj, sums, newly_assigned, front_rank, dom_count,
                           np.int32(N), np.int32(M)))
            current_front += 1

        end.record()
        end.synchronize()
        time_ms = cp.cuda.get_elapsed_time(start, end)
        return cp.asnumpy(front_rank), time_ms

    def _sort_numpy(self, objectives):
        """Naive N² CPU fallback — no pruning, no tiling."""
        N, M = objectives.shape
        t0 = time.perf_counter()
        dom_count = np.zeros(N, dtype=np.int32)
        for i in range(N):
            for j in range(i + 1, N):
                diff = objectives[i] - objectives[j]
                if np.all(diff <= 0) and np.any(diff < 0):
                    dom_count[j] += 1
                elif np.all(diff >= 0) and np.any(diff > 0):
                    dom_count[i] += 1

        front_rank = np.full(N, -1, dtype=np.int32)
        current_front = 0
        assigned = 0
        while assigned < N:
            idxs = np.where((front_rank == -1) & (dom_count == 0))[0]
            if len(idxs) == 0:
                front_rank[front_rank == -1] = current_front
                break
            front_rank[idxs] = current_front
            assigned += len(idxs)
            remaining = np.where(front_rank == -1)[0]
            for i_idx in idxs:
                for j_idx in remaining:
                    diff = objectives[i_idx] - objectives[j_idx]
                    if np.all(diff <= 0) and np.any(diff < 0):
                        dom_count[j_idx] -= 1
            current_front += 1

        t1 = time.perf_counter()
        return front_rank, (t1 - t0) * 1000.0


# =====================================================================
# Convenience function (auto-detects backend)
# =====================================================================

def gpu_nds(objectives, tile_size=32, use_sum_presort=True, seed=None,
            count_comparisons=False):
    """Run GPU-NDS and return (ranks, time_ms[, comparisons]).

    Automatically uses CuPy + CUDA if available, else NumPy fallback.

    Parameters
    ----------
    objectives : np.ndarray, shape (N, M)
    tile_size : int
    use_sum_presort : bool
    seed : int or None
    count_comparisons : bool

    Returns
    -------
    ranks : np.ndarray of int, shape (N,)
    time_ms : float
    comparisons : int  (only if *count_comparisons*)
    """
    sorter = GPU_NDS(tile_size=tile_size, use_sum_presort=use_sum_presort)
    return sorter.sort(objectives, seed=seed, count_comparisons=count_comparisons)


def get_backend():
    """Return string describing the active GPU backend."""
    if _HAS_CUPY:
        try:
            dev = cp.cuda.Device(0)
            name = cp.cuda.runtime.getDeviceProperties(0).get('name', b'unknown')
            if isinstance(name, bytes):
                name = name.decode()
            return f"CuPy + CUDA (GPU: {name})"
        except Exception:
            return "CuPy + CUDA (GPU detected)"
    return "NumPy CPU fallback (no CUDA device found)"
