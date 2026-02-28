import cupy as cp
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.python_gpu.gpu_nsga2 import GPU_NSGA2

def profile_one_generation(N, M, n_runs=20):
    """Profile per-operator GPU time for one NSGA-II generation."""
    rng = np.random.RandomState(42)
    
    # Warm up
    pop_F = cp.array(rng.rand(N, M).astype(np.float32))
    pop_X = cp.array(rng.rand(N, 12).astype(np.float32))
    gpu_nsga2 = GPU_NSGA2(pop_size=N, n_gen=1, problem='DTLZ2', n_obj=M, n_var=12, seed=42)
    gpu_nsga2._compile()
    
    timings = {
        'nds': [], 'crowding': [], 'selection': [],
        'crossover': [], 'mutation': [], 'eval': []
    }
    
    for _ in range(n_runs):
        for op_name in timings:
            start = cp.cuda.Event(disable_timing=False)
            end   = cp.cuda.Event(disable_timing=False)
            start.record()
            
            if op_name == 'nds':
                ranks_np, _ = gpu_nsga2._nds.sort(cp.asnumpy(pop_F))
                ranks_d = cp.asarray(ranks_np.astype(np.int32))
            elif op_name == 'crowding':
                cd = gpu_nsga2._crowding_distance(pop_F, ranks_d)
            elif op_name == 'selection':
                sel = gpu_nsga2._tournament_selection(ranks_d, cd, N, rng)
            elif op_name == 'crossover':
                parents_d = pop_X[sel]
                off_X = gpu_nsga2._sbx_crossover(parents_d, rng)
            elif op_name == 'mutation':
                off_X = gpu_nsga2._polynomial_mutation(off_X, rng)
            elif op_name == 'eval':
                off_F = gpu_nsga2._evaluate(off_X)
            
            end.record()
            end.synchronize()
            timings[op_name].append(
                cp.cuda.get_elapsed_time(start, end))
    
    print(f"\nPer-operator timing at N={N}, M={M} "
          f"(avg over {n_runs} runs):")
    total = 0
    for op, times in timings.items():
        avg = np.mean(times[5:])   # skip first 5 warmup
        total += avg
        print(f"  {op:12s}: {avg:7.3f} ms")
    print(f"  {'TOTAL':12s}: {total:7.3f} ms")
    print(f"  crowding %  : "
          f"{100*np.mean(timings['crowding'][5:])/total:.1f}%")

if __name__ == '__main__':
    for N in [500, 1000, 2000, 5000]:
        profile_one_generation(N, M=3)
