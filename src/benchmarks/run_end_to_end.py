import os
import sys
import numpy as np
import cupy as cp
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.python_gpu.gpu_nsga2 import GPU_NSGA2
from src.cpu_cpp.cpu_nsga2_full_wrapper import run_cpu_nsga2

if __name__ == '__main__':
    print("\n--- Running End-to-End Experiments (Exp 8 - Custom Kernel) ---")
    
    # Custom kernel config as requested
    Ns = [100, 200, 500, 1000, 2000, 5000, 10000]
    Ms = [3, 5]
    n_runs = 5
    
    # We will save to this explicit file path
    results_file = "experiments/results/exp8_end_to_end_v2_custom_kernel.csv"
    
    # Store results for summary
    summary_data = {n: {3: "-", 5: "-"} for n in Ns}
    
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        f.write("problem,N,M,n_gen,gpu_time_ms,cpu_time_ms,speedup,gpu_faster\n")
        
        for M in Ms:
            for N in Ns:
                # 100 for N<=2000, 50 for N>2000
                n_gen = 100 if N <= 2000 else 50
                
                print(f"Testing N={N}, M={M}, n_gen={n_gen}...")
                
                # CPU runs
                cpu_best = float('inf')
                for _ in range(n_runs):
                    problem_id = 'DTLZ2_M5' if M == 5 else 'DTLZ2'
                    _, _, ms_cpu = run_cpu_nsga2(problem_id, N, M, N, n_gen, seed=42)
                    if ms_cpu < cpu_best:
                        cpu_best = ms_cpu
                        
                # GPU runs
                gpu_best = float('inf')
                gpu_nsga2 = GPU_NSGA2(pop_size=N, n_gen=n_gen, problem='DTLZ2', 
                                      n_obj=M, n_var=N, seed=42)
                for _ in range(n_runs):
                    res = gpu_nsga2.run()
                    ms_gpu = res['total_ms']
                    if ms_gpu < gpu_best:
                        gpu_best = ms_gpu
                        
                speedup = cpu_best / gpu_best
                f.write(f"DTLZ2,{N},{M},{n_gen},{gpu_best:.2f},{cpu_best:.2f},{speedup:.4f},{speedup > 1}\n")
                f.flush()
                
                # Store for summary table
                summary_data[N][M] = f"{speedup:.2f}x"
                print(f"  CPU: {cpu_best:.2f} ms | GPU: {gpu_best:.2f} ms | Speedup: {speedup:.2f}x")

    # Print summary table
    print("\nSummary Table:")
    print("  N      | M=3 speedup | M=5 speedup")
    print("  -------+-------------+------------")
    for N in Ns:
        s3 = summary_data[N][3]
        s5 = summary_data[N][5]
        print(f"  {N:<6d} | {s3:>11s} | {s5:>10s}")
