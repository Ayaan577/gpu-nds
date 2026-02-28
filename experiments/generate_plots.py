"""
Generate all key figures for the GPU-NDS paper.
Reads CSV results from experiments/results/ and produces high-quality PNGs.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Publication-quality settings
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 13
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 9

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
PLOTS_DIR = os.path.join(SCRIPT_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)


def fig1_nds_speedup_vs_N():
    """Figure 1: NDS speedup — C++ vs GPU across N values."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'cpp_vs_gpu.csv'))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Panel A: Absolute timing (log scale)
    ax = axes[0]
    M = 5  # representative
    for algo in ['CPP-NSGA2', 'CPP-DCNS', 'CPP-BOS', 'GPU-NDS']:
        sub = df[(df['M'] == M) & (df['algorithm'] == algo)]
        label = algo.replace('CPP-', 'C++ ')
        style = '--' if algo.startswith('CPP') else '-'
        marker = 'o' if algo == 'GPU-NDS' else 's'
        ax.plot(sub['N'], sub['mean_ms'], style, marker=marker, label=label, linewidth=2, markersize=6)
    
    ax.set_xlabel('Population Size N')
    ax.set_ylabel('Time (ms)')
    ax.set_yscale('log')
    ax.set_title(f'(a) NDS Runtime (M={M})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel B: Speedup vs N for different M
    ax = axes[1]
    for M in [3, 5, 8, 10]:
        sub = df[(df['M'] == M) & (df['algorithm'] == 'CPP-DCNS')]
        if len(sub) > 0:
            ax.plot(sub['N'], sub['speedup'], '-o', label=f'M={M}', linewidth=2, markersize=5)
    
    ax.set_xlabel('Population Size N')
    ax.set_ylabel('Speedup (C++ DCNS / GPU-NDS)')
    ax.set_title('(b) GPU Speedup vs C++ DCNS')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'fig1_nds_speedup.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f'Saved {path}')
    plt.close()


def fig2_scalability_M():
    """Figure 2: NDS runtime vs number of objectives M."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'exp2_scalability_M.csv'))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    for panel_idx, problem in enumerate(['DTLZ2', 'DTLZ6']):
        ax = axes[panel_idx]
        sub_prob = df[df['problem'] == problem]
        
        for algo in ['CPU-NSGA2', 'CPU-DCNS', 'CPU-BOS', 'GPU-NDS']:
            sub = sub_prob[sub_prob['algorithm'] == algo].dropna(subset=['mean_ms'])
            if len(sub) > 0:
                style = '--' if algo.startswith('CPU') else '-'
                marker = 'o' if algo == 'GPU-NDS' else 's'
                ax.plot(sub['M'], sub['mean_ms'], style, marker=marker,
                       label=algo, linewidth=2, markersize=5)
        
        ax.set_xlabel('Number of Objectives M')
        ax.set_ylabel('Time (ms)')
        ax.set_yscale('log')
        ax.set_title(f'({chr(97+panel_idx)}) {problem} (N=2000)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'fig2_scalability_M.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f'Saved {path}')
    plt.close()


def fig3_python_vs_cpp_vs_gpu():
    """Figure 3: Three-way comparison showing why C++ baselines matter."""
    df_py = pd.read_csv(os.path.join(RESULTS_DIR, 'exp3_speedup.csv'))
    df_cpp = pd.read_csv(os.path.join(RESULTS_DIR, 'cpp_vs_gpu.csv'))
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    
    M = 5
    # Python DCNS speedup
    sub_py = df_py[df_py['M'] == M]
    ax.plot(sub_py['N'], sub_py['speedup'], '--s', label='GPU vs Python DCNS',
            linewidth=2, markersize=6, color='#e74c3c')
    
    # C++ DCNS speedup
    sub_cpp = df_cpp[(df_cpp['M'] == M) & (df_cpp['algorithm'] == 'CPP-DCNS')]
    ax.plot(sub_cpp['N'], sub_cpp['speedup'], '-o', label='GPU vs C++ DCNS',
            linewidth=2, markersize=6, color='#2ecc71')
    
    # C++ BOS speedup
    sub_bos = df_cpp[(df_cpp['M'] == M) & (df_cpp['algorithm'] == 'CPP-BOS')]
    ax.plot(sub_bos['N'], sub_bos['speedup'], '-^', label='GPU vs C++ BOS',
            linewidth=2, markersize=6, color='#3498db')
    
    ax.set_xlabel('Population Size N')
    ax.set_ylabel('Speedup (CPU / GPU)')
    ax.set_title(f'GPU Speedup: Python vs C++ Baselines (M={M})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
    ax.set_yscale('log')
    
    # Add annotation
    ax.annotate('Previous paper claims\n(inflated by Python overhead)',
               xy=(5000, float(sub_py[sub_py['N']==5000]['speedup'])),
               xytext=(2000, 5000),
               arrowprops=dict(arrowstyle='->', color='#e74c3c'),
               fontsize=9, color='#e74c3c')
    
    ax.annotate('Honest speedup\nvs optimized C++',
               xy=(5000, float(sub_cpp[sub_cpp['N']==5000]['speedup'])),
               xytext=(500, 8),
               arrowprops=dict(arrowstyle='->', color='#2ecc71'),
               fontsize=9, color='#2ecc71')
    
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'fig3_python_vs_cpp.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f'Saved {path}')
    plt.close()


def fig4_end_to_end():
    """Figure 4: End-to-end GPU vs CPU NSGA-II timing (M=3 and M=5)."""
    csv_path = os.path.join(RESULTS_DIR, 'exp8_end_to_end_fair_cpp.csv')
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax_idx, M_val in enumerate([3, 5]):
        ax = axes[ax_idx]
        sub = df[df['M'] == M_val].sort_values('pop_size')
        pops = sub['pop_size'].values
        gpu_ms = sub['gpu_ms'].values
        cpu_ms = sub['cpp_full_ms'].values

        x = np.arange(len(pops))
        width = 0.35

        ax.bar(x - width/2, gpu_ms, width, label='GPU-NSGA-II', color='#e74c3c', alpha=0.8)
        ax.bar(x + width/2, cpu_ms, width, label='C++-NSGA-II', color='#3498db', alpha=0.8)

        ax.set_xlabel('Population Size')
        ax.set_ylabel('Total Time (ms)')
        ax.set_title(f'DTLZ2, M={M_val}')
        ax.set_xticks(x)
        ax.set_xticklabels([f'N={p}' for p in pops])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        for i, (g, c) in enumerate(zip(gpu_ms, cpu_ms)):
            speedup = c / g
            color = '#2ecc71' if speedup > 1 else '#e74c3c'
            txt = f'{speedup:.2f}x'
            ax.text(i, max(g, c) * 1.05, txt, ha='center', fontsize=9,
                   fontweight='bold', color=color)

    plt.suptitle('End-to-End NSGA-II: GPU vs C++ Baseline', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'fig4_end_to_end.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f'Saved {path}')
    plt.close()


def fig5_ablation():
    """Figure 5: Ablation study — TILE size and presort."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'exp5_ablation.csv'))
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Panel A: TILE size
    ax = axes[0]
    tile_data = df[df['variant'].str.startswith('TILE')]
    tiles = [int(v.split('=')[1]) for v in tile_data['variant']]
    times = tile_data['mean_ms'].values
    stds = tile_data['std_ms'].values
    ax.bar(range(len(tiles)), times, yerr=stds, color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8)
    ax.set_xticks(range(len(tiles)))
    ax.set_xticklabels([f'TILE={t}' for t in tiles])
    ax.set_ylabel('Time (ms)')
    ax.set_title('(a) TILE Size Effect')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel B: Presort
    ax = axes[1]
    presort_data = df[df['variant'].str.startswith('presort')]
    ax.bar(['Presort ON', 'Presort OFF'],
           presort_data['mean_ms'].values,
           yerr=presort_data['std_ms'].values,
           color=['#2ecc71', '#e74c3c'], alpha=0.8)
    ax.set_ylabel('Time (ms)')
    ax.set_title('(b) Sum-of-Objectives Presort')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'fig5_ablation.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f'Saved {path}')
    plt.close()


if __name__ == '__main__':
    print("Generating paper figures...")
    fig1_nds_speedup_vs_N()
    fig2_scalability_M()
    fig3_python_vs_cpp_vs_gpu()
    fig4_end_to_end()
    fig5_ablation()
    print(f"\nAll figures saved to {PLOTS_DIR}")
