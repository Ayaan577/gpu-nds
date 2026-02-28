# GPU-Native Divide-and-Conquer Non-Dominated Sorting (GPU-NDS)

A complete research project implementing GPU-accelerated non-dominated sorting
for Multi-Objective Evolutionary Algorithms (MOEAs), targeting conference
publication (CEC/GECCO/EMO).

GPU-NDS implements a four-phase CUDA pipeline: sum-of-objectives pre-sort,
tiled shared-memory dominance checks, parallel front assignment via atomic
operations, and prefix-scan cleanup. It achieves significant speedups over
CPU baselines (NSGA-II, DCNS, BOS) for large populations and many-objective
problems, while producing provably correct Pareto front assignments.

## Installation

```bash
pip install -r requirements.txt
```

For GPU acceleration (recommended — requires an NVIDIA GPU with CUDA support):
```bash
pip install cupy-cuda11x
```

## Quick Start

### 1. Verify Correctness
```bash
cd gpu_nds
python -m src.benchmarks.correctness_check
```

### 2. Run Smoke Test (fast, small problem sizes)
```bash
python -m src.benchmarks.run_benchmarks --smoke
```

### 3. Run All Experiments (full benchmark suite)
```bash
python -m src.benchmarks.run_benchmarks --all
```

### 4. Generate Plots
```bash
python experiments/generate_plots.py
```

### 5. Run Performance Model Validation
```bash
python -m src.analysis.performance_model
```

## Reproducing All Paper Results

Single command to run everything:
```bash
python -m src.benchmarks.run_benchmarks --all && python experiments/generate_plots.py && python -m src.analysis.performance_model
```

## CUDA C Compilation (optional, requires nvcc)

```bash
cd src/gpu
nvcc -O3 -arch=sm_86 gpu_nds.cu -o gpu_nds
```

## Project Structure

```
gpu_nds/
├── src/
│   ├── cpu/
│   │   ├── nsga2_sort.py        # CPU baseline: Deb 2002 O(MN²)
│   │   ├── dcns.py              # CPU baseline: DCNS (Mishra 2019)
│   │   └── bos.py               # CPU baseline: Best Order Sort
│   ├── gpu/
│   │   ├── gpu_nds.cu           # CUDA C reference implementation
│   │   ├── gpu_nds_kernels.cuh  # CUDA kernel headers
│   │   ├── naive_gpu.cu         # Naive GPU baseline
│   │   └── utils.cuh            # CUDA utilities
│   ├── python_gpu/
│   │   └── gpu_nds_cupy.py      # CuPy GPU implementation (primary)
│   ├── benchmarks/
│   │   ├── run_benchmarks.py    # Master benchmark runner
│   │   ├── generate_problems.py # DTLZ/WFG problem generators
│   │   └── correctness_check.py # Correctness verification
│   └── analysis/
│       └── performance_model.py # Analytical performance model
├── experiments/
│   ├── results/                 # CSV experiment results
│   ├── plots/                   # Publication-quality figures
│   └── generate_plots.py       # Plot generation script
├── paper/
│   ├── main.tex                 # Full LaTeX paper (IEEEtran)
│   ├── sections/                # LaTeX sections
│   ├── figures/                 # Paper figures
│   └── references.bib          # BibTeX references
├── requirements.txt
├── setup.py
└── README.md
```

## Authors

- Ayaan
- Prof. Sumit Mishra

## License

MIT
