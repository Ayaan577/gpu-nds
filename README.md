# GPU-NDS: GPU-Accelerated Non-Dominated Sorting

> **Reproducibility package** for:  
> *GPU-NDS: GPU-Accelerated Non-Dominated Sorting with DCNS-Inspired Tiled Shared-Memory Dominance Checks*  
> Khan, Arif, Nayak, Mishra — NIT Warangal

GPU-NDS achieves up to **27.8× speedup** over optimized C++ baselines (compiled with `-O3`) for non-dominated sorting, and **2.09× end-to-end speedup** for a fully GPU-resident NSGA-II at N=2,000.

---

## Repository Structure

```
├── src/
│   ├── python_gpu/          # GPU-NSGA-II (CuPy + CUDA kernels)
│   │   ├── gpu_nds_cupy.py      # GPU-NDS sorting via CuPy
│   │   └── gpu_nsga2.py         # Full GPU-resident NSGA-II
│   ├── gpu/                 # Raw CUDA kernels (.cu/.cuh)
│   ├── gpu_kernels/         # Standalone CUDA crowding distance kernel
│   │   ├── crowding_distance.cu
│   │   ├── crowding_distance.cuh
│   │   ├── crowding_launcher.cu
│   │   └── Makefile
│   ├── cpu/                 # Python CPU baselines (NSGA-II, DCNS, BOS)
│   ├── cpu_cpp/             # C++ baselines compiled with -O3
│   │   ├── nds_algorithms.cpp   # C++ NDS: NSGA-II sort, DCNS, BOS
│   │   └── cpu_nsga2_full.cpp   # Full C++ NSGA-II (end-to-end baseline)
│   ├── benchmarks/          # Benchmark scripts
│   ├── analysis/            # Performance model validation
│   └── tests/               # Correctness tests
├── experiments/
│   ├── results/             # Raw CSV benchmark data
│   └── generate_plots.py    # Regenerate all figures
├── requirements.txt
├── setup.py
└── build_cuda.bat           # Build CUDA crowding kernel (Windows)
```

## Prerequisites

| Dependency | Version Tested |
|-----------|---------------|
| Python | 3.10+ |
| CUDA Toolkit | 13.1 |
| CuPy | `cupy-cuda12x` |
| NVIDIA GPU | RTX 3050 Ti (Compute 8.6) |
| C++ Compiler | GCC 9.2+ or MSVC 2022 |

## Reproducing Results

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Build C++ baselines

```bash
# Linux/macOS
cd src/cpu_cpp
g++ -O3 -ffast-math -std=c++17 -shared -fPIC -o cpu_nds.so nds_algorithms.cpp
g++ -O3 -ffast-math -std=c++17 -shared -fPIC -o cpu_nsga2_full.so cpu_nsga2_full.cpp

# Windows
cd src/cpu_cpp
build.bat
```

### 3. Build CUDA crowding distance kernel

```bash
# Windows
build_cuda.bat

# Linux (from src/gpu_kernels/)
make all
```

### 4. Run NDS benchmarks

```bash
python -m src.benchmarks.run_benchmarks
python -m src.benchmarks.run_cpp_benchmark
```

### 5. Run end-to-end NSGA-II comparison

```bash
python -m src.benchmarks.run_end_to_end
```

### 6. Run Python baseline timing

```bash
python -m src.benchmarks.run_python_baseline
```

### 7. Regenerate figures from CSV data

```bash
python experiments/generate_plots.py
```

## Key Results

| Comparison | Peak Speedup | Configuration |
|-----------|-------------|---------------|
| GPU-NDS vs C++ DCNS | **27.8×** | N=10,000, M=8 |
| GPU-NDS vs C++ NSGA-II sort | **22.3×** | N=5,000, M=10 |
| GPU-NDS vs C++ BOS | **7.9×** | N=10,000, M=10 |
| GPU-NSGA-II vs C++-NSGA-II (end-to-end) | **2.09×** | N=2,000, M=3 |
| CUDA crowding vs CuPy crowding | **3×** | N=2,000, M=3 |

## Correctness Validation

GPU-NDS produces **provably identical** front assignments to sequential non-dominated sorting. Validated across 480 configurations:

```bash
python -m src.benchmarks.correctness_check
```

## Citation

```bibtex
@inproceedings{khan2025gpunds,
  title     = {{GPU-NDS}: {GPU}-Accelerated Non-Dominated Sorting
               with {DCNS}-Inspired Tiled Shared-Memory Dominance Checks},
  author    = {Khan, Mohammed Azeez and Arif, Mohammed and
               Nayak, Supreet and Mishra, Sumit},
  booktitle = {Lecture Notes in Computer Science},
  year      = {2025},
  publisher = {Springer}
}
```

## License

This repository is provided for academic reproducibility purposes.
