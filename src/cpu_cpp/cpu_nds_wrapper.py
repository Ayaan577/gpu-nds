"""
cpu_nds_wrapper.py
Python ctypes wrapper for the C++ NDS algorithms shared library.
"""
import ctypes
import numpy as np
import os
import pathlib
import platform

_dir = pathlib.Path(__file__).parent

# Determine library name based on OS
if platform.system() == "Windows":
    _lib_name = "cpu_nds.dll"
else:
    _lib_name = "libcpu_nds.so"

_lib_path = _dir / _lib_name
if not _lib_path.exists():
    raise FileNotFoundError(
        f"C++ NDS library not found at {_lib_path}. "
        f"Run 'build.bat' (Windows) or 'make' (Linux) in {_dir}")

# On Windows, add DLL directory to search path so runtime DLLs are found
if platform.system() == "Windows":
    os.add_dll_directory(str(_dir))
    # Also add TDM-GCC bin to PATH for fallback
    _gcc_bin = r"C:\Program Files (x86)\Embarcadero\Dev-Cpp\TDM-GCC-64\bin"
    if os.path.isdir(_gcc_bin):
        os.add_dll_directory(_gcc_bin)

_lib = ctypes.CDLL(str(_lib_path))

# Type aliases
_f32p = ctypes.POINTER(ctypes.c_float)
_i32p = ctypes.POINTER(ctypes.c_int)
_i64p = ctypes.POINTER(ctypes.c_longlong)

# === Set argtypes and restypes ===
# nsga2_sort(float* obj, int N, int M, int* ranks) -> double
_lib.nsga2_sort.argtypes = [_f32p, ctypes.c_int, ctypes.c_int, _i32p]
_lib.nsga2_sort.restype = ctypes.c_double

# nsga2_omp_sort(float* obj, int N, int M, int* ranks) -> double
_lib.nsga2_omp_sort.argtypes = [_f32p, ctypes.c_int, ctypes.c_int, _i32p]
_lib.nsga2_omp_sort.restype = ctypes.c_double

# dcns_sort(float* obj, int N, int M, int* ranks,
#           long long* comparison_count, int use_presort) -> double
_lib.dcns_sort.argtypes = [_f32p, ctypes.c_int, ctypes.c_int, _i32p,
                           _i64p, ctypes.c_int]
_lib.dcns_sort.restype = ctypes.c_double

# bos_sort(float* obj, int N, int M, int* ranks) -> double
_lib.bos_sort.argtypes = [_f32p, ctypes.c_int, ctypes.c_int, _i32p]
_lib.bos_sort.restype = ctypes.c_double


def _prep(obj):
    """Prepare numpy array and output buffers for ctypes call."""
    obj = np.ascontiguousarray(obj, dtype=np.float32)
    N, M = obj.shape
    ranks = np.zeros(N, dtype=np.int32)
    fp = obj.ctypes.data_as(_f32p)
    rp = ranks.ctypes.data_as(_i32p)
    return obj, N, M, ranks, fp, rp


def nsga2_sort(obj):
    """Run C++ NSGA-II fast non-dominated sort.
    Args: obj: numpy (N, M) float32 objective matrix
    Returns: (ranks, time_ms)"""
    obj, N, M, ranks, fp, rp = _prep(obj)
    t = _lib.nsga2_sort(fp, N, M, rp)
    return ranks.copy(), float(t)


def nsga2_omp_sort(obj):
    """Run C++ OpenMP-parallel NSGA-II sort.
    Args: obj: numpy (N, M) float32 objective matrix
    Returns: (ranks, time_ms)"""
    obj, N, M, ranks, fp, rp = _prep(obj)
    t = _lib.nsga2_omp_sort(fp, N, M, rp)
    return ranks.copy(), float(t)


def dcns_sort(obj, get_comparisons=False, use_presort=True):
    """Run C++ recursive DCNS with sum-of-objectives presort.
    Args:
        obj: numpy (N, M) float32 objective matrix
        get_comparisons: if True, also return comparison count
        use_presort: if True, enable sum-of-objectives pruning
    Returns: (ranks, time_ms) or (ranks, time_ms, comparison_count)"""
    obj, N, M, ranks, fp, rp = _prep(obj)
    cmp = ctypes.c_longlong(0)
    if get_comparisons:
        cp = ctypes.byref(cmp)
    else:
        cp = ctypes.cast(ctypes.c_void_p(0), _i64p)
    t = _lib.dcns_sort(fp, N, M, rp, cp, ctypes.c_int(1 if use_presort else 0))
    if get_comparisons:
        return ranks.copy(), float(t), int(cmp.value)
    return ranks.copy(), float(t)


def bos_sort(obj):
    """Run C++ Best Order Sort.
    Args: obj: numpy (N, M) float32 objective matrix
    Returns: (ranks, time_ms)"""
    obj, N, M, ranks, fp, rp = _prep(obj)
    t = _lib.bos_sort(fp, N, M, rp)
    return ranks.copy(), float(t)
