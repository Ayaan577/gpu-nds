"""
Python wrapper for the full C++ NSGA-II implementation.
All operators run in compiled C++ — no Python in the loop.
"""
import ctypes
import numpy as np
import os
import pathlib
import platform
import sys

# Load shared library
_dir = pathlib.Path(__file__).parent
if platform.system() == 'Windows':
    _lib_name = 'cpu_nsga2_full.dll'
    # Add DLL directory for Windows
    dll_path = str(_dir)
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(dll_path)
    if dll_path not in os.environ.get('PATH', ''):
        os.environ['PATH'] = dll_path + os.pathsep + os.environ.get('PATH', '')
else:
    _lib_name = 'libcpu_nsga2_full.so'

_lib_path = _dir / _lib_name
_lib = ctypes.CDLL(str(_lib_path))

# Set up function signature
_lib.cpu_nsga2_run.restype = ctypes.c_double
_lib.cpu_nsga2_run.argtypes = [
    ctypes.c_int,                           # N (pop_size)
    ctypes.c_int,                           # n_var
    ctypes.c_int,                           # n_obj
    ctypes.c_int,                           # n_gen
    ctypes.POINTER(ctypes.c_float),         # xl
    ctypes.POINTER(ctypes.c_float),         # xu
    ctypes.c_int,                           # seed
    ctypes.POINTER(ctypes.c_float),         # F_out
    ctypes.POINTER(ctypes.c_float),         # X_out
    ctypes.c_int,                           # problem_id
]

PROBLEM_IDS = {
    'ZDT1': 0,
    'ZDT2': 1,
    'ZDT3': 2,
    'DTLZ2_M3': 3,
    'DTLZ2_M5': 4,
    'DTLZ2': 3,  # default DTLZ2 = 3 objectives
}


def run_cpu_nsga2(problem_name, n_var, n_obj, pop_size, n_gen, seed=42):
    """
    Run the full C++ NSGA-II.

    Returns:
        X_out: (pop_size, n_var) final decision variables
        F_out: (pop_size, n_obj) final objective values
        t_ms:  elapsed time in milliseconds
    """
    xl = np.zeros(n_var, dtype=np.float32)
    xu = np.ones(n_var, dtype=np.float32)
    F_out = np.zeros((pop_size, n_obj), dtype=np.float32)
    X_out = np.zeros((pop_size, n_var), dtype=np.float32)

    pid = PROBLEM_IDS.get(problem_name, 3)

    t_ms = _lib.cpu_nsga2_run(
        pop_size, n_var, n_obj, n_gen,
        xl.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        xu.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        seed,
        F_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        X_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        pid,
    )

    return X_out, F_out, float(t_ms)
