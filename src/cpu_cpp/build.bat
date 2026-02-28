@echo off
REM Build script for C++ NDS algorithms on Windows (TDM-GCC)
REM Uses the 64-bit TDM-GCC compiler

set GCC="C:\Program Files (x86)\Embarcadero\Dev-Cpp\TDM-GCC-64\bin\g++.exe"

echo Building cpu_nds.dll with TDM-GCC 9.2.0 (64-bit)...
%GCC% -O3 -march=native -ffast-math -std=c++17 -fopenmp -shared -o cpu_nds.dll nds_algorithms.cpp

if %ERRORLEVEL% EQU 0 (
    echo BUILD SUCCESSFUL: cpu_nds.dll
    dir cpu_nds.dll
) else (
    echo BUILD FAILED with error code %ERRORLEVEL%
)
