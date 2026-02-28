@echo off
REM --- Build script for CUDA crowding distance kernel ---
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;%PATH%

cd /d "%~dp0src\gpu_kernels"

echo Compiling crowding_distance.obj...
nvcc -O3 -arch=sm_86 --use_fast_math -std=c++17 -dc -o crowding_distance.obj crowding_distance.cu
if errorlevel 1 exit /b 1

echo Compiling crowding_launcher.obj...
nvcc -O3 -arch=sm_86 --use_fast_math -std=c++17 -dc -o crowding_launcher.obj crowding_launcher.cu
if errorlevel 1 exit /b 1

echo Linking libcrowding.dll...
nvcc -O3 -arch=sm_86 --use_fast_math -std=c++17 -shared -o libcrowding.dll crowding_distance.obj crowding_launcher.obj
if errorlevel 1 exit /b 1

echo Compiling Correctness Test (static link)...
nvcc -O3 -arch=sm_86 --use_fast_math -std=c++17 -c -o test_crowding.obj ..\tests\test_crowding.cpp
if errorlevel 1 exit /b 1
nvcc -O3 -arch=sm_86 -o test_crowding.exe test_crowding.obj crowding_distance.obj crowding_launcher.obj -lcudart
if errorlevel 1 exit /b 1

echo Compiling Benchmark (static link)...
nvcc -O3 -arch=sm_86 --use_fast_math -std=c++17 -c -o bench_crowding.obj ..\benchmarks\bench_crowding.cu
if errorlevel 1 exit /b 1
nvcc -O3 -arch=sm_86 -o bench_crowding.exe bench_crowding.obj crowding_distance.obj crowding_launcher.obj -lcudart
if errorlevel 1 exit /b 1

echo ALL BUILDS SUCCESSFUL.
