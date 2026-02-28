NVCC       = nvcc
# Assuming sm_86 for Ampere RTX 3050 Ti
# Since the OS is Windows, we build a .dll instead of .so to be loaded by ctypes if necessary, 
# but the prompt specifies .so and typical build. We will use .so (or .dll depending on OS later, we'll try .so first 
# or .dll. Let's output .dll to be safe since it's Windows, but let's conform to the prompt's `libcrowding.so` just in case).
# Wait, MSVC might be needed if using nvcc on Windows.
# I will use .dll because it's Windows.
NVCCFLAGS  = -O3 -arch=sm_86 -Xcompiler -fPIC --use_fast_math -std=c++17

LIB        = libcrowding.so
OBJS       = crowding_distance.o crowding_distance_launcher.o crowding_wrapper.o

all: $(LIB)

$(LIB): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -shared -o $@ $^

crowding_distance.o: crowding_distance.cu crowding_distance.cuh
	$(NVCC) $(NVCCFLAGS) -dc -o $@ $<

crowding_distance_launcher.o: crowding_distance_launcher.cu crowding_distance.cuh
	$(NVCC) $(NVCCFLAGS) -dc -o $@ $<

crowding_wrapper.o: crowding_wrapper.cpp crowding_distance.cuh
	$(NVCC) $(NVCCFLAGS) -dc -o $@ $<

test_crowding: test_crowding_correctness.o $(LIB)
	$(NVCC) $(NVCCFLAGS) -o $@ $< -L. -lcrowding -lcuda -cudart=shared

test_crowding_correctness.o: ../tests/test_crowding_correctness.cpp
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

bench_crowding: bench_crowding.o $(LIB)
	$(NVCC) $(NVCCFLAGS) -o $@ $< -L. -lcrowding -lcuda -cudart=shared

bench_crowding.o: ../benchmarks/bench_crowding.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

clean:
	del /f /q *.o $(LIB) test_crowding bench_crowding test_crowding.exe bench_crowding.exe >nul 2>&1 || exit 0
