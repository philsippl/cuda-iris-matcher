NVCC ?= nvcc
NVCCFLAGS ?= -O3 -arch=sm_89

# Iris dimensions (override with: make IR_R_DIM=8 IR_THETA_DIM=128)
ifdef IR_R_DIM
NVCCFLAGS += -DIR_R_DIM=$(IR_R_DIM)
endif
ifdef IR_THETA_DIM
NVCCFLAGS += -DIR_THETA_DIM=$(IR_THETA_DIM)
endif

# Main library object
SRCS := iris.cu
OBJS := iris.o iris_fallback.o

# Benchmark binaries
BENCHMARK := benchmark
BENCHMARK_FALLBACK := benchmark_fallback

all: $(BENCHMARK)

# Compile iris.cu to object file (tensor core version)
iris.o: iris.cu iris_params.h
	$(NVCC) $(NVCCFLAGS) -dc -o $@ iris.cu

# Compile iris.cu to object file (fallback version - forced scalar)
iris_fallback.o: iris.cu iris_params.h
	$(NVCC) $(NVCCFLAGS) -DFORCE_FALLBACK -dc -o $@ iris.cu

# Compile and link benchmark (tensor core version)
$(BENCHMARK): benchmark.cu iris.o iris_params.h
	$(NVCC) $(NVCCFLAGS) -o $@ benchmark.cu iris.o

# Compile and link benchmark (fallback version)
$(BENCHMARK_FALLBACK): benchmark.cu iris_fallback.o iris_params.h
	$(NVCC) $(NVCCFLAGS) -DFORCE_FALLBACK -o $@ benchmark.cu iris_fallback.o

run: $(BENCHMARK)
	./$(BENCHMARK)

# Quick benchmark with smaller dataset
quick: $(BENCHMARK)
	./$(BENCHMARK) 5000 3 10

# Full benchmark
bench: $(BENCHMARK)
	./$(BENCHMARK) 32768 0 1

# Build both versions for comparison
both: $(BENCHMARK) $(BENCHMARK_FALLBACK)

# Run comparison benchmark
compare: both
	@echo "===== TENSOR CORE VERSION ====="
	./$(BENCHMARK) 10000 3 10
	@echo ""
	@echo "===== FALLBACK (SCALAR) VERSION ====="
	./$(BENCHMARK_FALLBACK) 10000 3 10

clean:
	rm -f $(BENCHMARK) $(BENCHMARK_FALLBACK) $(OBJS)

# Run Python tests for both implementations
test:
	@chmod +x run_tests.sh && ./run_tests.sh

# Run Python tests for tensor core version only
test-tc:
	FORCE_FALLBACK=0 pip install -e . --no-build-isolation -q
	python -m pytest tests/python_tests/ -v

# Run Python tests for fallback version only
test-fallback:
	FORCE_FALLBACK=1 pip install -e . --no-build-isolation -q
	python -m pytest tests/python_tests/ -v

# Python benchmark targets
pybench: pybench-install
	python benchmarking/benchmark.py

pybench-install:
	pip install -e . --no-build-isolation -q

# Quick Python benchmark (small sizes)
pybench-quick: pybench-install
	python benchmarking/benchmark.py --sizes 256,512,1024 --warmup 2 --repeats 3

# Full Python benchmark (all sizes)
pybench-full: pybench-install
	python benchmarking/benchmark.py --sizes 256,512,1024,2048,4096,8192,16384 --warmup 3 --repeats 5

# Python benchmark with JSON output
pybench-json: pybench-install
	python benchmarking/benchmark.py --json

.PHONY: all run quick bench both compare clean test test-tc test-fallback pybench pybench-install pybench-quick pybench-full pybench-json

