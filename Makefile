NVCC ?= nvcc
NVCCFLAGS ?= -O3 -arch=sm_89

# Main library object
SRCS := iris.cu
OBJS := iris.o

# Benchmark binary
BENCHMARK := benchmark

all: $(BENCHMARK)

# Compile iris.cu to object file (reusable)
iris.o: iris.cu iris_params.h
	$(NVCC) $(NVCCFLAGS) -dc -o $@ iris.cu

# Compile and link benchmark
$(BENCHMARK): benchmark.cu iris.o iris_params.h
	$(NVCC) $(NVCCFLAGS) -o $@ benchmark.cu iris.o

run: $(BENCHMARK)
	./$(BENCHMARK)

# Quick benchmark with smaller dataset
quick: $(BENCHMARK)
	./$(BENCHMARK) 5000 3 10

# Full benchmark
bench: $(BENCHMARK)
	./$(BENCHMARK) 10000 5 20

clean:
	rm -f $(BENCHMARK) $(OBJS)

.PHONY: all run quick bench clean

