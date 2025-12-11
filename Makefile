NVCC ?= nvcc
NVCCFLAGS ?= -O3 -arch=sm_89

TARGET ?= iris
SRCS := iris.cu

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCCFLAGS) $(SRCS) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean

