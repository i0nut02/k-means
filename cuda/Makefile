# Compiler settings
NVCC = nvcc
CFLAGS = -O3 -Wall
CUDA_FLAGS = -arch=sm_60

# Directories
SRC_DIR = src
BIN_DIR = bin
BUILD_DIR = build

# Source and output files
SRC = $(SRC_DIR)/main.cu
TARGET = $(BIN_DIR)/kmeans
OBJ = $(BUILD_DIR)/main.o

# Default compute capability, can be overridden on command line
COMPUTE_CAP ?= 60

# Debug flags
ifdef DEBUG
    CFLAGS += -DDEBUG -g
endif

# Create directories if they don't exist
$(shell mkdir -p $(BUILD_DIR))
$(shell mkdir -p $(BIN_DIR))

# Default target
all: $(TARGET)

# Compile CUDA source file
$(BUILD_DIR)/main.o: $(SRC)
	$(NVCC) $(CFLAGS) $(CUDA_FLAGS) -c $< -o $@

# Link the final binary
$(TARGET): $(OBJ)
	$(NVCC) $(CFLAGS) $(CUDA_FLAGS) $^ -o $@

# Clean build files
clean:
	rm -rf $(BUILD_DIR)/*.o $(TARGET)

# Clean everything including directories
distclean: clean
	rm -rf $(BUILD_DIR) $(BIN_DIR)

# Run the program with test data (adjust parameters as needed)
run: $(TARGET)
	./$(TARGET) input.txt 3 100 1.0 0.001 output.txt

.PHONY: all clean distclean run