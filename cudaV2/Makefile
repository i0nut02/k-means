# Compiler settings
NVCC = nvcc
CFLAGS = -O3
LIBS = -lm

# Directories
SRC_DIR = src
BIN_DIR = bin
BUILD_DIR = build

# Source and output files
SRC = $(SRC_DIR)/main.cu
TARGET = $(BIN_DIR)/kmeans
OBJ = $(BUILD_DIR)/main.o

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
	$(NVCC) $(CFLAGS) -c $< -o $@

# Link the final binary
$(TARGET): $(OBJ)
	$(NVCC) $(CFLAGS) $(OBJ) -o $@ $(LIBS)

# Clean build files
clean:
	rm -rf $(BUILD_DIR)/*.o $(TARGET)

.PHONY: all clean
