# Variables
CC = clang
CXX = clang++
CFLAGS = -std=c99 -Wall -O3 -mfma
CXXFLAGS = -std=c++14 -Wall -O3 -mfma
BUILD_DIR = build
SRC_DIR = src
INCLUDE_DIR = include
BIN_DIR = bin
EXECUTABLE = kmeans

# Collect source files
C_SRCS := $(wildcard $(SRC_DIR)/*.c)
HEADERS := $(wildcard $(INCLUDE_DIR)/*.h)
OBJECTS := $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(C_SRCS))

# Default target
all: $(BIN_DIR)/$(EXECUTABLE)

# Create output directories
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Build object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c $(HEADERS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Link the final executable
$(BIN_DIR)/$(EXECUTABLE): $(OBJECTS) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(OBJECTS) -lm -o $@

# Debug target
debug: CFLAGS += -g
debug: all
	@echo "Built in Debug mode."

# Clean target
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

# Help target
help:
	@echo "Makefile for KMeans Project"
	@echo "Targets:"
	@echo "  all       - Build the project (default target)"
	@echo "  debug     - Build the project with debug information"
	@echo "  clean     - Remove all build files"
	@echo "  help      - Display this help message"

.PHONY: all clean debug help
