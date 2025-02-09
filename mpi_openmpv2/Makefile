# Compiler and Flags
CC = mpicc
CFLAGS = -g -fopenmp -fsanitize=thread -O3 -I$(INC_DIR) -mfma
LDFLAGS = -lm

# Directories
SRC_DIR = src
INC_DIR = include
OBJ_DIR = obj
BIN_DIR = bin

# Source and Object Files
SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)
TARGET = $(BIN_DIR)/kmeans

# Default target is $(TARGET)
all: $(TARGET)

# Create necessary directories before compiling
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Compilation rule for .c to .o
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Linking the final executable
$(TARGET): $(BIN_DIR) $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(TARGET) $(LDFLAGS)

# Clean build
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
