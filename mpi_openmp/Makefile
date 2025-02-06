# Compiler and Flags
CC = mpicc
CFLAGS = -g -fopenmp -O3 -I$(INC_DIR) -mfma
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

# Debugging info (uncomment to print file lists)
# $(info SRCS: $(SRCS))
# $(info OBJS: $(OBJS))

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

# Run the program
run: $(TARGET)
	mpirun -np 4 --allow-run-as-root $(TARGET) ../test_files/input100D2.inp 10 300 0 0 ./out ./logs
