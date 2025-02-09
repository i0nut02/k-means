#
# K-means 
#
# Parallel computing (Degree in Computer Engineering)
# 2022/2023
#
# (c) 2023 Diego Garcia-Alvarez and Arturo Gonzalez-Escribano
# Grupo Trasgo, Universidad de Valladolid (Spain)
#

# Compilers
CC=gcc
OMPFLAG=-fopenmp
MPICC=mpicc
CUDACC=nvcc

# Flags for optimization and libs
FLAGS=-O3 -Wall
LIBS=-lm

# Default target when no argument is given
all: kmeans

# Build the kmeans executable (OpenMP and MPI)
kmeans: main.c
	$(MPICC) $(FLAGS) $(DEBUG) $(OMPFLAG) $< $(LIBS) -o kmeans

# Remove the target files
clean:
	rm -rf kmeans

# Compile in debug mode
debug:
	make DEBUG="-DDEBUG -g" FLAGS= all
