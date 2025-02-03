#!/bin/bash
mpirun -np 4 mpi_openmp/bin/kmeans "$@"