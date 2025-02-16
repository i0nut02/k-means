# k-means

This repository implements the K-Means clustering algorithm and explores the performance optimizations achieved through parallel computing techniques like CUDA, MPI, and OpenMP. The goal of the project is to reduce the execution time and increase scalability for large datasets by utilizing parallelism on both the GPU and multi-core CPUs.

## Technologies Used
- **CUDA**: A parallel computing platform and API for leveraging GPU acceleration.
- **MPI + OpenMP**: MPI (Message Passing Interface) is used for distributed processing across multiple nodes, while OpenMP (Open Multi-Processing) provides shared-memory parallelism within each node. 

## Results and Performance Analysis

The performance of the **CUDA**, and **hybrid MPI+OpenMP** implementations was evaluated based on execution time and speedup. The algorithm was tested in a **cluster environment using HTCondor**, ensuring scalability across multiple nodes.

### Execution Time Comparison (CUDA and Sequential)

| Model     | 100D2  | 100D2X2 | 100DX4 | 100DX8 |
|-----------|--------|---------|--------|--------|
| Sequential| 158.16 | 316.43  | 633.70 | 1268.90|
| CUDA      | 3.09   | 6.87    | 14.53  | 29.84  |
| CUDA V2   | 2.68   | 6.77    | 14.55  | 29.79  |

### Speedup Analysis for CUDA Implementations

| Model     | 100D2  | 100D2X2 | 100DX4 | 100DX8 |
|-----------|--------|---------|--------|--------|
| CUDA      | 51.06  | 46.04   | 43.59  | 42.51  |
| CUDA V2   | 59.26  | 46.84   | 43.55  | 42.59  |

### Speedup Analysis of MPI+OpenMP Implementations

| MPI Process / OpenMP Threads | 100D2  | 100D2X2 | 100DX4 | 100DX8 |
|-----------------------|--------|---------|--------|--------|
| 1_4                   | 40.1   | 81.0    | 162.06 | 331.71 |
| 2_4                   | 21.37  | 41.41   | 85.92  | 165.05 |
| 4_4                   | 10.54  | 22.12   | 43.31  | 86.97  |
| 8_4                   | 6.67   | 11.91   | 23.89  | 46.61  |
| 16_4                  | 4.22   | 6.24    | 13.91  | 26.04  |
| 4_1                   | 40.58  | 80.33   | 160.52 | 323.37 |
| 4_2                   | 20.47  | 40.58   | 81.02  | 166.57 |
| 4_4                   | 10.54  | 22.12   | 43.31  | 86.97  |
| 4_8                   | 6.83   | 11.73   | 21.13  | 47.25  |
| 4_16                  | 3.52   | 7.01    | 13.63  | 25.81  |
| 4_32                  | 2.19   | 3.96    | 7.73   | 14.51  |

