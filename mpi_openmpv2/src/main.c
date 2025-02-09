#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>

#include <mpi.h>
#include <omp.h>

#include "../include/const.h"
#include "../include/kmeans.h"
#include "../include/file.h"

int main(int argc, char* argv[]) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    if (provided < MPI_THREAD_FUNNELED) {
        MPI_Abort(MPI_COMM_WORLD, -100);
    }

    int rank, size, error;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 9) {
        if (rank == 0) {
            fprintf(stderr, "EXECUTION ERROR K-MEANS: Parameters are not correct.\n");
            fprintf(stderr, "./kmeans [Num OMP Threads] [Input Filename] [Number of clusters] [Number of iterations] [Number of changes] [Threshold] [Output data file] [Log file]\n");
            fflush(stderr);
        }
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    // Parse the number of OpenMP threads
    int numThreads = atoi(argv[8]);
    if (numThreads <= 0) {
        if (rank == 0) {
            fprintf(stderr, "Invalid number of OpenMP threads: %d\n", numThreads);
            fflush(stderr);
        }
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    omp_set_num_threads(numThreads);

    char line[400];
    clock_t start, end;
    MPI_Barrier(MPI_COMM_WORLD);
    start = clock();

    int numPoints = 0, dimPoints = 0;
    char *outputMsg;
    float *data;

    if (rank == 0) {
        outputMsg = (char *)calloc(100*(atoi(argv[3]) + 50), sizeof(char));
        if (outputMsg == NULL) {
            MPI_Abort(MPI_COMM_WORLD, MEMORY_ALLOCATION_ERR);
        }
    }

    // Read input data in rank 0
    if (rank == 0) {
        error = readInput(argv[1], &numPoints, &dimPoints);
        if(error != 0) {
            showFileError(error,argv[1]);
            MPI_Abort(MPI_COMM_WORLD, error);
        }

        data = (float*)calloc(numPoints*dimPoints,sizeof(float));
        if (data == NULL) {
            fprintf(stderr,"Memory allocation error.\n");
            MPI_Abort(MPI_COMM_WORLD, MEMORY_ALLOCATION_ERR);
        }

        error = readInput2(argv[1], data);
        if(error != 0) {
            showFileError(error,argv[1]);
            MPI_Abort(MPI_COMM_WORLD, error);
        }
    }

    // Broadcast dimensions to all processes
    MPI_Bcast(&numPoints, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dimPoints, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate local range for MPI process
    int localStart, localPoints;
    getLocalRange(rank, size, numPoints, &localStart, &localPoints);

    // Parse parameters
    int K = atoi(argv[2]);
    int maxIterations = atoi(argv[3]);
    int minChanges = (int)(numPoints*atof(argv[4])/100.0);
    float maxThreshold = atof(argv[5]);

    // Allocate arrays for thread-local data
    float **threadLocalData = NULL;
    int **threadLocalClassMap = NULL;
    int *threadLocalPoints = NULL;
    int *threadStartIndices = NULL;

    // Calculate points per thread
    threadLocalPoints = (int*)malloc(numThreads * sizeof(int));
    threadStartIndices = (int*)malloc(numThreads * sizeof(int));
    
    for (int i = 0; i < numThreads; i++) {
        int start, count;
        getLocalRange(i, numThreads, localPoints, &start, &count);
        threadLocalPoints[i] = count;
        threadStartIndices[i] = start;
    }

    // Allocate memory for thread-local arrays
    threadLocalData = (float**)malloc(numThreads * sizeof(float*));
    threadLocalClassMap = (int**)malloc(numThreads * sizeof(int*));
    
    for (int i = 0; i < numThreads; i++) {
        threadLocalData[i] = (float*)calloc(threadLocalPoints[i] * dimPoints, sizeof(float));
        threadLocalClassMap[i] = (int*)malloc(threadLocalPoints[i] * sizeof(int));
        if (!threadLocalData[i] || !threadLocalClassMap[i]) {
            fprintf(stderr, "Thread memory allocation error.\n");
            MPI_Abort(MPI_COMM_WORLD, MEMORY_ALLOCATION_ERR);
        }
    }

    // Shared arrays
    float *centroids = (float*)calloc(K*dimPoints, sizeof(float));
    float *auxCentroids = (float*)calloc(K*dimPoints, sizeof(float));
    int *pointsPerClass = (int*)calloc(K, sizeof(int));

    if (centroids == NULL || auxCentroids == NULL || pointsPerClass == NULL) {
        fprintf(stderr,"Memory allocation error.\n");
        MPI_Abort(MPI_COMM_WORLD, MEMORY_ALLOCATION_ERR);
    }

    // Initialize centroids in rank 0 and broadcast
    if (rank == 0) {
        initCentroids(data, centroids, K, numPoints, dimPoints);
    }
    MPI_Bcast(centroids, K * dimPoints, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Scatter data to MPI processes
    int *counts = NULL, *displs = NULL;
    float *localData = (float*)calloc(localPoints * dimPoints, sizeof(float));
    
    if (rank == 0) {
        counts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        for (int i = 0; i < size; i++) {
            int _start, count;
            getLocalRange(i, size, numPoints, &_start, &count);
            counts[i] = count * dimPoints;
            displs[i] = _start * dimPoints;
        }
    }

    MPI_Scatterv(data, counts, displs, MPI_FLOAT,
                 localData, localPoints * dimPoints, MPI_FLOAT,
                 0, MPI_COMM_WORLD);

    // Distribute local data to thread-local arrays
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        memcpy(threadLocalData[tid], 
               &localData[threadStartIndices[tid] * dimPoints],
               threadLocalPoints[tid] * dimPoints * sizeof(float));
    }

    // Main clustering loop
    int it = 0;
    int changes;
    float maxDist;
    
    do {
        it++;
        changes = 0;
        maxDist = 0.0f;

        elementIntArray(pointsPerClass, 0, K);
        elementFloatArray(auxCentroids, 0.0f, K * dimPoints);

        // Process data in parallel using OpenMP
        #pragma omp parallel reduction(+:changes)
        {
            int tid = omp_get_thread_num();
            
            // Initialize thread-local class map
            elementIntArray(threadLocalClassMap[tid], DEFAULT_CLASS, threadLocalPoints[tid]);
            
            // Assign points to nearest centroids
            int threadChanges = 0;
            assignDataToCentroids(threadLocalData[tid], centroids, 
                                threadLocalClassMap[tid], threadLocalPoints[tid],
                                dimPoints, K, &threadChanges);
            changes += threadChanges;

            // Update local variables
            #pragma omp critical
            {
                updateLocalVariables(threadLocalData[tid], auxCentroids,
                                   threadLocalClassMap[tid], pointsPerClass,
                                   threadLocalPoints[tid], dimPoints, K);
            }
        }

        // Global reduction across MPI processes
        MPI_Allreduce(MPI_IN_PLACE, auxCentroids, K * dimPoints, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, pointsPerClass, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &changes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // Update centroids
        maxDist = updateCentroids(centroids, auxCentroids, pointsPerClass, dimPoints, K);
        memcpy(centroids, auxCentroids, K * dimPoints * sizeof(float));

        if (rank == 0) {
            sprintf(line, "\n[%d] Cluster changes: %d\tMax. centroid distance: %f", 
                    it, changes, maxDist);
            outputMsg = strcat(outputMsg, line);
        }
    } while ((changes > minChanges) && (it < maxIterations) && (maxDist > maxThreshold));

    // Gather results
    int* classPoints = NULL;
    if (rank == 0) {
        classPoints = (int*)malloc(numPoints * sizeof(int));
    }

    // Combine thread-local results
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        memcpy(&localData[threadStartIndices[tid]], 
               threadLocalData[tid],
               threadLocalPoints[tid] * dimPoints * sizeof(float));
    }

    // Gather results to rank 0
    int *recvcounts = NULL, *rdispls = NULL;
    if (rank == 0) {
        recvcounts = (int*)malloc(size * sizeof(int));
        rdispls = (int*)malloc(size * sizeof(int));
        for (int i = 0; i < size; i++) {
            int start, count;
            getLocalRange(i, size, numPoints, &start, &count);
            recvcounts[i] = count;
            rdispls[i] = start;
        }
    }

    // Combine thread-local class maps
    int *localClassMap = (int*)malloc(localPoints * sizeof(int));
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        memcpy(&localClassMap[threadStartIndices[tid]],
               threadLocalClassMap[tid],
               threadLocalPoints[tid] * sizeof(int));
    }

    MPI_Gatherv(localClassMap, localPoints, MPI_INT,
                classPoints, recvcounts, rdispls, MPI_INT,
                0, MPI_COMM_WORLD);

    // Write results and clean up
    if (rank == 0) {
        error = writeResult(classPoints, numPoints, argv[6]);
        if(error != 0) {
            showFileError(error, argv[6]);
            MPI_Abort(MPI_COMM_WORLD, error);
        }

        error = writeLog(argv[7], outputMsg);
        if(error != 0) {
            showFileError(error, argv[7]);
            MPI_Abort(MPI_COMM_WORLD, error);
        }
    }

    // Cleanup
    for (int i = 0; i < numThreads; i++) {
        free(threadLocalData[i]);
        free(threadLocalClassMap[i]);
    }
    free(threadLocalData);
    free(threadLocalClassMap);
    free(threadLocalPoints);
    free(threadStartIndices);
    free(localData);
    free(localClassMap);
    free(centroids);
    free(auxCentroids);
    free(pointsPerClass);

    if (rank == 0) {
        free(outputMsg);
        free(data);
        free(counts);
        free(displs);
        free(classPoints);
        free(recvcounts);
        free(rdispls);
    }

    MPI_Finalize();
    return 0;
}