#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>

#include <mpi.h>

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

    if (argc != 9) {  // Now expecting 8 arguments + program name
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

    // Set OpenMP threads
    omp_set_num_threads(numThreads);

    char line[400];
    clock_t start, end;

    MPI_Barrier(MPI_COMM_WORLD);
    start = clock();

    int numPoints = 0, dimPoints = 0;  
    char *outputMsg;
    float *data; 

    if (rank == 0) {
        outputMsg = (char *)calloc(25000, sizeof(char));
        if (outputMsg == NULL) {
            MPI_Abort(MPI_COMM_WORLD, MEMORY_ALLOCATION_ERR);
        }
    }

    if (rank == 0) {
        error = readInput(argv[1], &numPoints, &dimPoints);
        if(error != 0) {
            showFileError(error,argv[1]);
            MPI_Abort(MPI_COMM_WORLD, error);
        }

        data = (float*)calloc(numPoints*dimPoints,sizeof(float));
        if (data == NULL) {
            printf("[ERROR] Process %d: Failed to allocate data array\n", rank);
            fprintf(stderr,"Memory allocation error.\n");
            MPI_Abort(MPI_COMM_WORLD, MEMORY_ALLOCATION_ERR);
        }

        error = readInput2(argv[1], data);
        if(error != 0) {
            showFileError(error,argv[1]);
            MPI_Abort(MPI_COMM_WORLD, error);
        }
    }

    MPI_Bcast(&numPoints, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dimPoints, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int localStart, localPoints;
    getLocalRange(rank, size, numPoints, &localStart, &localPoints);

    int K = atoi(argv[2]); 
    int maxIterations = atoi(argv[3]);
    int minChanges = (int)(numPoints*atof(argv[4])/100.0);
    float maxThreshold = atof(argv[5]);

    float *localData = (float*)calloc(localPoints * dimPoints, sizeof(float));
    float *centroids = (float*)calloc(K*dimPoints, sizeof(float));
    float *auxCentroids = (float*)calloc(K*dimPoints, sizeof(float));
    int *localClassMap = (int*)malloc(localPoints * sizeof(int));
    int *pointsPerClass = (int*)calloc(K, sizeof(int));

    if (localData == NULL || centroids == NULL || localClassMap == NULL || 
        auxCentroids == NULL || pointsPerClass == NULL) {
        fprintf(stderr,"Host memory allocation error.\n");
        exit(MEMORY_ALLOCATION_ERR);
    }

    elementIntArray(localClassMap, DEFAULT_CLASS, localPoints);
    if (rank == 0) {
        initCentroids(data, centroids, K, numPoints, dimPoints);
    }

    MPI_Bcast(centroids, K * dimPoints, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    int *counts = NULL, *displs = NULL;
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

    #ifdef DEBUG
        // Print configuration information
        printf("\n\tData file: %s \n\tPoints: %d\n\tDimensions: %d\n", argv[1], numPoints, dimPoints);
        printf("\tNumber of clusters: %d\n", K);
        printf("\tMaximum number of iterations: %d\n", maxIterations);
        printf("\tMinimum number of changes: %d [%g%% of %d points]\n", minChanges, atof(argv[4]), numPoints);
        printf("\tMaximum centroid precision: %f\n", maxThreshold);
    #endif

    MPI_Barrier(MPI_COMM_WORLD);
    end = clock();
    if (rank == 0) {
        printf("\nMemory allocation: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
        fflush(stdout);
        sprintf(line,"\nMemory allocation: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
        outputMsg = strcat(outputMsg,line);
        sprintf(line,"\nNumber of threads = %d\n", numThreads);
        outputMsg = strcat(outputMsg,line);

    }

    MPI_Barrier(MPI_COMM_WORLD);
    // Start clustering computation
    start = clock();

    /*
     * START Part to pararellilize
     */
    int it = 0;
    int changes;
    float maxDist;

    double totalElementTime = 0.0;
    double totalAssignTime = 0.0;
    double totalUpdateLocalTime = 0.0;
    double totalUpdateCentroidsTime = 0.0;

    do {
        it++;
        changes = 0;
        maxDist = 0.0f;

        clock_t lStart = clock();
        elementIntArray(pointsPerClass, 0, K);
        elementFloatArray(auxCentroids, 0.0f, K * dimPoints);
        clock_t lEnd = clock();

        double elementTime = (double)(lEnd - lStart) / CLOCKS_PER_SEC;  // Convert to seconds
        totalElementTime += elementTime;
        printf("[%d] time for element = %f seconds\n", it, elementTime);

        lStart = clock();
        assignDataToCentroids(localData, centroids, localClassMap, localPoints, dimPoints, K, &changes);
        MPI_Barrier(MPI_COMM_WORLD);
        lEnd = clock();

        double assignTime = (double)(lEnd - lStart) / CLOCKS_PER_SEC;  // Convert to seconds
        totalAssignTime += assignTime;
        printf("[%d] time for assignDataToCentroids = %f seconds\n", it, assignTime);

        lStart = clock();
        updateLocalVariables(localData, auxCentroids, localClassMap, pointsPerClass, localPoints, dimPoints, K);
        MPI_Barrier(MPI_COMM_WORLD);
        lEnd = clock();
        double updateLocalTime = (double)(lEnd - lStart) / CLOCKS_PER_SEC;  // Convert to seconds
        totalUpdateLocalTime += updateLocalTime;
        printf("[%d] time for updateLocalVariables = %f seconds\n", it, updateLocalTime);

        MPI_Allreduce(MPI_IN_PLACE, auxCentroids, K * dimPoints, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, pointsPerClass, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        int s = 0;
        for (int i = 0; i < K; i++) {
            s += pointsPerClass[i];
        }

        lStart = clock();
        maxDist = updateCentroids(centroids, auxCentroids, pointsPerClass, dimPoints, K);
        lEnd = clock();
        double updateCentroidsTime = (double)(lEnd - lStart) / CLOCKS_PER_SEC;  // Convert to seconds
        totalUpdateCentroidsTime += updateCentroidsTime;
        printf("[%d] time for updateCentroids = %f seconds\n", it, updateCentroidsTime);

        MPI_Allreduce(MPI_IN_PLACE, &changes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        memcpy(centroids, auxCentroids, (K * dimPoints * sizeof(float)));

        if (rank == 0) {
            sprintf(line, "\n[%d] Cluster changes: %d\tMax. centroid distance: %f", it, changes, maxDist);
            outputMsg = strcat(outputMsg, line);
        }
    } while ((changes > minChanges) && (it < maxIterations) && (maxDist > maxThreshold));

    /*
    * STOP HERE
    */
    if (rank == 0) {
        char tempLine[1000];
        sprintf(tempLine, 
            "\n\nTiming Summary (total across %d iterations):"
            "\nElement Arrays time: %f seconds"
            "\nAssign Data to Centroids time: %f seconds"
            "\nUpdate Local Variables time: %f seconds"
            "\nUpdate Centroids time: %f seconds"
            "\nTotal time per function:"
            "\nElement Arrays avg: %f seconds"
            "\nAssign Data to Centroids avg: %f seconds"
            "\nUpdate Local Variables avg: %f seconds"
            "\nUpdate Centroids avg: %f seconds",
            it,
            totalElementTime,
            totalAssignTime,
            totalUpdateLocalTime,
            totalUpdateCentroidsTime,
            totalElementTime / it,
            totalAssignTime / it,
            totalUpdateLocalTime / it,
            totalUpdateCentroidsTime / it
        );
        outputMsg = strcat(outputMsg, tempLine);
    }
    #ifdef DEBUG
        // Print results and termination conditions
        printf("%s", outputMsg);
    #endif

    MPI_Barrier(MPI_COMM_WORLD);
    end = clock();

    if (rank == 0) {
        printf("\nComputation: %f seconds", (double)(end - start) / CLOCKS_PER_SEC);
        fflush(stdout);

        sprintf(line,"\n\nComputation: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
        outputMsg = strcat(outputMsg,line);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start = clock();

    int* classPoints = NULL;
    int *recvcounts = NULL;
    int *rdispls = NULL;

    if (rank == 0) {
        classPoints = (int*)malloc(numPoints*sizeof(int));
        recvcounts = (int*)malloc(size * sizeof(int));
        rdispls = (int*)malloc(size * sizeof(int));

        if (classPoints == NULL || recvcounts == NULL || rdispls == NULL) {
            MPI_Abort(MPI_COMM_WORLD, MEMORY_ALLOCATION_ERR);
        }

        for (int i = 0; i < size; i++) {
            int start, count;
            getLocalRange(i, size, numPoints, &start, &count);
            recvcounts[i] = count;
            rdispls[i] = start;
        }
    }

    MPI_Gatherv(localClassMap, localPoints, MPI_INT, classPoints, recvcounts, rdispls, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (changes <= minChanges) {
            printf("\n\nTermination condition:\nMinimum number of changes reached: %d [%d]", changes, minChanges);
        }
        else if (it >= maxIterations) {
            printf("\n\nTermination condition:\nMaximum number of iterations reached: %d [%d]", it, maxIterations);
        }
        else {
            printf("\n\nTermination condition:\nCentroid update precision reached: %g [%g]", maxDist, maxThreshold);
        } 

        error = writeResult(classPoints, numPoints, argv[6]);
        if(error != 0) {
            showFileError(error, argv[6]);
            MPI_Abort(MPI_COMM_WORLD, error);
        }
    }

    free(localData);
    free(centroids);
    free(localClassMap);
    free(auxCentroids);
    free(pointsPerClass);

    MPI_Barrier(MPI_COMM_WORLD);
    end = clock();

    if (rank == 0) {
        printf("\nMemory deallocation: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
        fflush(stdout);

        sprintf(line,"\nMemory deallocation: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
        outputMsg = strcat(outputMsg,line);
    
        // Write to log file
        error = writeLog(argv[7], outputMsg);
        if(error != 0) {
            showFileError(error, argv[7]);
            exit(error);
        }
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