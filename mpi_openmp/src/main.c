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
    char line[100];

    omp_set_num_threads(4);

    if (rank == 0) printf("[DEBUG] Process %d: Program started\n", rank);

    if(argc != 8) {
        if (rank == 0) {
            fprintf(stderr,"EXECUTION ERROR K-MEANS: Parameters are not correct.\n");
            fprintf(stderr,"./kmeans [Input Filename] [Number of clusters] [Number of iterations] [Number of changes] [Threshold] [Output data file] [Log file]\n");
            fflush(stderr);
        }
        MPI_Finalize();
        exit(INPUT_ERR);
    }

    clock_t start, end;

    MPI_Barrier(MPI_COMM_WORLD);
    start = clock();

    int numPoints = 0, dimPoints = 0;  
    char *outputMsg;
    float *data; 

    if (rank == 0) {
        printf("[DEBUG] Process %d: Allocating outputMsg\n", rank);
        outputMsg = (char *)calloc(25000, sizeof(char));
        if (outputMsg == NULL) {
            printf("[ERROR] Process %d: Failed to allocate outputMsg\n", rank);
            MPI_Abort(MPI_COMM_WORLD, MEMORY_ALLOCATION_ERR);
        }
    }

    if (rank == 0) {
        printf("[DEBUG] Process %d: Reading input file %s\n", rank, argv[1]);
        error = readInput(argv[1], &numPoints, &dimPoints);
        if(error != 0) {
            printf("[ERROR] Process %d: Failed to read input dimensions\n", rank);
            showFileError(error,argv[1]);
            MPI_Abort(MPI_COMM_WORLD, error);
        }
        printf("[DEBUG] Process %d: Read dimensions - numPoints: %d, dimPoints: %d\n", rank, numPoints, dimPoints);

        printf("[DEBUG] Process %d: Allocating data array of size %d\n", rank, numPoints*dimPoints);
        data = (float*)calloc(numPoints*dimPoints,sizeof(float));
        if (data == NULL) {
            printf("[ERROR] Process %d: Failed to allocate data array\n", rank);
            fprintf(stderr,"Memory allocation error.\n");
            MPI_Abort(MPI_COMM_WORLD, MEMORY_ALLOCATION_ERR);
        }

        error = readInput2(argv[1], data);
        if(error != 0) {
            printf("[ERROR] Process %d: Failed to read input data\n", rank);
            showFileError(error,argv[1]);
            MPI_Abort(MPI_COMM_WORLD, error);
        }
    }

    printf("[DEBUG] Process %d: Broadcasting dimensions\n", rank);
    MPI_Bcast(&numPoints, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dimPoints, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("[DEBUG] Process %d: Received dimensions - numPoints: %d, dimPoints: %d\n", rank, numPoints, dimPoints);

    int localStart, localPoints;
    getLocalRange(rank, size, numPoints, &localStart, &localPoints);
    printf("[DEBUG] Process %d: Local range - start: %d, points: %d\n", rank, localStart, localPoints);

    int K = atoi(argv[2]); 
    int maxIterations = atoi(argv[3]);
    int minChanges = (int)(numPoints*atof(argv[4])/100.0);
    float maxThreshold = atof(argv[5]);

    printf("[DEBUG] Process %d: Allocating local arrays\n", rank);
    float *localData = (float*)calloc(localPoints * dimPoints, sizeof(float));
    float *centroids = (float*)calloc(K*dimPoints, sizeof(float));
    float *auxCentroids = (float*)calloc(K*dimPoints, sizeof(float));
    int *localClassMap = (int*)malloc(numPoints * sizeof(int));
    int *pointsPerClass = (int*)calloc(K, sizeof(int));

    if (localData == NULL || centroids == NULL || localClassMap == NULL || 
        auxCentroids == NULL || pointsPerClass == NULL) {
        printf("[ERROR] Process %d: Failed to allocate local arrays\n", rank);
        fprintf(stderr,"Host memory allocation error.\n");
        exit(MEMORY_ALLOCATION_ERR);
    }

    printf("[DEBUG] Process %d: Initializing data structures\n", rank);
    elementIntArray(localClassMap, DEFAULT_CLASS, numPoints);
    if (rank == 0) {
        printf("[DEBUG] Process %d: Initializing centroids\n", rank);
        initCentroids(data, centroids, K, numPoints, dimPoints);
    }

    printf("[DEBUG] Process %d: Broadcasting centroids\n", rank);
    MPI_Bcast(centroids, K * dimPoints, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    int *counts = NULL, *displs = NULL;
    if (rank == 0) {
        printf("[DEBUG] Process %d: Setting up scatter arrays\n", rank);
        counts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        for (int i = 0; i < size; i++) {
            int start, count;
            getLocalRange(i, size, numPoints, &start, &count);
            counts[i] = count * dimPoints;
            displs[i] = start * dimPoints;
            printf("[DEBUG] Process %d: Scatter setup - rank %d, count: %d, displacement: %d\n", 
                   rank, i, counts[i], displs[i]);
        }
        free(counts);
        free(displs);
    }

    printf("[DEBUG] Process %d: Scattering data\n", rank);
    MPI_Scatterv(data, counts, displs, MPI_FLOAT,
                localData, localPoints * dimPoints, MPI_FLOAT,
                0, MPI_COMM_WORLD);
    printf("[DEBUG] Process %d: Data scattered successfully\n", rank);

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
    
    do {
        it++;
         printf("[DEBUG] Process %d: Iteration %d\n", rank, it);
        changes = 0;
        maxDist = 0.0f;

        assignDataToCentroids(localData, centroids, localClassMap, localPoints, dimPoints, K, &changes);
        printf("[DEBUG] Process %d: Assigned Data to Centroids %d\n", rank, it);
        MPI_Barrier(MPI_COMM_WORLD);

        updateLocalVariables(localData, auxCentroids, localClassMap, pointsPerClass, localPoints, dimPoints, K);
        printf("[DEBUG] Process %d: Update varaibles %d\n", rank, it);
        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Allreduce(MPI_IN_PLACE, auxCentroids, K * dimPoints, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, pointsPerClass, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        printf("[DEBUG] Process %d: AllReduce done %d\n", rank, it);

        maxDist = updateCentroids(centroids, auxCentroids, pointsPerClass, dimPoints, K);

        MPI_Allreduce(MPI_IN_PLACE, &changes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        if (rank == 0) {
            sprintf(line,"\n[%d] Cluster changes: %d\tMax. centroid distance: %f", it, changes, maxDist);
            outputMsg = strcat(outputMsg,line);
        }
    } while((changes > minChanges) && (it < maxIterations) && (maxDist > maxThreshold));
    /*
     * STOP HERE
     */

    
    #ifdef DEBUG
        // Print results and termination conditions
        printf("%s", outputMsg);
    #endif

    MPI_Barrier(MPI_COMM_WORLD);
    end = clock();

    printf("\nComputation: %f seconds", (double)(end - start) / CLOCKS_PER_SEC);
    fflush(stdout);

    if (rank == 0) {
        sprintf(line,"\n\nComputation: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
        outputMsg = strcat(outputMsg,line);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start = clock();

    if (changes <= minChanges) {
        printf("\n\nTermination condition:\nMinimum number of changes reached: %d [%d]", changes, minChanges);
    }
    else if (it >= maxIterations) {
        printf("\n\nTermination condition:\nMaximum number of iterations reached: %d [%d]", it, maxIterations);
    }
    else {
        printf("\n\nTermination condition:\nCentroid update precision reached: %g [%g]", maxDist, maxThreshold);
    }    

    if (rank == 0) {
        error = writeResult(localClassMap, numPoints, argv[6]);
        if(error != 0) {
            showFileError(error, argv[6]);
            MPI_Abort(MPI_COMM_WORLD, error);
        }
    }

    free(localData);
    free(centroids);
    free(auxCentroids);
    free(localClassMap);
    free(pointsPerClass);

    MPI_Barrier(MPI_COMM_WORLD);
    end = clock();

    printf("\nMemory deallocation: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
    fflush(stdout);

    if (rank == 0) {
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
    }
    printf("[FINISH] %d\n", rank);
    MPI_Finalize();
    return 0;
}