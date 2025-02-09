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
    
    // centroids, auxCentroids, localClassMap, pointsPerClass
    float *localData = (float*)calloc(localPoints * dimPoints, sizeof(float));
    float *paddedCentroids = (float*)aligned_alloc(CACHE_LINE_SIZE, K * dimPoints * PAD_FLOAT * sizeof(float));
    float *paddedAuxCentroids = (float*)aligned_alloc(CACHE_LINE_SIZE, K * dimPoints * PAD_FLOAT * sizeof(float));
    int *paddedLocalClassMap = (int*)aligned_alloc(CACHE_LINE_SIZE, localPoints * PAD_INT * sizeof(int));
    int *paddedPointsPerClass = (int*)aligned_alloc(CACHE_LINE_SIZE, K * PAD_INT * sizeof(int));

    if (localData == NULL || paddedCentroids == NULL || paddedAuxCentroids == NULL || 
        paddedLocalClassMap == NULL || paddedPointsPerClass == NULL) {
        fprintf(stderr,"Host memory allocation error.\n");
        exit(MEMORY_ALLOCATION_ERR);
    }

    elementPaddedIntArray(paddedLocalClassMap, DEFAULT_CLASS, localPoints);
    if (rank == 0) {
        initCentroids(data, paddedCentroids, K, numPoints, dimPoints);
    }

    MPI_Bcast(paddedCentroids, K * dimPoints * PAD_FLOAT, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    int *counts = NULL, *displs = NULL;
    if (rank == 0) {
        counts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        for (int i = 0; i < size; i++) {
            int start, count;
            getLocalRange(i, size, numPoints, &start, &count);
            counts[i] = count * dimPoints;
            displs[i] = start * dimPoints;
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
        changes = 0;
        maxDist = 0.0f;

        elementPaddedIntArray(paddedPointsPerClass, 0, K);
        elementPaddedFloatArray(paddedAuxCentroids, 0.0f, K * dimPoints);

        assignDataToCentroids(localData, paddedCentroids, paddedLocalClassMap, localPoints, dimPoints, K, &changes);
        MPI_Barrier(MPI_COMM_WORLD);

        updateLocalVariables(localData, paddedAuxCentroids, paddedLocalClassMap, paddedPointsPerClass, localPoints, dimPoints, K);
        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Allreduce(MPI_IN_PLACE, paddedAuxCentroids, K * dimPoints * PAD_FLOAT, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, paddedPointsPerClass, K * PAD_INT, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        maxDist = updateCentroids(paddedCentroids, paddedAuxCentroids, paddedPointsPerClass, dimPoints, K);

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

    if (rank == 0) {
        printf("\nComputation: %f seconds", (double)(end - start) / CLOCKS_PER_SEC);
        fflush(stdout);

        sprintf(line,"\n\nComputation: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
        outputMsg = strcat(outputMsg,line);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start = clock();


    int* localClassMap = (int*)malloc(localPoints * sizeof(int));
    if (localClassMap == NULL) {
        MPI_Abort(MPI_COMM_WORLD, MEMORY_ALLOCATION_ERR);
    }
    for (int i = 0; i < localPoints; i++) {
        localClassMap[i] = paddedLocalClassMap[i * PAD_INT];
    }

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
    free(paddedCentroids);
    free(paddedLocalClassMap);
    free(paddedAuxCentroids);
    free(paddedPointsPerClass);
    free(localClassMap);

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