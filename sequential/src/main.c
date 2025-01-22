/*
 * k-Means clustering algorithm
 *
 * Reference sequential version (Do not modify this code)
 *
 * Parallel computing (Degree in Computer Engineering)
 * 2022/2023
 *
 * Version: 1.0
 *
 * (c) 2022 Diego García-Álvarez, Arturo Gonzalez-Escribano
 * Grupo Trasgo, Universidad de Valladolid (Spain)
 *
 * This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.
 * https://creativecommons.org/licenses/by-sa/4.0/
 */
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>

#include "../include/const.h"
#include "../include/file.h"
#include "../include/kmeans.h"

//Macros
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

int main(int argc, char* argv[]) {
    //**************************************************
	/*
	* PARAMETERS
	*
	* argv[1]: Input data file
	* argv[2]: Number of clusters
	* argv[3]: Maximum number of iterations of the method. Algorithm termination condition.
	* argv[4]: Minimum percentage of class changes. Algorithm termination condition.
	*          If between one iteration and the next, the percentage of class changes is less than
	*          this percentage, the algorithm stops.
	* argv[5]: Precision in the centroid distance after the update.
	*          It is an algorithm termination condition. If between one iteration of the algorithm 
	*          and the next, the maximum distance between centroids is less than this precision, the
	*          algorithm stops.
	* argv[6]: Output file. Class assigned to each point of the input file.
	* */
    if(argc != 8) {
        fprintf(stderr,"EXECUTION ERROR K-MEANS: Parameters are not correct.\n");
        fprintf(stderr,"./KMEANS [Input Filename] [Number of clusters] [Number of iterations] [Number of changes] [Threshold] [Output data file] [Log file]\n");
        fflush(stderr);
        exit(INPUT_ERR);
    }

    // Initialize timing variables
    clock_t start, end;
    start = clock();

    // Initialize variables for clustering
    int numPoints = 0, dimPoints = 0;
    char *outputMsg = (char *)calloc(25000, sizeof(char));
    char line[100];
    
    // Read input data dimensions
    int error = readInput(argv[1], &numPoints, &dimPoints);
    if(error != 0) {
        showFileError(error,argv[1]);
        exit(error);
    }

    // Allocate and read input data
    float *data = (float*)calloc(numPoints*dimPoints,sizeof(float));
    if (data == NULL) {
        fprintf(stderr,"Memory allocation error.\n");
        exit(MEMORY_ALLOCATION_ERR);
    }

    error = readInput2(argv[1], data);
    if(error != 0) {
        showFileError(error,argv[1]);
        exit(error);
    }

    // Parse and set algorithm parameters
    int K = atoi(argv[2]); 
    int maxIterations = atoi(argv[3]);
    int minChanges = (int)(numPoints*atof(argv[4])/100.0);
    float maxThreshold = atof(argv[5]);

    // Allocate memory for clustering data structures
    float *centroids = (float*)calloc(K*dimPoints,sizeof(float));
    int *classMap = (int*)malloc(numPoints * sizeof(int));
    int *pointsPerClass = (int *)malloc(K*sizeof(int));
    float *auxCentroids = (float*)malloc(K*dimPoints*sizeof(float));
    float *distCentroids = (float*)malloc(K*sizeof(float)); 

    if (centroids == NULL || classMap == NULL || pointsPerClass == NULL || 
        auxCentroids == NULL || distCentroids == NULL) {
        fprintf(stderr,"Memory allocation error.\n");
        exit(MEMORY_ALLOCATION_ERR);
    }

    // Initialize data structures
    elementIntArray(classMap, DEFAULT_CLASS, numPoints);
    initCentroids(data, centroids, K, numPoints, dimPoints);

    #ifdef DEBUG
        // Print configuration information
        printf("\n\tData file: %s \n\tPoints: %d\n\tDimensions: %d\n", argv[1], numPoints, dimPoints);
        printf("\tNumber of clusters: %d\n", K);
        printf("\tMaximum number of iterations: %d\n", maxIterations);
        printf("\tMinimum number of changes: %d [%g%% of %d points]\n", minChanges, atof(argv[4]), numPoints);
        printf("\tMaximum centroid precision: %f\n", maxThreshold);
    #endif

    end = clock();
    printf("\nMemory allocation: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
    fflush(stdout);

    sprintf(line,"\nMemory allocation: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
    outputMsg = strcat(outputMsg,line);
    

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
        changes = assignDataToCentroids(data, centroids, classMap, K, numPoints, dimPoints);
        maxDist = updateCentroids(data, centroids, classMap, pointsPerClass, auxCentroids, K, numPoints, dimPoints);
        
        sprintf(line,"\n[%d] Cluster changes: %d\tMax. centroid distance: %f", it, changes, maxDist);
        outputMsg = strcat(outputMsg,line);
    } while((changes > minChanges) && (it < maxIterations) && (maxDist > maxThreshold));
    /*
     * STOP HERE
     */

    #ifdef DEBUG
        // Print results and termination conditions
        printf("%s", outputMsg);
    #endif
    end = clock();

    printf("\n\nComputation: %f seconds", (double)(end - start) / CLOCKS_PER_SEC);
    fflush(stdout);

    sprintf(line,"\n\nComputation: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
    outputMsg = strcat(outputMsg,line);

    start = clock();

    // Print termination condition
    if (changes <= minChanges) {
        printf("\n\nTermination condition:\nMinimum number of changes reached: %d [%d]", changes, minChanges);
    }
    else if (it >= maxIterations) {
        printf("\n\nTermination condition:\nMaximum number of iterations reached: %d [%d]", it, maxIterations);
    }
    else {
        printf("\n\nTermination condition:\nCentroid update precision reached: %g [%g]", maxDist, maxThreshold);
    }

    // Write results to output file
    error = writeResult(classMap, numPoints, argv[6]);
    if(error != 0) {
        showFileError(error, argv[6]);
        exit(error);
    }

    // Cleanup and free memory
    free(data);
    free(classMap);
    free(centroids);
    free(distCentroids);
    free(pointsPerClass);
    free(auxCentroids);

    end = clock();

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
    return 0;
}