#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>

#include <mpi.h>
#include <omp.h>

// Constants that were in const.h
#define MAXLINE 200000
#define DEFAULT_CLASS -1
#define SEED 0

// Error codes
#define INPUT_ERR -1
#define MEMORY_ALLOCATION_ERR -2
#define TOO_MUCH_COLUMNS_ERR -3
#define READ_ERR -4
#define WRITE_ERR -5

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))


void elementIntArray(int *array, int value, int size) {
    #pragma omp parallel for
    for(int i = 0; i < size; i++) {
        array[i] = value;
    }
}

void initCentroids(const float* data, float* centroids, const int K, const int n, const int dim) {
    int* dataPointAssigned = (int*) calloc(n, sizeof(int));

    if (dataPointAssigned == NULL) {
        exit(MEMORY_ALLOCATION_ERR);
    }

    srand(SEED);
    for (int i = 0; i < K; i++) {
        int idx = rand() % n;
        while (dataPointAssigned[idx] != 0) {
            idx = (idx + 1) % n;
        }
        memcpy(&centroids[i*dim], &data[idx*dim], dim * sizeof(float));
        dataPointAssigned[idx] = -1;
    }
    
    free(dataPointAssigned);
    return;
}

void showFileError(int error, char* filename) {
    printf("Error\n");
    switch (error) {
        case TOO_MUCH_COLUMNS_ERR:
            fprintf(stderr,"\tFile %s has too many columns.\n", filename);
            fprintf(stderr,"\tThe maximum number of columns has been exceeded. MAXLINE: %d.\n", MAXLINE);
            break;
        case READ_ERR:
            fprintf(stderr,"Error reading file: %s.\n", filename);
            break;
        case WRITE_ERR:
            fprintf(stderr,"Error writing file: %s.\n", filename);
            break;
    }
    fflush(stderr);    
}

int readInput(char* filename, int *lines, int *samples) {
    FILE *fp;
    char line[MAXLINE] = "";
    char *ptr;
    const char *delim = "\t";
    int contlines, contsamples = 0;
    
    contlines = 0;

    if ((fp=fopen(filename,"r"))!=NULL) {
        while(fgets(line, MAXLINE, fp)!= NULL) {
            if (strchr(line, '\n') == NULL) {
                return -1;
            }
            contlines++;       
            ptr = strtok(line, delim);
            contsamples = 0;
            while(ptr != NULL) {
                contsamples++;
                ptr = strtok(NULL, delim);
            }        
        }
        fclose(fp);
        *lines = contlines;
        *samples = contsamples;  
        return 0;
    }
    else {
        return READ_ERR;
    }
}

int readInput2(char* filename, float* data) {
    FILE *fp;
    char line[MAXLINE] = "";
    char *ptr;
    const char *delim = "\t";
    int i = 0;
    
    if ((fp=fopen(filename,"rt"))!=NULL) {
        while(fgets(line, MAXLINE, fp)!= NULL) {         
            ptr = strtok(line, delim);
            while(ptr != NULL) {
                data[i] = atof(ptr);
                i++;
                ptr = strtok(NULL, delim);
            }
        }
        fclose(fp);
        return 0;
    }
    else {
        return READ_ERR;
    }
}

int writeResult(int *classMap, int lines, const char* filename) {    
    FILE *fp;
    
    if ((fp=fopen(filename,"wt"))!=NULL) {
        for(int i=0; i<lines; i++) {
            fprintf(fp,"%d\n",classMap[i]);
        }
        fclose(fp);  
   
        return 0;
    }
    else {
        return WRITE_ERR;
    }
}

int writeLog(const char* filename, const char* message) {    
    FILE *fp;
    
    if ((fp=fopen(filename,"wt"))!=NULL) {
        fprintf(fp, "%s", message);
        fclose(fp);   
        return 0;
    }
    else {
        return WRITE_ERR;
    }
}

// OpenMP
void assignDataToCentroids(const float *data, const float *centroids, int *classMap, int numPoints, int dimPoints, int K, int *changes) {
    int localChanges = 0;
    #pragma omp parallel for reduction(+:localChanges)
    for (int i = 0; i < numPoints; i++) {
        float minDist = FLT_MAX;
        int newClass = classMap[i];

        for (int k = 0; k < K; k++) {
            float dist = 0.0f;
            for (int d = 0; d < dimPoints; d++) {
                float diff = data[i * dimPoints + d] - centroids[k * dimPoints + d];
                dist += diff * diff;
            }
            if (dist < minDist) {
                minDist = dist;
                newClass = k;
            }
        }
        if (classMap[i] != newClass) {
            classMap[i] = newClass;
            localChanges++;
        }
    }
    *changes = localChanges;
}

void updateLocalVariables(const float *data, float *auxCentroids, const int *classMap, int *pointsPerClass, int numPoints, int dimPoints, int K) {
    #pragma omp parallel for reduction(+:pointsPerClass[:K], auxCentroids[:K * dimPoints])
        for (int i = 0; i < numPoints; i++) {
            int class_id = classMap[i];
            pointsPerClass[class_id]++;

            for (int d = 0; d < dimPoints; d++) {
                auxCentroids[class_id * dimPoints + d] += data[i * dimPoints + d];
            }
        }
}

void updateCentroids(float *centroids, const float *auxCentroids, int *pointsPerClass, int dimPoints, int K) {
    #pragma omp parallel for
    for (int k = 0; k < K; k++) {
        int kPoints = pointsPerClass[k];
        if (kPoints > 0) {
            for (int d = 0; d < dimPoints; d++) {
                centroids[k * dimPoints + d] = auxCentroids[k * dimPoints + d] / kPoints;
            }
        }
    }
}

// Get local data range for each MPI process
void getLocalRange(int rank, int size, int totalPoints, int *start, int *count) {
    int quotient = totalPoints / size;
    int remainder = totalPoints % size;
    *start = quotient * rank + (rank < remainder ? rank : remainder);
    *count = quotient + (rank < remainder ? 1 : 0);
}


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

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(argc != 8) {
        if (rank == 0) {
            fprintf(stderr,"EXECUTION ERROR K-MEANS: Parameters are not correct.\n");
            fprintf(stderr,"./kmeans [Input Filename] [Number of clusters] [Number of iterations] [Number of changes] [Threshold] [Output data file] [Log file]\n");
            fflush(stderr);
        }
        MPI_Finalize();
        exit(INPUT_ERR);
    }

    // Initialize timing variables
    clock_t start, end;
    start = clock();

    // Initialize variables for clustering
    int numPoints = 0, dimPoints = 0;  
    char *outputMsg;
    float *data; 

    if (rank == 0) {
        outputMsg = (char *)calloc(25000, sizeof(char));
        if (outputMsg == NULL) {
            MPI_Abort(MPI_COMM_WORLD, MEMORY_ALLOCATION_ERR);
        }
    }

    char line[100];
    
    int error;
    
    // all staff regarding the reading of the file are done by rank 0 
    if (rank == 0) {
        // Read input data dimensions 
        error = readInput(argv[1], &allPoints, &dimPoints);
        if(error != 0) {
            showFileError(error,argv[1]);
            MPI_Abort(MPI_COMM_WORLD, error);
        }

        // Allocate and read input data
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

    int localStart, localPoints;
    getLocalRange(rank, size, numPoints, &localStart, &localPoints);
    
    int K = atoi(argv[2]); 
    int maxIterations = atoi(argv[3]);
    int minChanges = (int)(numPoints*atof(argv[4])/100.0);
    float maxThreshold = atof(argv[5]);

    // Allocate local arrays
    float *localData = (float*)calloc(localPoints * dimPoints, sizeof(float));
    float *centroids = (float*)calloc(K*dimPoints, sizeof(float));
    float *auxCentroids = (float*)calloc(K*dimPoints, sizeof(float));
    int *localClassMap = (int*)malloc(numPoints * sizeof(int));
    int *pointsPerClass = (int*)calloc(K*dimPoints * sizeof(int));

    if (localPoints == NULL || centroids == NULL || localClassMap == NULL || auxCentroids == NULL || localPointsPerClaster == NULL) {
        fprintf(stderr,"Host memory allocation error.\n");
        exit(MEMORY_ALLOCATION_ERR);
    }

    // Initialize data structures
    elementIntArray(localClassMap, DEFAULT_CLASS, numPoints);
    if (rank == 0) {
        initCentroids(data, centroids, K, numPoints, dimPoints);
    }

    MPI_Bcast(centroids, K * dimPoints, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Distribute data using MPI_Scatterv
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

    end = clock();
    if (rank == 0) {
        printf("\nMemory allocation: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
        fflush(stdout);
        sprintf(line,"\nMemory allocation: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
        outputMsg = strcat(outputMsg,line);
    }

    // Start clustering computation
    start = clock();

    /*
     * START Part to pararellilize
     */
    int it = 0;
    int changes = 0;
    float maxDist;

    do {
        it++;
        
        changes = 0;
        maxDist = 0.0f;

        assignDataToCentroids(localData, centroids, localClassMap, localPoints, dimPoints, K, &changes);

        updateLocalVariables(data, auxCentroids, localClassMap, pointsPerClass, numPoints, dimPoints, K);

        MPI_Allreduce(MPI_IN_PLACE, auxCentroids, K * dimPoints, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, pointsPerClass, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        updateCentroids(centroids, auxCentroids, pointsPerClass, dimPoints, K);

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
    end = clock();

    printf("\nComputation: %f seconds", (double)(end - start) / CLOCKS_PER_SEC);
    fflush(stdout);

    sprintf(line,"\n\nComputation: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
    outputMsg = strcat(outputMsg,line);

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

    CHECK_CUDA_ERROR(cudaMemcpy(classMap, d_classMap, numPoints * sizeof(int), cudaMemcpyDeviceToHost));

    error = writeResult(classMap, numPoints, argv[6]);
    if(error != 0) {
        showFileError(error, argv[6]);
        exit(error);
    }

    // Cleanup and free memory
    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_auxCentroids);
    cudaFree(d_classMap);
    cudaFree(d_pointsPerClass);
    cudaFree(d_distCentroids);
    cudaFree(d_changes);
    free(data);
    free(centroids);
    free(classMap);

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
