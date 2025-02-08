#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>
#include <cuda_runtime.h>

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


void elementIntArray(int *array, int value, int size);
void initCentroids(const float* data, float* centroids, const int K, const int n, const int dim);

// Implementation of utility functions
void elementIntArray(int *array, int value, int size) {
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

// Macros for CUDA error checking and min/max
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// CUDA kernels
__global__ void assignDataToCentroidsKernel(
    const float* data,
    const float* centroids,
    int* classMap,
    const int numPoints,
    const int dimPoints,
    const int K,
    int* changes
) {
    __shared__ float sh_changes;
    if (threadIdx.x == 0) {
        sh_changes = 0;
    }
    __syncthreads();
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < numPoints) {
        float minDist = FLT_MAX;
        int oldClass = classMap[tid];
        int newClass = oldClass;
        
        for (int k = 0; k < K; k++) { // For each Centroid
            float dist = 0.0f;
            for (int d = 0; d < dimPoints; d++) { // For each Coordinate in centroid k-th
                float diff = data[tid * dimPoints + d] - centroids[k * dimPoints + d];
                dist += diff * diff;
            }
            if (dist < minDist) {
                minDist = dist;
                newClass = k;
            }
        }
        
        if (oldClass != newClass) {
            atomicAdd(&sh_changes, 1); // It will be added at most one time per Data Point
            classMap[tid] = newClass;
        }
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        atomicAdd(changes, sh_changes);
    }
}

__global__ void updateCentroidsKernel(
    const float* data,
    float* auxCentroids,
    int* pointsPerClass,
    const int* classMap,
    const int numPoints,
    const int dimPoints,
    const int K
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < numPoints) {
        int class_id = classMap[tid];
        
        for (int d = threadIdx.x; d < threadIdx.x + dimPoints; d++) { // access in a cascade way
            int index = d % dimPoints;
            atomicAdd(&auxCentroids[class_id * dimPoints + index], data[tid * dimPoints + index]);
        }
        atomicAdd(&pointsPerClass[class_id], 1);
    }
}

__device__ float atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
            __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void finalizeCentroidsKernel(
    float* centroids,
    const float* auxCentroids,
    const int* pointsPerClass,
    const int K,
    const int dimPoints,
    float* distCentroids
) {
    __shared__ float sh_distCentroids;
    if (threadIdx.x == 0) {
        sh_distCentroids = 0.0f;
    }
    __syncthreads();
    
    int k = blockIdx.x * blockDim.x + threadIdx.x; // Each thread handles one centroid

    if (k < K) { // Check bounds for number of centroids
        float dist = 0.0f;
        for (int d = 0; d < dimPoints; ++d) {
            int index = k * dimPoints + d; // Index for the dimension
            float oldCentroid = centroids[index];
            float newCentroid = auxCentroids[index] / (float)pointsPerClass[k];
            
            centroids[index] = newCentroid;
            
            // Calculate the distance for this dimension
            float diff = newCentroid - oldCentroid;
            dist += diff * diff;
        }
        // Update the global max distance
        atomicMaxFloat(&sh_distCentroids, dist); // update for each centroid
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        atomicMaxFloat(distCentroids, sh_distCentroids);
    }
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
    
    int K = atoi(argv[2]); 
    int maxIterations = atoi(argv[3]);
    int minChanges = (int)(numPoints*atof(argv[4])/100.0);
    float maxThreshold = atof(argv[5]);

    float *centroids = (float*)calloc(K*dimPoints, sizeof(float));
    int *classMap = (int*)malloc(numPoints * sizeof(int));

    if (centroids == NULL || classMap == NULL) {
        fprintf(stderr,"Host memory allocation error.\n");
        exit(MEMORY_ALLOCATION_ERR);
    }

    // Initialize data structures
    elementIntArray(classMap, DEFAULT_CLASS, numPoints);
    initCentroids(data, centroids, K, numPoints, dimPoints);

    float *d_data, *d_centroids, *d_auxCentroids;
    int *d_classMap, *d_pointsPerClass;
    float *d_distCentroids;
    int *d_changes;

    CHECK_CUDA_ERROR(cudaMalloc(&d_data, numPoints * dimPoints * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_centroids, K * dimPoints * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_auxCentroids, K * dimPoints * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_classMap, numPoints * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pointsPerClass, K * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_distCentroids, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_changes, sizeof(int)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_data, data, numPoints * dimPoints * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_centroids, centroids, K * dimPoints * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_classMap, classMap, numPoints * sizeof(int), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (numPoints + blockSize - 1) / blockSize;
    int blocksPerGrid = (K + blockSize - 1) / blockSize;

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
    int changes = 0;
    float maxDist;

    do {
        it++;
        
        changes = 0;
        maxDist = 0.0f;
        CHECK_CUDA_ERROR(cudaMemset(d_changes, 0, sizeof(int)));
        CHECK_CUDA_ERROR(cudaMemset(d_distCentroids, 0, sizeof(float)));
        CHECK_CUDA_ERROR(cudaMemset(d_auxCentroids, 0, K * dimPoints * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMemset(d_pointsPerClass, 0, K * sizeof(int)));

        assignDataToCentroidsKernel<<<numBlocks, blockSize>>>(
            d_data, d_centroids, d_classMap, numPoints, dimPoints, K, d_changes
        );
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        updateCentroidsKernel<<<numBlocks, blockSize>>>(
            d_data, d_auxCentroids, d_pointsPerClass, d_classMap,
            numPoints, dimPoints, K
        );
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        finalizeCentroidsKernel<<<blocksPerGrid, blockSize>>>(
            d_centroids, d_auxCentroids, d_pointsPerClass, K, dimPoints, d_distCentroids
        );
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        CHECK_CUDA_ERROR(cudaMemcpy(&changes, d_changes, sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(&maxDist, d_distCentroids, sizeof(float), cudaMemcpyDeviceToHost));

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
