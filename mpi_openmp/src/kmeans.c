#include "../include/kmeans.h"

void elementIntArray(int *array, int value, int size) {
    for(int i = 0; i < size; i++) {
        array[i] = value;
    }
}

void elementFloatArray(float *array, float value, int size) {
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

// Get local data range for each MPI process
void getLocalRange(int rank, int size, int totalPoints, int *start, int *count) {
    int quotient = totalPoints / size;
    int remainder = totalPoints % size;
    *start = quotient * rank + (rank < remainder ? rank : remainder);
    *count = quotient + (rank < remainder ? 1 : 0);
}

// OpenMP
void assignDataToCentroids(const float *data, const float *centroids, int *classMap, 
    int numPoints, int dimPoints, int K, int *changes) {
    int localChanges = 0;

    #pragma omp parallel for private(newClass, minDist, k, dist) reduction(+:localChanges) schedule(guided)
    for (int i = 0; i < numPoints; i++) {
        float minDist = FLT_MAX;
        int newClass = classMap[i];

        for (int k = 0; k < K; k++) {
            float dist = 0.0f;

            #pragma omp simd reduction(+:dist)
            for (int d = 0; d < dimPoints; d++) {
                float diff = data[i * dimPoints + d] - centroids[k * dimPoints + d];
                dist = fmaf(diff, diff, dist);
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


void updateLocalVariables(const float *data, float *auxCentroids, const int *classMap, 
    int *pointsPerClass, int numPoints, int dimPoints, int K) {
 
     #pragma omp parallel for reduction(+:pointsPerClass[:K], auxCentroids[:K * dimPoints])
     for (int i = 0; i < numPoints; i++) {
         int class_id = classMap[i];
         pointsPerClass[class_id]++;
 
         #pragma omp simd
         for (int d = 0; d < dimPoints; d++) {
             auxCentroids[class_id * dimPoints + d] += data[i * dimPoints + d];
         }
     }
 }

float updateCentroids(float *centroids, const float *auxCentroids, 
    const int *pointsPerClass, int dimPoints, int K) {

    #pragma omp for
    for(int k = 0; k < K; k++) {
        int pointsInClass = pointsPerClass[k];
        for(int j = 0; j < dimPoints; j++){
            auxCentroids[k * dimPoints + j] /= pointsInClass;
        } 
    }

    float maxDist = 0.0f;

    #pragma omp parallel for private(dist) reduction(max:maxDist)
    for(int k = 0; k < K; k++){

        float dist = 0.0f;
        #pragma omp simd reduction(+:dist)
        for (int d = 0; d < dimPoints; d++) {
            float diff = centroids[k * dimPoints + d] - auxCentroids[k * dimPoints + d];
            dist = fmaf(diff, diff, dist);
        }

        if(dist > maxDist) {
            maxDist = dist;
        }
    }

    return maxDist;
}

