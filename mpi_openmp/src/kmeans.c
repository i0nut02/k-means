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

    #pragma omp parallel for reduction(+:localChanges) schedule(static, PAD_INT)
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
 
     #pragma omp parallel for reduction(+:pointsPerClass[:K], auxCentroids[:K * dimPoints]) schedule(static, PAD_INT)
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
    float globalMaxDist = 0.0f;

    for (int k = 0; k < K; k++) {
        float dist = 0.0f;
        int kPoints = pointsPerClass[k];

        if (kPoints > 0) {
            float invKPoints = 1.0f / kPoints;

            #pragma omp parallel for reduction(+:dist) schedule(static, PAD_INT)
            for (int d = 0; d < dimPoints; d++) {
                float old = centroids[k * dimPoints + d];
                float newCentroid = auxCentroids[k * dimPoints + d] * invKPoints;
                centroids[k * dimPoints + d] = newCentroid;
                dist = fmaf(newCentroid - old, newCentroid - old, dist);
            }
        }
        // Update the global max distance using reduction max
        globalMaxDist = fmaxf(globalMaxDist, dist);
    }

    return globalMaxDist;
}

