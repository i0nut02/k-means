#include "../include/kmeans.h"


void elementPaddedIntArray(int *array, int value, int size) {
    #pragma omp parallel for
    for(int i = 0; i < size; i++) {
        array[i * PAD_INT] = value;
    }
}

void elementPaddedFloatArray(float *array, float value, int size) {
    #pragma omp parallel for
    for(int i = 0; i < size; i++) {
        array[i * PAD_FLOAT] = value;
    }
}

void elementIntArray(int *array, int value, int size) {
    #pragma omp parallel for
    for(int i = 0; i < size; i++) {
        array[i] = value;
    }
}

void elementFloatArray(float *array, float value, int size) {
    #pragma omp parallel for
    for(int i = 0; i < size; i++) {
        array[i] = value;
    }
}

void initCentroids(const float* data, float* paddedCentroids, const int K, const int n, const int dim) {
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
        memcpy(&paddedCentroids[i*dim * PAD_FLOAT], &data[idx*dim], dim * sizeof(float));
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
void assignDataToCentroids(const float *data, const float *paddedCentroids, int *paddedClassMap, int numPoints, int dimPoints, int K, int *changes) {
    int localChanges = 0;
    #pragma omp parallel for reduction(+:localChanges)
    for (int i = 0; i < numPoints; i++) {
        float minDist = FLT_MAX;
        int newClass = paddedClassMap[i * PAD_INT];

        for (int k = 0; k < K; k++) {
            float dist = 0.0f;

            #pragma omp simd reduction(+:dist)
            for (int d = 0; d < dimPoints; d++) {
                float diff = data[i * dimPoints + d] - paddedCentroids[(k * dimPoints + d) * PAD_FLOAT];
                dist = fmaf(diff, diff, dist);
            }
            if (dist < minDist) {
                minDist = dist;
                newClass = k;
            }
        }
        if (paddedClassMap[i * PAD_INT] != newClass) {
            paddedClassMap[i * PAD_INT] = newClass;
            localChanges++;
        }
    }
    *changes = localChanges;
    return;
}

void updateLocalVariables(const float *data, float *paddedAuxCentroids, const int *paddedClassMap, int *paddedPointsPerClass, int numPoints, int dimPoints, int K) {
    #pragma omp parallel for reduction(+:paddedPointsPerClass[:K * PAD_INT], paddedAuxCentroids[:K * dimPoints * PAD_FLOAT])
    for (int i = 0; i < numPoints; i++) {
        int class_id = paddedClassMap[i * PAD_INT];
        paddedPointsPerClass[class_id * PAD_INT]++;
        
        #pragma omp simd
        for (int d = 0; d < dimPoints; d++) {
            paddedAuxCentroids[(class_id * dimPoints + d) * PAD_FLOAT] += data[i * dimPoints + d];
        }
    }
}

float updateCentroids(float *paddedCentroids, const float *paddedAuxCentroids,const int *paddedPointsPerClass, int dimPoints, int K) {
    float localMaxDist = 0.0f;

    #pragma omp parallel for reduction(max:localMaxDist)
    for (int k = 0; k < K; k++) {
        float dist = 0.0f;
        int kPoints = paddedPointsPerClass[k * PAD_INT];
        if (kPoints > 0) {
            float invKPoints = 1.0f / kPoints;

            #pragma omp simd reduction(+:dist)
            for (int d = 0; d < dimPoints; d++) {
                float old = paddedCentroids[(k * dimPoints + d) * PAD_FLOAT];
                float newCentroid = paddedAuxCentroids[(k * dimPoints + d) * PAD_FLOAT] * invKPoints;
                paddedCentroids[(k * dimPoints + d) * PAD_FLOAT] = newCentroid;
                dist = fmaf(newCentroid - old, newCentroid - old, dist);
            }
        }
        localMaxDist = fmaxf(localMaxDist, dist);
    }
    return localMaxDist;
}