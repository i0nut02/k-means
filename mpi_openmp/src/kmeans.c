#include "../include/kmeans.h"


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
void assignDataToCentroids(const float *data, const float *centroids, int *classMap, int numPoints, int dimPoints, int K, int *changes) {
    int localChanges = 0;
    #pragma omp parallel for reduction(+:localChanges) schedule(static, PAD_INT)
    for (int i = 0; i < numPoints; i++) {
        float minDist = FLT_MAX;
        int newClass = classMap[i]; // no false sharing for it
        int thread_id = omp_get_thread_num();

        for (int k = thread_id; k < K + thread_id; k++) { // reduce false sharing
            int index = k % K;
            float dist = 0.0f;

            #pragma omp simd reduction(+:dist)
            for (int d = 0; d < dimPoints; d++) {
                float diff = data[i * dimPoints + d] - centroids[index * dimPoints + d];
                dist = fmaf(diff, diff, dist);
            }
            if (dist < minDist) {
                minDist = dist;
                newClass = index;
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
    #pragma omp parallel for reduction(+:pointsPerClass[:K], auxCentroids[:K * dimPoints]) schedule(static, PAD_INT)
    for (int i = 0; i < numPoints; i++) {
        int class_id = classMap[i]; // reduce false sharing
        pointsPerClass[class_id]++;
        
        #pragma omp simd
        for (int d = 0; d < dimPoints; d++) {
            auxCentroids[class_id * dimPoints + d] += data[i * dimPoints + d];
        }
    }
}

float updateCentroids(float *centroids, const float *auxCentroids,const int *pointsPerClass, int dimPoints, int K) {
    float localMaxDist = 0.0f;

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();

        #pragma omp for reduction(max:localMaxDist)
        for (int k = thread_id; k < K + thread_id; k ++) {
            int index = k % K

            float dist = 0.0f;
            int kPoints = pointsPerClass[k]; // hard to avoid false sharing 

            if (kPoints > 0) {
                float invKPoints = 1.0f / kPoints;

                #pragma omp simd reduction(+:dist)
                for (int d = 0; d < dimPoints; d++) {
                    float old = centroids[k * dimPoints + d];
                    float newCentroid = auxCentroids[k * dimPoints + d] * invKPoints;
                    centroids[k * dimPoints + d] = newCentroid;
                    dist = fmaf(newCentroid - old, newCentroid - old, dist);
                }
            }
            localMaxDist = fmaxf(localMaxDist, dist);
        }
    }
    return localMaxDist;
}