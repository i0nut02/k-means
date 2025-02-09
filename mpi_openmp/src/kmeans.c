#include "../include/kmeans.h"

static inline int min(int a, int b) {
    return (a < b) ? a : b;
}

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
    
    // Cache blocking parameters
    const int POINT_BLOCK_SIZE = 64;  // Process points in blocks
    const int DIM_BLOCK_SIZE = 32;    // Process dimensions in blocks
    const int CENTROID_BLOCK_SIZE = 8; // Process centroids in blocks
    
    // Parallelize over blocks of points for better cache utilization
    #pragma omp parallel reduction(+:localChanges)
    {
        // Thread-local storage to avoid false sharing
        float* local_distances = (float*)aligned_alloc(64, K * sizeof(float));
        
        // Process points in blocks
        #pragma omp for schedule(dynamic, 1) nowait
        for (int iBlock = 0; iBlock < numPoints; iBlock += POINT_BLOCK_SIZE) {
            int iEnd = min(iBlock + POINT_BLOCK_SIZE, numPoints);
            
            // Process each point in the block
            for (int i = iBlock; i < iEnd; i++) {
                float minDist = FLT_MAX;
                int newClass = -1;
                
                // Prefetch next point's data
                __builtin_prefetch(&data[(i + 1) * dimPoints], 0, 3);
                
                // Initialize distances for each centroid
                for (int k = 0; k < K; k++) {
                    local_distances[k] = 0.0f;
                }
                
                // Process dimensions in blocks
                for (int dBlock = 0; dBlock < dimPoints; dBlock += DIM_BLOCK_SIZE) {
                    int dEnd = min(dBlock + DIM_BLOCK_SIZE, dimPoints);
                    
                    // Process centroids in blocks
                    for (int kBlock = 0; kBlock < K; kBlock += CENTROID_BLOCK_SIZE) {
                        int kEnd = min(kBlock + CENTROID_BLOCK_SIZE, K);
                        
                        // Prefetch next centroid block
                        if (kBlock + CENTROID_BLOCK_SIZE < K) {
                            __builtin_prefetch(&centroids[(kBlock + CENTROID_BLOCK_SIZE) * dimPoints + dBlock], 0, 3);
                        }
                        
                        // Calculate partial distances for this block
                        for (int k = kBlock; k < kEnd; k++) {
                            float partial_dist = 0.0f;
                            
                            // Unrolled inner dimension loop
                            for (int d = dBlock; d < dEnd; d += 4) {
                                if (d + 3 < dEnd) {
                                    float diff0 = data[i * dimPoints + d] - centroids[k * dimPoints + d];
                                    float diff1 = data[i * dimPoints + d + 1] - centroids[k * dimPoints + d + 1];
                                    float diff2 = data[i * dimPoints + d + 2] - centroids[k * dimPoints + d + 2];
                                    float diff3 = data[i * dimPoints + d + 3] - centroids[k * dimPoints + d + 3];
                                    
                                    partial_dist = fmaf(diff0, diff0, partial_dist);
                                    partial_dist = fmaf(diff1, diff1, partial_dist);
                                    partial_dist = fmaf(diff2, diff2, partial_dist);
                                    partial_dist = fmaf(diff3, diff3, partial_dist);
                                } else {
                                    // Handle remaining dimensions
                                    for (int d_rem = d; d_rem < dEnd; d_rem++) {
                                        float diff = data[i * dimPoints + d_rem] - centroids[k * dimPoints + d_rem];
                                        partial_dist = fmaf(diff, diff, partial_dist);
                                    }
                                }
                            }
                            local_distances[k] += partial_dist;
                        }
                    }
                }
                
                // Find minimum distance and corresponding centroid
                for (int k = 0; k < K; k++) {
                    if (local_distances[k] < minDist) {
                        minDist = local_distances[k];
                        newClass = k;
                    }
                }
                
                // Update class if needed
                if (classMap[i] != newClass) {
                    classMap[i] = newClass;
                    localChanges++;
                }
            }
        }
        
        free(local_distances);
    }
    
    *changes = localChanges;
}


void updateLocalVariables(const float *data, float *auxCentroids, const int *classMap, 
    int *pointsPerClass, int numPoints, int dimPoints, int K) {

    #pragma omp parallel
    {
        int* localPointsPerClass = (int*) calloc(K, sizeof(int));
        float* localAuxCentroids = (float*) calloc(K * dimPoints, sizeof(float));

		#pragma omp for
        for(int i = 0; i < numPoints; i++) {
            int class = classMap[i];
            localPointsPerClass[class] += 1;

            for(int j = 0; j < dimPoints; j++) {
                localAuxCentroids[class * dimPoints + j] += data[i * dimPoints + j];
            }
        }

		#pragma omp critical
        {
            for (int k = 0; k < K; k++) {
                pointsPerClass[k] += localPointsPerClass[k];
                for (int j = 0; j < dimPoints; j++)
                {
                    auxCentroids[k * dimPoints + j] += localAuxCentroids[k * dimPoints + j];
                }
            }
        }
        free(localPointsPerClass);
        free(localAuxCentroids);
    }
 }

float updateCentroids(float *centroids, float *auxCentroids, 
    const int *pointsPerClass, int dimPoints, int K) {

    #pragma omp for
    for(int k = 0; k < K; k++) {
        int pointsInClass = pointsPerClass[k];
        for(int j = 0; j < dimPoints; j++){
            auxCentroids[k * dimPoints + j] = auxCentroids[k * dimPoints + j] / pointsInClass;
        } 
    }

    float maxDist = 0.0f;

    #pragma omp parallel for reduction(max:maxDist)
    for(int k = 0; k < K; k++){

        float dist = 0.0f;
        #pragma omp reduction(+:dist)
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

