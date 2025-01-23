#include "../include/kmeans.h"

void printMatrixF(const float* arr, const int rows, const int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%lf ", arr[i*cols + j]);
        }
        printf("\n");
    }
}

void printMatrixI(const int* arr, const int rows, const int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", arr[i*cols + j]);
        }
        printf("\n");
    }
}

/*
    Return the euclidian diustance between two points for
    the firsts n dimention/coordinates
*/
float euclideanDistance(const float* point1, const float* point2, const int dim) {
    float dist = 0.0;

    for (int i = 0; i < dim; i++) {
        dist += (point1[i] - point2[i]) * (point1[i] - point2[i]); 
    }
    return dist;
}

/*
    Modify the input vector for the defined rows and colums, and 
    the array will be considered as a matrix
*/
void zeroFloatMatriz(float *matrix, const int rows, const int columns) {
    memset(matrix, 0, rows*columns*sizeof(float));
    return;
}

/*
    Set to 0.0 the first n elements of the array
*/
void elementIntArray(int* array, const int el, const int n) {
    for (int i = 0; i < n; i++) {
        array[i] = el;
    }
    return;
}

/*
    Consider the first k datapoints as centroids
*/
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

int assignDataToCentroids(const float* data, const float* centroids, int* classifications, const int K, const int n, const int dim) {
    int changes = 0;

    for (int i = 0; i < n; i++) {
        int nearestCentroid = -1;
        float distNearestCentroid = FLT_MAX;

        for (int j = 0; j < K; j++) {
            float distance = euclideanDistance(&data[i*dim], &centroids[j*dim], dim);
            if (distance < distNearestCentroid) {
                nearestCentroid = j;
                distNearestCentroid = distance;
            }
        }
        if (classifications[i] != nearestCentroid) {
            changes++;
        }
        classifications[i] = nearestCentroid;
    }
    return changes;
}

float updateCentroids(const float* data, float* centroids, int* classifications, int* pointsPerCluster, float* auxCentroids, const int K, const int n, const int dim) {
    zeroFloatMatriz(auxCentroids, K, dim);
    elementIntArray(pointsPerCluster, 0, K);

    // for each point make redo the new centroid
    for (int i = 0; i < n; i++) {
        int cluster = classifications[i];
        pointsPerCluster[cluster] += 1;

        for (int j = 0; j < dim; j++) {
            /*
                The division can't be done here considering the
                notation of floating points that has a little bit
                of problems with division because do not ensure precision
            */
            auxCentroids[cluster * dim + j] += data[i*dim+ j];
        }
    }

    float maxDist = FLT_MIN;
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < dim; j++) {
            auxCentroids[i*dim + j] /= pointsPerCluster[i];
        }
        float dist = euclideanDistance(&centroids[i*dim], &auxCentroids[i*dim], dim);
        if (dist > maxDist) {
            maxDist = dist;
        }
    }
    memcpy(centroids, auxCentroids, K * dim * sizeof(float));
    
    return maxDist;
}
