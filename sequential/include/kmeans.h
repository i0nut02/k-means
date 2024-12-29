#ifndef KMEANS_H
#define KMEANS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "const.h"

void printMatrixF(const float* arr, const int rows, const int cols);

void printMatrixI(const int* arr, const int rows, const int cols);

float euclideanDistance(const float* point, const float* center, const int dim);

void zeroFloatMatriz(float *matrix, const int rows, const int columns);

void elementIntArray(int* array, const int el, const int n);

void initCentroids(const float* data, float* centroids, const int K, const int n, const int dim);

int assignDataToCentroids(const float* data, const float* centroids, int* classifications, const int K, const int n, const int dim);

float updateCentroids(const float* data, float* centroids, int* classifications, int* pointsPerCluster, float* auxCentroids, const int K, const int n, const int dim);

#ifdef __cplusplus
}
#endif

#endif