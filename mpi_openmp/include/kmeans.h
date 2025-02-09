#ifndef KMEANS_H
#define KMEANS_H

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>

#include <omp.h>

#include "const.h"

void elementPaddedIntArray(int *array, int value, int size);

void elementPaddedFloatArray(int *array, int value, int size);

void elementIntArray(int *array, int value, int size);

void elementFloatArray(float *array, float value, int size);

void initCentroids(const float* data, float* centroids, const int K, const int n, const int dim);

void getLocalRange(int rank, int size, int totalPoints, int *start, int *count);

// OpenMP
void assignDataToCentroids(const float *data, const float *centroids, int *classMap, int numPoints, int dimPoints, int K, int *changes);

void updateLocalVariables(const float *data, float *auxCentroids, const int *classMap, int *pointsPerClass, int numPoints, int dimPoints, int K);

float updateCentroids(float *centroids, const float *auxCentroids,const int *pointsPerClass, int dimPoints, int K);

#endif