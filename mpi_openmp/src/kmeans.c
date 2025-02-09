void assignDataToCentroids(const float *data, const float *centroids, int *classMap, 
    int numPoints, int dimPoints, int K, int *changes) {
int localChanges = 0;

// Ensure proper alignment and padding for thread-private centroids
const int padK = ((K + 15) / 16) * 16;  // Pad K to multiple of 16

#pragma omp parallel reduction(+:localChanges)
{
int tid = omp_get_thread_num();
int nthreads = omp_get_num_threads();

// Divide work among threads
int chunk = (numPoints + nthreads - 1) / nthreads;
int start = tid * chunk;
int end = min(start + chunk, numPoints);

#pragma omp for schedule(static, PAD_INT)
for (int i = start; i < end; i++) {
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
}
*changes = localChanges;
}

void updateLocalVariables(const float *data, float *auxCentroids, const int *classMap, 
   int *pointsPerClass, int numPoints, int dimPoints, int K) {
const int padK = ((K + 15) / 16) * 16;  // Pad K to multiple of 16
const int padDim = ((dimPoints + 15) / 16) * 16;  // Pad dimensions

#pragma omp parallel
{
int tid = omp_get_thread_num();
int nthreads = omp_get_num_threads();

// Thread-private arrays
float *localAuxCentroids = (float *)aligned_alloc(64, padK * padDim * sizeof(float));
int *localPointsPerClass = (int *)aligned_alloc(64, padK * sizeof(int));

// Initialize thread-private arrays
memset(localAuxCentroids, 0, padK * padDim * sizeof(float));
memset(localPointsPerClass, 0, padK * sizeof(int));

// Divide work among threads
int chunk = (numPoints + nthreads - 1) / nthreads;
int start = tid * chunk;
int end = min(start + chunk, numPoints);

// Process local chunk
for (int i = start; i < end; i++) {
int class_id = classMap[i];
localPointsPerClass[class_id]++;

#pragma omp simd
for (int d = 0; d < dimPoints; d++) {
localAuxCentroids[class_id * padDim + d] += data[i * dimPoints + d];
}
}

// Reduce thread-private results to shared arrays
#pragma omp critical
{
for (int k = 0; k < K; k++) {
pointsPerClass[k] += localPointsPerClass[k];
for (int d = 0; d < dimPoints; d++) {
auxCentroids[k * dimPoints + d] += localAuxCentroids[k * padDim + d];
}
}
}

free(localAuxCentroids);
free(localPointsPerClass);
}
}

float updateCentroids(float *centroids, const float *auxCentroids, 
const int *pointsPerClass, int dimPoints, int K) {
float globalMaxDist = 0.0f;
const int padK = ((K + 15) / 16) * 16;
const int padDim = ((dimPoints + 15) / 16) * 16;

#pragma omp parallel
{
int tid = omp_get_thread_num();
float localMaxDist = 0.0f;

#pragma omp for reduction(max:globalMaxDist)
for (int k = 0; k < K; k++) {
float dist = 0.0f;
int kPoints = pointsPerClass[k];

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
globalMaxDist = fmaxf(globalMaxDist, localMaxDist);
}
return globalMaxDist;
}