#include <gtest/gtest.h>
#include "../include/kmeans.h"


TEST(KMeansTests, MaxDistanceDecreasing) {
    const int K = 3;   // Number of clusters
    const int n = 6;   // Number of data points
    const int dim = 2; // Dimensionality of data points

    // Example data points
    float data[n * dim] = {
        1.0, 1.0,
        115.5, 25.0,
        332.0, 443.0,
        543.0, 744.0,
        3.5, -205.0,
        -40.5, -65.0
    };

    // Buffers
    float centroids[K * dim];                 // Centroids
    int classifications[n];                   // Cluster assignments
    int pointsPerCluster[K];                  // Points per cluster
    float auxCentroids[K * dim];              // Auxiliary centroids for recalculation

    // Initialize centroids
    initCentroids(data, centroids, K, n, dim);

    // Variable to track maximum distance
    float prevMaxDist = std::numeric_limits<float>::max();

    // Perform multiple iterations
    for (int iter = 0; iter < 10; ++iter) {
        // Assign data points to centroids
        assignDataToCentroids(data, centroids, classifications, K, n, dim);

        // Update centroids and get maximum distance
        float maxDist = updateCentroids(data, centroids, classifications, pointsPerCluster, auxCentroids, K, n, dim);

        // Ensure max distance is decreasing
        EXPECT_LE(maxDist, prevMaxDist) << "Max distance did not decrease at iteration " << iter;
        prevMaxDist = maxDist;
    }
}
