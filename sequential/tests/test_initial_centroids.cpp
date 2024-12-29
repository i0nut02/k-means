#include <gtest/gtest.h>
#include "../include/kmeans.h"

// A helper function to compare two centroids for equality with a small tolerance
bool compareCentroids(const float* centroids1, const float* centroids2, int K, int dim) {
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < dim; j++) {
            if (std::abs(centroids1[i * dim + j] - centroids2[i * dim + j]) > 1e-5) {
                return false;
            }
        }
    }
    return true;
}

TEST(InitCentroidsTests, MorePointsThanClusters) {
    const int K = 3;  // 3 clusters
    const int n = 5;  // 5 data points
    const int dim = 2;  // 2-dimensional data points

    float data[10] = {1.0, 2.0,   // Point 1
                     2.0, 3.0,   // Point 2
                     3.0, 4.0,   // Point 3
                     4.0, 5.0,   // Point 4
                     5.0, 6.0};  // Point 5

    float centroids[6] = {0.0, 0.0,  // Initial centroids
                          0.0, 0.0,
                          0.0, 0.0};

    // Call the function to initialize the centroids
    initCentroids(data, centroids, K, n, dim);

    // Manually define the expected centroids based on SEED (since this is deterministic)
    float expectedCentroids[6] = {1.0, 2.0,   // Expected centroid 1
                                  2.0, 3.0,   // Expected centroid 2
                                  3.0, 4.0};  // Expected centroid 3

    // Check if the centroids are correctly initialized
    EXPECT_TRUE(compareCentroids(centroids, expectedCentroids, K, dim));
}

TEST(InitCentroidsTests, EqualPointsOfClusters) {
    const int K = 3;  // 3 clusters
    const int n = 5;  // 5 data points
    const int dim = 2;  // 2-dimensional data points

    float data[10] = {1.0, 2.0,   // Point 1
                     2.0, 3.0,   // Point 2
                     3.0, 4.0};  // Point 5

    float centroids[6] = {0.0, 0.0,  // Initial centroids
                          0.0, 0.0,
                          0.0, 0.0};

    // Call the function to initialize the centroids
    initCentroids(data, centroids, K, n, dim);

    // Manually define the expected centroids based on SEED (since this is deterministic)
    float expectedCentroids[6] = {1.0, 2.0,   // Expected centroid 1
                                  2.0, 3.0,   // Expected centroid 2
                                  3.0, 4.0};  // Expected centroid 3

    // Check if the centroids are correctly initialized
    EXPECT_TRUE(compareCentroids(centroids, expectedCentroids, K, dim));
}
