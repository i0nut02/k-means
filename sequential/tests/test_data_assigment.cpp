#include <gtest/gtest.h>
#include "../include/kmeans.h"

// Helper function to compare classifications
bool compareClassifications(const int* actual, const int* expected, const int n) {
    for (int i = 0; i < n; i++) {
        if (actual[i] != expected[i]) {
            return false;
        }
    }
    return true;
}

// Test cases
TEST(AssignDataToCentroidsTests, BasicAssignment) {
    const int K = 2;
    const int n = 4;
    const int dim = 2;

    float data[8] = {1.0, 1.0,  // Point 1
                     2.0, 2.0,  // Point 2
                     8.0, 8.0,  // Point 3
                     9.0, 9.0}; // Point 4

    float centroids[4] = {1.0, 1.0,  // Centroid 1
                          8.0, 8.0}; // Centroid 2

    int classifications[4] = {-1, -1, -1, -1};

    assignDataToCentroids(data, centroids, classifications, K, n, dim);

    int expectedClassifications[4] = {0, 0, 1, 1};  // Points 1 and 2 -> Centroid 1, Points 3 and 4 -> Centroid 2

    EXPECT_TRUE(compareClassifications(classifications, expectedClassifications, n));
}

TEST(AssignDataToCentroidsTests, PointsEquallyDistantFromTwoCentroids) {
    const int K = 2;
    const int n = 1;
    const int dim = 2;

    float data[2] = {5.0, 5.0};  // Point equally distant from both centroids
    float centroids[4] = {0.0, 0.0,  // Centroid 1
                          10.0, 10.0}; // Centroid 2

    int classifications[1] = {-1};

    assignDataToCentroids(data, centroids, classifications, K, n, dim);

    // Since the point is equidistant, it can be assigned to either centroid; we'll accept centroid 0 as a default
    int expectedClassifications[1] = {0};  

    EXPECT_EQ(classifications[0], expectedClassifications[0]);
}

TEST(AssignDataToCentroidsTests, SingleCluster) {
    const int K = 1;
    const int n = 3;
    const int dim = 2;

    float data[6] = {1.0, 2.0,  // Point 1
                     3.0, 4.0,  // Point 2
                     5.0, 6.0}; // Point 3

    float centroids[2] = {3.0, 4.0};  // Single centroid

    int classifications[3] = {-1, -1, -1};

    assignDataToCentroids(data, centroids, classifications, K, n, dim);

    int expectedClassifications[3] = {0, 0, 0};  // All points -> Single centroid

    EXPECT_TRUE(compareClassifications(classifications, expectedClassifications, n));
}

TEST(AssignDataToCentroidsTests, MultipleDimensions) {
    const int K = 3;
    const int n = 3;
    const int dim = 3;

    float data[9] = {1.0, 1.0, 1.0,  // Point 1
                     5.0, 5.0, 5.0,  // Point 2
                     9.0, 9.0, 9.0}; // Point 3

    float centroids[9] = {0.0, 0.0, 0.0,  // Centroid 1
                          4.0, 4.0, 4.0,  // Centroid 2
                          10.0, 10.0, 10.0}; // Centroid 3

    int classifications[3] = {-1, -1, -1};

    assignDataToCentroids(data, centroids, classifications, K, n, dim);

    int expectedClassifications[3] = {0, 1, 2};  // Each point -> Closest centroid

    EXPECT_TRUE(compareClassifications(classifications, expectedClassifications, n));
}
