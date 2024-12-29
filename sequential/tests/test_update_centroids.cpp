#include <gtest/gtest.h>
#include "../include/kmeans.h"

TEST(UpdateCentroidsTests, OnePointPerCluster) {
    const int K = 3;  // 3 clusters
    const int n = 3;  // 3 data points
    const int dim = 2;  // 2-dimensional data points

    float data[6] = {1.0, 2.0,  // Point 1
                     3.0, 4.0,  // Point 2
                     5.0, 6.0}; // Point 3

    float centroids[6] = {1.0, 2.5,  // Initial centroid 1
                          3.0, 3.0,  // Initial centroid 2
                          7.0, 5.0}; // Initial centroid 3

    int classifications[3] = {0, 1, 2};  // Points assigned to different clusters
    int pointsPerCluster[3] = {1, 1, 1};  // Each cluster has 1 point
    float auxCentroids[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    
    // distance from point 3 to initial centroid 3
    float expDist = sqrt((7.0 - 5.0)*(7.0 - 5.0) + (6.0 - 5.0)*(6.0 - 5.0));

    float maxDist = updateCentroids(data, centroids, classifications, pointsPerCluster, auxCentroids, K, n, dim);
    
    
    EXPECT_FLOAT_EQ(maxDist, expDist);

    EXPECT_FLOAT_EQ(centroids[0], 1.0);
    EXPECT_FLOAT_EQ(centroids[1], 2.0);
    EXPECT_FLOAT_EQ(centroids[2], 3.0);
    EXPECT_FLOAT_EQ(centroids[3], 4.0);
    EXPECT_FLOAT_EQ(centroids[4], 5.0);
    EXPECT_FLOAT_EQ(centroids[5], 6.0);
}

TEST(UpdateCentroidsTests, MultiplePointsInSameCluster) {
    const int K = 2;  // 2 clusters
    const int n = 4;  // 4 data points
    const int dim = 2;  // 2-dimensional data points

    float data[8] = {1.0, 2.0,   // Point 1
                     2.0, 3.0,   // Point 2
                     8.0, 9.0,   // Point 3
                     9.0, 10.0}; // Point 4

    float centroids[4] = {1.0, 2.0,   // Initial centroid 1
                          8.0, 9.0};  // Initial centroid 2

    int classifications[4] = {0, 0, 1, 1};  // Points 1 and 2 belong to cluster 0, Points 3 and 4 to cluster 1
    int pointsPerCluster[2] = {2, 2};        // Each cluster has 2 points
    float auxCentroids[4] = {0.0, 0.0, 0.0, 0.0};
    
    // distance from new centroid 1.5, 2.5 to the old one 1.0, 2.0
    float expDist = sqrt(0.5*0.5 + 0.5*0.5);

    // Call the function
    float maxDist = updateCentroids(data, centroids, classifications, pointsPerCluster, auxCentroids, K, n, dim);

    EXPECT_FLOAT_EQ(maxDist, expDist);

    // Expected centroids: cluster 0 = (1+2)/2, (2+3)/2 = (1.5, 2.5), cluster 1 = (8+9)/2, (9+10)/2 = (8.5, 9.5)
    EXPECT_FLOAT_EQ(centroids[0], 1.5);
    EXPECT_FLOAT_EQ(centroids[1], 2.5);
    EXPECT_FLOAT_EQ(centroids[2], 8.5);
    EXPECT_FLOAT_EQ(centroids[3], 9.5);
}

TEST(UpdateCentroidsTests, AllPointsInSameCluster) {
    const int K = 1;  // 1 cluster
    const int n = 5;  // 5 data points
    const int dim = 2;  // 2-dimensional data points

    float data[10] = {1.0, 2.0,   // Point 1
                     2.0, 3.0,   // Point 2
                     3.0, 4.0,   // Point 3
                     4.0, 5.0,   // Point 4
                     5.0, 6.0};  // Point 5

    float centroids[2] = {0.0, 0.0};  // Initial centroid

    int classifications[5] = {0, 0, 0, 0, 0};  // All points assigned to cluster 0
    int pointsPerCluster[1] = {5};               // Only one cluster with all points
    float auxCentroids[2] = {0.0, 0.0};

    // THe distance between initial centroid 0, 0 and the new one 3, 4
    float expDist = sqrt((3.0*3.0) + (4.0*4.0));

    // Call the function
    float maxDist = updateCentroids(data, centroids, classifications, pointsPerCluster, auxCentroids, K, n, dim);

    EXPECT_FLOAT_EQ(maxDist, expDist);

    // Expected centroid: (1+2+3+4+5)/5, (2+3+4+5+6)/5 = (3.0, 4.0)
    EXPECT_FLOAT_EQ(centroids[0], 3.0);
    EXPECT_FLOAT_EQ(centroids[1], 4.0);
}

TEST(UpdateCentroidsTests, DifferentDimensionality) {
    const int K = 2;  // 2 clusters
    const int n = 3;  // 3 data points
    const int dim = 3;  // 3-dimensional data points

    float data[9] = {1.0, 2.0, 3.0,   // Point 1
                     4.0, 5.0, 6.0,   // Point 2
                     7.0, 8.0, 9.0};  // Point 3

    float centroids[6] = {1.0, 2.0, 3.0,   // Initial centroid 1
                          6.0, 6.0, 6.0};  // Initial centroid 2

    int classifications[3] = {0, 1, 1};  // Points 1 and 3 belong to cluster 0, Point 2 to cluster 1
    int pointsPerCluster[2] = {2, 1};     // Cluster 0 has 2 points, Cluster 1 has 1 point
    float auxCentroids[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // Call the function
    float maxDist = updateCentroids(data, centroids, classifications, pointsPerCluster, auxCentroids, K, n, dim);

    // It will be the distance between (6, 6, 6) and (5.5, 6.5, 7.5)
    float expDist = sqrt(0.5*0.5 + 0.5*0.5 + 1.5*1.5);

    EXPECT_FLOAT_EQ(maxDist, expDist);

    // Expected centroids: cluster 0 = (1+7)/2, (2+8)/2, (3+9)/2 = (4.0, 5.0, 6.0)
    // cluster 1 = (4.0, 5.0, 6.0) for point 2
    EXPECT_FLOAT_EQ(centroids[0], 1.0);
    EXPECT_FLOAT_EQ(centroids[1], 2.0);
    EXPECT_FLOAT_EQ(centroids[2], 3.0);
    EXPECT_FLOAT_EQ(centroids[3], 5.5);
    EXPECT_FLOAT_EQ(centroids[4], 6.5);
    EXPECT_FLOAT_EQ(centroids[5], 7.5);
}
