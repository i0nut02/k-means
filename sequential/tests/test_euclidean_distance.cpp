#include <gtest/gtest.h>
#include "../include/kmeans.h"

TEST(EuclideanDistanceTests, SamePoints) {
    float point1[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    float point2[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    EXPECT_EQ(euclideanDistance(point1, point2, 10), 0.0);
}


TEST(EuclideanDistanceTests, ZeroDimention) {
    float point1[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    float point2[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    EXPECT_EQ(euclideanDistance(point1, point2, 0), 0.0);
}

TEST(EuclideanDistanceTests, DifferentPoints) {
    float point1[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    float point2[10] = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    float expectedDistance = sqrt(9*9 + 7*7 + 5*5 + 3*3 + 1*1 + 1*1 +3*3 + 5*5 + 7*7 + 9*9);
    EXPECT_FLOAT_EQ(euclideanDistance(point1, point2, 10), expectedDistance);
}

TEST(EuclideanDistanceTests, DifferentPointsWithNegativeValues) {
    float point1[10] = {-1, -2, -3, -4, -5, -6, -7, -8, -9, -10};
    float point2[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    float expectedDistance = sqrt(2*2 + 4*4 + 6*6 + 8*8 + 10*10 + 12*12 + 14*14 + 16*16 + 18*18 + 20*20);
    EXPECT_FLOAT_EQ(euclideanDistance(point1, point2, 10), expectedDistance);
}

TEST(EuclideanDistanceTests, DifferentDimensionality) {
    float point1[9] = {1, 2, 3, 4, 5, -1, -1, -1, -1};
    float point2[5] = {5, 4, 3, 2, 1};
    float expectedDistance = sqrt(4*4 + 2*2 + 0*0 + 2*2 + 4*4);
    EXPECT_FLOAT_EQ(euclideanDistance(point1, point2, 5), expectedDistance);
}

TEST(EuclideanDistanceTests, OnePointAtOrigin) {
    float point1[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    float point2[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    float expectedDistance = sqrt(1*1 + 2*2 + 3*3 + 4*4 + 5*5 + 6*6 + 7*7 + 8*8 + 9*9 + 10*10);
    EXPECT_FLOAT_EQ(euclideanDistance(point1, point2, 10), expectedDistance);
}
