#include "meanshift/Dataset.hpp"
#include "meanshift/MeanShift.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>

void testEuclideanDistance() {
    meanshift::Point p1 = {0.0, 0.0};
    meanshift::Point p2 = {3.0, 4.0};
    double dist = meanshift::MeanShift::euclideanDistance(p1, p2);
    if (dist != 5.0) throw std::runtime_error("testEuclideanDistance failed");
    std::cout << "testEuclideanDistance passed." << std::endl;
}

void testMeanShiftClustering() {
    meanshift::MeanShift ms(2.0, 1e-3, 100);
    meanshift::Data data = {
        {0.0, 0.0}, {0.1, 0.1}, {0.2, 0.2},
        {10.0, 10.0}, {10.1, 10.1}, {10.2, 10.2}
    };
    ms.fit(data);
    auto clusters = ms.getClusters();
    if (clusters.size() != 2) throw std::runtime_error("testMeanShiftClustering failed");
    std::cout << "testMeanShiftClustering passed." << std::endl;
}

int main() {
    testEuclideanDistance();
    testMeanShiftClustering();
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
