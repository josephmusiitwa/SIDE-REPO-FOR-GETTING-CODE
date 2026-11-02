#include "meanshift/Dataset.hpp"
#include "meanshift/MeanShift.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <cstdio>

using namespace std;

void testEuclideanDistance() {
    meanshift::Point p1 = {0.0, 0.0};
    meanshift::Point p2 = {3.0, 4.0};
    double dist = meanshift::MeanShift::euclideanDistance(p1, p2);
    if (dist != 5.0) throw runtime_error("testEuclideanDistance failed");
    cout << "testEuclideanDistance passed." << endl;
}

void testMeanShiftClustering() {
    meanshift::MeanShift ms(2.0, 1e-3, 100);
    meanshift::Data data = {
        {0.0, 0.0}, {0.1, 0.1}, {0.2, 0.2},
        {10.0, 10.0}, {10.1, 10.1}, {10.2, 10.2}
    };
    ms.fit(data);
    auto clusters = ms.getClusters();
    if (clusters.size() != 2) throw runtime_error("testMeanShiftClustering failed");
    cout << "testMeanShiftClustering passed." << endl;
}

void testGaussianKernel() {
    meanshift::MeanShift ms(2.0, 1e-3, 100);
    double w0 = ms.gaussianKernel(0.0);
    double w2 = ms.gaussianKernel(2.0);
    if (abs(w0 - 1.0) > 1e-6 || abs(w2 - exp(-0.5)) > 1e-6) {
        throw runtime_error("testGaussianKernel failed");
    }
    cout << "testGaussianKernel passed." << endl;
}

void testShiftPoint() {
    meanshift::MeanShift ms(2.0, 1e-3, 100);
    meanshift::Data data = {
        {0.0, 0.0}, {2.0, 2.0}
    };
    meanshift::Point mid = {1.0, 1.0};
    meanshift::Point shifted = ms.shiftPoint(mid, data);
    if (abs(shifted[0] - 1.0) > 1e-6 || abs(shifted[1] - 1.0) > 1e-6) {
        throw runtime_error("testShiftPoint failed");
    }
    cout << "testShiftPoint passed." << endl;
}

void testEstimateBandwidth() {
    meanshift::MeanShift ms;
    meanshift::Data data = {
        {0.0, 0.0}, {0.1, 0.1}, {0.2, 0.2},
        {10.0, 10.0}, {10.1, 10.1}, {10.2, 10.2}
    };
    ms.estimateBandwidth(data);
    ms.fit(data);
    if (ms.getClusters().size() != 2) throw runtime_error("testEstimateBandwidth failed");
    cout << "testEstimateBandwidth passed." << endl;
}

void testGetLabels() {
    meanshift::MeanShift ms(2.0, 1e-3, 100);
    meanshift::Data data = {
        {0.0, 0.0}, {0.1, 0.1}, {0.2, 0.2},
        {10.0, 10.0}, {10.1, 10.1}, {10.2, 10.2}
    };
    ms.fit(data);
    auto labels = ms.getLabels();
    if (labels.size() != 6 ||
        labels[0] != labels[1] || labels[1] != labels[2] ||
        labels[3] != labels[4] || labels[4] != labels[5] ||
        labels[0] == labels[3]) {
        throw runtime_error("testGetLabels failed");
    }
    cout << "testGetLabels passed." << endl;
}

void testEvaluate() {
    meanshift::MeanShift ms(2.0, 1e-3, 100);
    meanshift::Data data = {
        {0.0, 0.0}, {0.1, 0.1}, {0.2, 0.2},
        {10.0, 10.0}, {10.1, 10.1}, {10.2, 10.2}
    };
    ms.fit(data);
    double score = ms.evaluate(data);
    if (score <= 0.0 || score > 0.5) throw runtime_error("testEvaluate failed");
    cout << "testEvaluate passed." << endl;
}

void testEmptyData() {
    meanshift::MeanShift ms;
    meanshift::Data emptyData;
    ms.fit(emptyData);
    if (!ms.getClusters().empty() || !ms.getLabels().empty() || ms.evaluate(emptyData) != 0.0) {
        throw runtime_error("testEmptyData failed");
    }
    cout << "testEmptyData passed." << endl;
}

void testLoadFromCSV() {
    const string tempFile = "test_temp_data.csv";
    ofstream out(tempFile);
    out << "Feature1,Feature2\n";
    out << "1.0,2.0\n";
    out << "3.0,4.0\n";
    out.close();

    meanshift::Dataset dataset;
    bool loaded = dataset.loadFromCSV(tempFile, true);
    remove(tempFile.c_str());

    const auto& data = dataset.getData();
    if (!loaded || data.size() != 2 || data[0][0] != 1.0 || data[1][1] != 4.0) {
        throw runtime_error("testLoadFromCSV failed");
    }
    if (dataset.loadFromCSV("non_existent_file.csv", true)) {
        throw runtime_error("testLoadFromCSV failed");
    }
    cout << "testLoadFromCSV passed." << endl;
}

void testStandardize() {
    const string tempFile = "test_temp_std.csv";
    ofstream out(tempFile);
    out << "Feature1,Feature2\n";
    out << "1.0,10.0\n";
    out << "3.0,30.0\n";
    out.close();

    meanshift::Dataset dataset;
    dataset.loadFromCSV(tempFile, true);
    remove(tempFile.c_str());

    dataset.standardize();
    const auto& data = dataset.getData();
    if (data.size() != 2 ||
        abs(data[0][0] - (-1.0)) > 1e-6 || abs(data[1][0] - 1.0) > 1e-6 ||
        abs(data[0][1] - (-1.0)) > 1e-6 || abs(data[1][1] - 1.0) > 1e-6) {
        throw runtime_error("testStandardize failed");
    }
    cout << "testStandardize passed." << endl;
}

void testNormalize() {
    const string tempFile = "test_temp_norm.csv";
    ofstream out(tempFile);
    out << "Feature1,Feature2\n";
    out << "10.0,50.0\n";
    out << "20.0,100.0\n";
    out << "30.0,150.0\n";
    out.close();

    meanshift::Dataset dataset;
    dataset.loadFromCSV(tempFile, true);
    remove(tempFile.c_str());

    dataset.normalize();
    const auto& data = dataset.getData();
    if (data.size() != 3 ||
        abs(data[0][0] - 0.0) > 1e-6 || abs(data[1][0] - 0.5) > 1e-6 || abs(data[2][0] - 1.0) > 1e-6 ||
        abs(data[0][1] - 0.0) > 1e-6 || abs(data[1][1] - 0.5) > 1e-6 || abs(data[2][1] - 1.0) > 1e-6) {
        throw runtime_error("testNormalize failed");
    }
    cout << "testNormalize passed." << endl;
}

int main() {
    testEuclideanDistance();
    testMeanShiftClustering();
    testGaussianKernel();
    testShiftPoint();
    testEstimateBandwidth();
    testGetLabels();
    testEvaluate();
    testEmptyData();
    testLoadFromCSV();
    testStandardize();
    testNormalize();
    cout << "All tests passed!" << endl;
    return 0;
}
