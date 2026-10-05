#include "meanshift/Dataset.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

using namespace std;

namespace meanshift {

Dataset::Dataset() {}

bool Dataset::loadFromCSV(const string& filename, bool hasHeader) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Cannot open file " << filename << endl;
        return false;
    }

    string line;
    if (hasHeader) {
        getline(file, line); // Skip header
    }

    data_.clear();
    while (getline(file, line)) {
        if (line.empty()) continue;

        stringstream ss(line);
        string token;
        Point p;
        while (getline(ss, token, ',')) {
            try {
                p.push_back(stod(token));
            } catch (const exception& e) {
                // In case of non-numeric data, we can ignore the point or handle it.
                // For simplicity, we assume all columns are features.
            }
        }
        if (!p.empty()) {
            data_.push_back(p);
        }
    }
    return true;
}

const Data& Dataset::getData() const {
    return data_;
}

void Dataset::standardize() {
    if (data_.empty()) return;

    size_t numFeatures = data_[0].size();
    Point means(numFeatures, 0.0);
    Point stdDevs(numFeatures, 0.0);

    // Calculate means
    for (const auto& point : data_) {
        for (size_t i = 0; i < numFeatures; ++i) {
            means[i] += point[i];
        }
    }
    for (size_t i = 0; i < numFeatures; ++i) {
        means[i] /= data_.size();
    }

    // Calculate standard deviations
    for (const auto& point : data_) {
        for (size_t i = 0; i < numFeatures; ++i) {
            stdDevs[i] += pow(point[i] - means[i], 2);
        }
    }
    for (size_t i = 0; i < numFeatures; ++i) {
        stdDevs[i] = sqrt(stdDevs[i] / data_.size());
    }

    // Standardize
    for (auto& point : data_) {
        for (size_t i = 0; i < numFeatures; ++i) {
            if (stdDevs[i] != 0) {
                point[i] = (point[i] - means[i]) / stdDevs[i];
            }
        }
    }
}

void Dataset::normalize() {
    if (data_.empty()) return;

    size_t numFeatures = data_[0].size();
    Point mins = data_[0];
    Point maxs = data_[0];

    // Find min and max for each feature
    for (const auto& point : data_) {
        for (size_t i = 0; i < numFeatures; ++i) {
            if (point[i] < mins[i]) mins[i] = point[i];
            if (point[i] > maxs[i]) maxs[i] = point[i];
        }
    }

    // Normalize
    for (auto& point : data_) {
        for (size_t i = 0; i < numFeatures; ++i) {
            if (maxs[i] - mins[i] != 0) {
                point[i] = (point[i] - mins[i]) / (maxs[i] - mins[i]);
            }
        }
    }
}

} // namespace meanshift
