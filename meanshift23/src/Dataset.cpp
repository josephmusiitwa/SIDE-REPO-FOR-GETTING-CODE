#include "meanshift/Dataset.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

namespace meanshift {

Dataset::Dataset() {}

bool Dataset::loadFromCSV(const std::string& filename, bool hasHeader) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }

    std::string line;
    if (hasHeader) {
        std::getline(file, line); // Skip header
    }

    data_.clear();
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string token;
        Point p;
        while (std::getline(ss, token, ',')) {
            try {
                p.push_back(std::stod(token));
            } catch (const std::exception& e) {
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

    std::size_t numFeatures = data_[0].size();
    Point means(numFeatures, 0.0);
    Point stdDevs(numFeatures, 0.0);

    // Calculate means
    for (const auto& point : data_) {
        for (std::size_t i = 0; i < numFeatures; ++i) {
            means[i] += point[i];
        }
    }
    for (std::size_t i = 0; i < numFeatures; ++i) {
        means[i] /= data_.size();
    }

    // Calculate standard deviations
    for (const auto& point : data_) {
        for (std::size_t i = 0; i < numFeatures; ++i) {
            stdDevs[i] += std::pow(point[i] - means[i], 2);
        }
    }
    for (std::size_t i = 0; i < numFeatures; ++i) {
        stdDevs[i] = std::sqrt(stdDevs[i] / data_.size());
    }

    // Standardize
    for (auto& point : data_) {
        for (std::size_t i = 0; i < numFeatures; ++i) {
            if (stdDevs[i] != 0) {
                point[i] = (point[i] - means[i]) / stdDevs[i];
            }
        }
    }
}

void Dataset::normalize() {
    if (data_.empty()) return;

    std::size_t numFeatures = data_[0].size();
    Point mins = data_[0];
    Point maxs = data_[0];

    // Find min and max for each feature
    for (const auto& point : data_) {
        for (std::size_t i = 0; i < numFeatures; ++i) {
            if (point[i] < mins[i]) mins[i] = point[i];
            if (point[i] > maxs[i]) maxs[i] = point[i];
        }
    }

    // Normalize
    for (auto& point : data_) {
        for (std::size_t i = 0; i < numFeatures; ++i) {
            if (maxs[i] - mins[i] != 0) {
                point[i] = (point[i] - mins[i]) / (maxs[i] - mins[i]);
            }
        }
    }
}

} // namespace meanshift
