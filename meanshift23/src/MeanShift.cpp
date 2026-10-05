#include "meanshift/MeanShift.hpp"
#include <cmath>
#include <algorithm>

using namespace std;

namespace meanshift {

MeanShift::MeanShift(double bandwidth, double epsilon, int maxIterations)
    : bandwidth_(bandwidth), epsilon_(epsilon), maxIterations_(maxIterations) {}

double MeanShift::euclideanDistance(const Point& a, const Point& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(sum);
}

double MeanShift::gaussianKernel(double distance) const {
    return exp(-0.5 * (distance * distance) / (bandwidth_ * bandwidth_));
}

void MeanShift::estimateBandwidth(const Data& data) {
    if (data.empty()) return;
    
    // Simple heuristic: average distance between all pairs (or a subset)
    // To avoid O(N^2) on large datasets, we use a subset.
    size_t numSamples = min<size_t>(500, data.size());
    double totalDist = 0;
    int count = 0;
    for (size_t i = 0; i < numSamples; ++i) {
        for (size_t j = i + 1; j < numSamples; ++j) {
            totalDist += euclideanDistance(data[i], data[j]);
            count++;
        }
    }
    bandwidth_ = (count > 0) ? (totalDist / count) * 0.5 : 1.0;
    if (bandwidth_ <= 0) bandwidth_ = 1.0;
}

Point MeanShift::shiftPoint(const Point& p, const Data& data) const {
    Point shiftedP = p;
    Point numerator(p.size(), 0.0);
    double denominator = 0.0;

    for (const auto& other_p : data) {
        double dist = euclideanDistance(p, other_p);
        double weight = gaussianKernel(dist);

        for (size_t i = 0; i < p.size(); ++i) {
            numerator[i] += other_p[i] * weight;
        }
        denominator += weight;
    }

    if (denominator > 0) {
        for (size_t i = 0; i < p.size(); ++i) {
            shiftedP[i] = numerator[i] / denominator;
        }
    }

    return shiftedP;
}

void MeanShift::fit(const Data& data) {
    if (data.empty()) return;

    if (bandwidth_ <= 0) {
        estimateBandwidth(data);
    }

    vector<Point> shiftedPoints = data;
    vector<bool> stopMoving(data.size(), false);

    for (int iter = 0; iter < maxIterations_; ++iter) {
        int maxDistMoved = 0;
        bool allStopped = true;

        for (size_t i = 0; i < shiftedPoints.size(); ++i) {
            if (stopMoving[i]) continue;

            Point newPoint = shiftPoint(shiftedPoints[i], data);
            double dist = euclideanDistance(shiftedPoints[i], newPoint);
            
            if (dist < epsilon_) {
                stopMoving[i] = true;
            } else {
                allStopped = false;
            }
            shiftedPoints[i] = newPoint;
        }

        if (allStopped) {
            break;
        }
    }

    // Cluster formation
    clusters_.clear();
    labels_.assign(data.size(), -1);

    for (size_t i = 0; i < shiftedPoints.size(); ++i) {
        int clusterIdx = 0;
        bool found = false;

        for (auto& cluster : clusters_) {
            if (euclideanDistance(shiftedPoints[i], cluster.centroid) < (bandwidth_ / 2.0)) {
                cluster.pointIndices.push_back(i);
                labels_[i] = clusterIdx;
                found = true;
                break;
            }
            clusterIdx++;
        }

        if (!found) {
            Cluster newCluster;
            newCluster.centroid = shiftedPoints[i];
            newCluster.pointIndices.push_back(i);
            clusters_.push_back(newCluster);
            labels_[i] = clusters_.size() - 1;
        }
    }
}

const vector<Cluster>& MeanShift::getClusters() const {
    return clusters_;
}

const vector<int>& MeanShift::getLabels() const {
    return labels_;
}

// Simplified intra-cluster variance evaluation (lower is better, cohesion)
double MeanShift::evaluate(const Data& data) const {
    if (clusters_.empty() || data.empty()) return 0.0;
    
    double totalVariance = 0.0;
    for (const auto& cluster : clusters_) {
        double clusterVar = 0.0;
        for (size_t idx : cluster.pointIndices) {
            clusterVar += euclideanDistance(data[idx], cluster.centroid);
        }
        totalVariance += clusterVar;
    }
    return totalVariance / data.size();
}

} // namespace meanshift
