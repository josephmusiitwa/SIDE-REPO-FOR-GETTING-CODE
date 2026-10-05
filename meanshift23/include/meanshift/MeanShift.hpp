#ifndef MEANSHIFT_HPP
#define MEANSHIFT_HPP

#include "meanshift/Dataset.hpp"
#include <vector>
#include <cstddef>

namespace meanshift {

struct Cluster {
    Point centroid;
    std::vector<std::size_t> pointIndices;
};

class MeanShift {
public:
    MeanShift(double bandwidth = -1.0, double epsilon = 1e-3, int maxIterations = 300);

    // Distance calculation (Euclidean)
    static double euclideanDistance(const Point& a, const Point& b);

    // Kernel function (Gaussian)
    double gaussianKernel(double distance) const;

    // Bandwidth selection (estimates if bandwidth <= 0)
    void estimateBandwidth(const Data& data);

    // Main Mean Shift clustering algorithm
    void fit(const Data& data);

    // Mean shift vector calculation and Centroid updating
    Point shiftPoint(const Point& p, const Data& data) const;

    // Get final clusters
    const std::vector<Cluster>& getClusters() const;

    // Get assignments (cluster index for each point)
    const std::vector<int>& getLabels() const;

    // Cluster evaluation (e.g. Silhouette Score or simplified cohesion)
    double evaluate(const Data& data) const;

private:
    double bandwidth_;
    double epsilon_;
    int maxIterations_;

    std::vector<Cluster> clusters_;
    std::vector<int> labels_;
};

} // namespace meanshift

#endif // MEANSHIFT_HPP
