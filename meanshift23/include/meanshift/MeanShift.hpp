#ifndef MEANSHIFT_HPP
#define MEANSHIFT_HPP

// ---------------------------------------------------------------------------
// BasicMeanShift<T, R, K>: Mean Shift clustering for any numeric data type.
//
//   T  input type  (Numeric):        int, short, long, unsigned, float, double, ...
//   R  compute type (RealNumber):    used for distances, weights and centroids.
//                                    Defaults to T for floating types, double for
//                                    integer types (the mean of ints is not an int).
//   K  kernel policy (KernelFunction<R>): GaussianKernel (default), FlatKernel,
//                                    EpanechnikovKernel, or any user callable.
//
// `MeanShift` is kept as an alias for BasicMeanShift<double>, so existing code
// compiles unchanged.
// ---------------------------------------------------------------------------

#include "meanshift/Concepts.hpp"
#include "meanshift/Dataset.hpp"
#include "meanshift/Kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace meanshift {

// ----- Cluster --------------------------------------------------------------
// A centroid is a *mean*, so it is stored in the compute type R even when the
// input data are integers.
template <RealNumber R>
struct BasicCluster {
    BasicPoint<R> centroid;
    std::vector<std::size_t> pointIndices;
};

using Cluster = BasicCluster<double>;   // backward-compatible name

// ----- euclideanDistance ----------------------------------------------------
// Works across types (e.g. an int data point vs. a double centroid).  Every
// coordinate is converted to R *before* subtracting: for unsigned types,
// a[i] - b[i] would otherwise wrap around (2u - 5u == 4294967293u).
template <RealNumber R, Numeric A, Numeric B>
R euclideanDistance(const BasicPoint<A>& a, const BasicPoint<B>& b) {
    R sum = R(0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        const R diff = static_cast<R>(a[i]) - static_cast<R>(b[i]);
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// ----- BasicMeanShift -------------------------------------------------------
template <Numeric T,
          RealNumber R = default_real_t<T>,
          KernelFunction<R> K = GaussianKernel>
class BasicMeanShift {
public:
    using value_type   = T;
    using real_type    = R;
    using kernel_type  = K;
    using point_type   = BasicPoint<T>;
    using data_type    = BasicData<T>;
    using real_point   = BasicPoint<R>;
    using cluster_type = BasicCluster<R>;

    explicit BasicMeanShift(R bandwidth = R(-1), R epsilon = R(1e-3), int maxIterations = 300)
        : bandwidth_(bandwidth), epsilon_(epsilon), maxIterations_(maxIterations) {}

    // Distance calculation (Euclidean), computed in R.
    template <Numeric A, Numeric B>
    static R euclideanDistance(const BasicPoint<A>& a, const BasicPoint<B>& b) {
        return meanshift::euclideanDistance<R>(a, b);
    }

    // Gaussian kernel weight (kept for backward compatibility and tests).
    R gaussianKernel(R distance) const { return GaussianKernel{}(distance, bandwidth_); }

    // Weight from the kernel policy K (what fit()/shiftPoint() actually use).
    R kernelWeight(R distance) const { return static_cast<R>(kernel_(distance, bandwidth_)); }

    R getBandwidth() const { return bandwidth_; }

    // Bandwidth selection (estimates if bandwidth <= 0)
    void estimateBandwidth(const data_type& data);

    // Main Mean Shift clustering algorithm
    void fit(const data_type& data);

    // Mean shift vector calculation and centroid updating.  The point being
    // shifted can be of any numeric type U (it is usually already of type R).
    template <Numeric U>
    real_point shiftPoint(const BasicPoint<U>& p, const data_type& data) const;

    // Get final clusters
    const std::vector<cluster_type>& getClusters() const { return clusters_; }

    // Get assignments (cluster index for each point)
    const std::vector<int>& getLabels() const { return labels_; }

    // Cluster evaluation: mean distance from each point to its centroid
    // (cohesion, lower is better).
    R evaluate(const data_type& data) const;

private:
    R bandwidth_;
    R epsilon_;
    int maxIterations_;
    [[no_unique_address]] K kernel_{};

    std::vector<cluster_type> clusters_;
    std::vector<int> labels_;
};

// ----- Member definitions ---------------------------------------------------

template <Numeric T, RealNumber R, KernelFunction<R> K>
void BasicMeanShift<T, R, K>::estimateBandwidth(const data_type& data) {
    if (data.empty()) return;

    // Simple heuristic: half the average distance between pairs of points.
    // To avoid O(N^2) on large datasets, we use at most the first 500 points.
    const std::size_t numSamples = std::min<std::size_t>(500, data.size());
    R totalDist = R(0);
    std::size_t count = 0;
    for (std::size_t i = 0; i < numSamples; ++i) {
        for (std::size_t j = i + 1; j < numSamples; ++j) {
            totalDist += euclideanDistance(data[i], data[j]);
            ++count;
        }
    }
    bandwidth_ = (count > 0) ? (totalDist / static_cast<R>(count)) * R(0.5) : R(1);
    if (bandwidth_ <= R(0)) bandwidth_ = R(1);
}

template <Numeric T, RealNumber R, KernelFunction<R> K>
template <Numeric U>
auto BasicMeanShift<T, R, K>::shiftPoint(const BasicPoint<U>& p, const data_type& data) const
    -> real_point {
    real_point shiftedP(p.begin(), p.end());          // converts U -> R
    real_point numerator(p.size(), R(0));
    R denominator = R(0);

    for (const auto& other : data) {
        const R weight = kernelWeight(euclideanDistance(p, other));
        for (std::size_t i = 0; i < p.size(); ++i) {
            numerator[i] += static_cast<R>(other[i]) * weight;
        }
        denominator += weight;
    }

    if (denominator > R(0)) {
        for (std::size_t i = 0; i < p.size(); ++i) {
            shiftedP[i] = numerator[i] / denominator;
        }
    }
    return shiftedP;
}

template <Numeric T, RealNumber R, KernelFunction<R> K>
void BasicMeanShift<T, R, K>::fit(const data_type& data) {
    clusters_.clear();
    labels_.clear();
    if (data.empty()) return;

    if (bandwidth_ <= R(0)) {
        estimateBandwidth(data);
    }

    // Moving copies of the data, already converted to the compute type R.
    std::vector<real_point> shiftedPoints;
    shiftedPoints.reserve(data.size());
    for (const auto& p : data) shiftedPoints.emplace_back(p.begin(), p.end());

    std::vector<bool> stopMoving(data.size(), false);

    for (int iter = 0; iter < maxIterations_; ++iter) {
        bool allStopped = true;

        for (std::size_t i = 0; i < shiftedPoints.size(); ++i) {
            if (stopMoving[i]) continue;

            real_point newPoint = shiftPoint(shiftedPoints[i], data);
            const R dist = euclideanDistance(shiftedPoints[i], newPoint);

            if (dist < epsilon_) {
                stopMoving[i] = true;
            } else {
                allStopped = false;
            }
            shiftedPoints[i] = std::move(newPoint);
        }

        if (allStopped) {
            break;
        }
    }

    // Cluster formation: merge converged points closer than bandwidth / 2.
    labels_.assign(data.size(), -1);

    for (std::size_t i = 0; i < shiftedPoints.size(); ++i) {
        int clusterIdx = 0;
        bool found = false;

        for (auto& cluster : clusters_) {
            if (euclideanDistance(shiftedPoints[i], cluster.centroid) < (bandwidth_ / R(2))) {
                cluster.pointIndices.push_back(i);
                labels_[i] = clusterIdx;
                found = true;
                break;
            }
            ++clusterIdx;
        }

        if (!found) {
            cluster_type newCluster;
            newCluster.centroid = shiftedPoints[i];
            newCluster.pointIndices.push_back(i);
            clusters_.push_back(std::move(newCluster));
            labels_[i] = static_cast<int>(clusters_.size() - 1);
        }
    }
}

template <Numeric T, RealNumber R, KernelFunction<R> K>
R BasicMeanShift<T, R, K>::evaluate(const data_type& data) const {
    if (clusters_.empty() || data.empty()) return R(0);

    R total = R(0);
    for (const auto& cluster : clusters_) {
        for (std::size_t idx : cluster.pointIndices) {
            total += euclideanDistance(data[idx], cluster.centroid);
        }
    }
    return total / static_cast<R>(data.size());
}

// Backward-compatible name.
using MeanShift = BasicMeanShift<double>;

// Compiled once in src/MeanShift.cpp; other translation units reuse it.
extern template class BasicMeanShift<int>;
extern template class BasicMeanShift<float>;
extern template class BasicMeanShift<double>;

} // namespace meanshift

#endif // MEANSHIFT_HPP
