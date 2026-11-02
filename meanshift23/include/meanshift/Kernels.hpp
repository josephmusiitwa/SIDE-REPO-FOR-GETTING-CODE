#ifndef MEANSHIFT_KERNELS_HPP
#define MEANSHIFT_KERNELS_HPP

// ---------------------------------------------------------------------------
// Kernel policies for BasicMeanShift.
//
// Each kernel is a tiny stateless struct with a *templated* call operator, so
// the same kernel works for float, double and long double.  BasicMeanShift
// takes the kernel as a template parameter (a "policy"), which means the call
// is resolved at compile time and can be inlined: no virtual functions.
// ---------------------------------------------------------------------------

#include "meanshift/Concepts.hpp"
#include <cmath>

namespace meanshift {

// Gaussian kernel: weight = exp(-d^2 / (2 h^2)).  Every point has some
// influence, closer points have more.  This is the library's default.
struct GaussianKernel {
    template <RealNumber R>
    R operator()(R distance, R bandwidth) const {
        return std::exp(R(-0.5) * (distance * distance) / (bandwidth * bandwidth));
    }
};

// Flat (uniform) kernel: weight = 1 inside the bandwidth, 0 outside.  This is
// the classic textbook Mean Shift; it is cheaper but can leave a point with no
// neighbours (BasicMeanShift::shiftPoint handles a zero denominator).
struct FlatKernel {
    template <RealNumber R>
    R operator()(R distance, R bandwidth) const {
        return distance <= bandwidth ? R(1) : R(0);
    }
};

// Epanechnikov kernel: weight = 1 - (d/h)^2 inside the bandwidth, 0 outside.
struct EpanechnikovKernel {
    template <RealNumber R>
    R operator()(R distance, R bandwidth) const {
        const R u = distance / bandwidth;
        return u <= R(1) ? R(1) - u * u : R(0);
    }
};

static_assert(KernelFunction<GaussianKernel, double>);
static_assert(KernelFunction<FlatKernel, float>);
static_assert(KernelFunction<EpanechnikovKernel, long double>);

} // namespace meanshift

#endif // MEANSHIFT_KERNELS_HPP
