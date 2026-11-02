#ifndef DATASET_HPP
#define DATASET_HPP

// ---------------------------------------------------------------------------
// BasicDataset<T>: loads numeric CSV data and scales it.
//
// T is the type of every feature value (int, short, float, double, ...).
// It must satisfy the Numeric concept, so BasicDataset<std::string> or
// BasicDataset<bool> are rejected at compile time with a readable error.
//
// The non-template name `Dataset` is kept as an alias for BasicDataset<double>
// (like std::string is an alias for std::basic_string<char>), so all existing
// code keeps compiling unchanged.
// ---------------------------------------------------------------------------

#include "meanshift/Concepts.hpp"

#include <charconv>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace meanshift {

// ----- Point / Data ---------------------------------------------------------
// A point is one CSV row; Data is a whole table (a row-major matrix).
template <Numeric T> using BasicPoint = std::vector<T>;
template <Numeric T> using BasicData  = std::vector<BasicPoint<T>>;

using Point = BasicPoint<double>;   // backward-compatible names
using Data  = BasicData<double>;

// ----- parseNumber ----------------------------------------------------------
// Converts one CSV cell to T.  Returns false if the cell is not a complete,
// in-range number of type T.  Examples for T = int:
//   " 42 " -> 42 (surrounding spaces / '\r' are ignored)
//   "+7"   -> 7
//   "2.5"  -> rejected (not an integer; we do not silently truncate)
//   "abc"  -> rejected
// For T = short, "70000" is rejected (out of range) instead of wrapping.
// For floating types, "nan" and "inf" are rejected: they would poison every
// mean and distance computed later.
template <Numeric T>
bool parseNumber(std::string_view text, T& out) {
    const auto isSpace = [](char c) {
        return c == ' ' || c == '\t' || c == '\r' || c == '\n';
    };
    while (!text.empty() && isSpace(text.front())) text.remove_prefix(1);
    while (!text.empty() && isSpace(text.back()))  text.remove_suffix(1);
    if (!text.empty() && text.front() == '+') text.remove_prefix(1);
    if (text.empty()) return false;

    T value{};
    const char* first = text.data();
    const char* last  = text.data() + text.size();
    const auto [ptr, ec] = std::from_chars(first, last, value);
    if (ec != std::errc{} || ptr != last) return false;

    if constexpr (std::floating_point<T>) {
        if (!std::isfinite(value)) return false;
    }
    out = value;
    return true;
}

// ----- BasicDataset ---------------------------------------------------------
template <Numeric T>
class BasicDataset {
public:
    using value_type = T;
    using point_type = BasicPoint<T>;
    using data_type  = BasicData<T>;

    BasicDataset() = default;
    explicit BasicDataset(data_type data) : data_(std::move(data)) {}

    // Dataset loading from CSV.  Cells that are not valid numbers of type T
    // (names, labels, "2.5" when T is int, ...) are skipped.
    bool loadFromCSV(const std::string& filename, bool hasHeader = true);

    // Get the raw data
    const data_type& getData() const { return data_; }

    // Feature scaling: Standard Scaler (z-score normalization).
    // Constrained: z-scores are fractional, so scaling an int dataset in place
    // would truncate every value to -1, 0 or 1.  Convert first with cast<>().
    void standardize() requires RealNumber<T>;

    // Feature scaling: Min-Max Scaler (values in [0, 1]).  Same constraint.
    void normalize() requires RealNumber<T>;

    // Converts every value to another numeric type U, e.g.
    //   BasicDataset<int> raw; ... ; auto real = raw.cast<double>(); real.standardize();
    // Converting floating -> integer truncates toward zero (static_cast rules).
    template <Numeric U>
    BasicDataset<U> cast() const {
        BasicData<U> out;
        out.reserve(data_.size());
        for (const auto& p : data_) {
            BasicPoint<U> q;
            q.reserve(p.size());
            for (T v : p) q.push_back(static_cast<U>(v));
            out.push_back(std::move(q));
        }
        return BasicDataset<U>(std::move(out));
    }

private:
    data_type data_;
};

// ----- Member definitions ---------------------------------------------------
// Templates must be visible where they are used, so the definitions live in
// the header.  The common types are also explicitly instantiated once in
// src/Dataset.cpp (see the `extern template` lines at the bottom).

template <Numeric T>
bool BasicDataset<T>::loadFromCSV(const std::string& filename, bool hasHeader) {
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
        if (line.empty() || line == "\r") continue;

        std::stringstream ss(line);
        std::string token;
        point_type p;
        while (std::getline(ss, token, ',')) {
            T value{};
            if (parseNumber(token, value)) {
                p.push_back(value);
            }
            // Non-numeric cells (e.g. a name column) are ignored.
        }
        if (!p.empty()) {
            data_.push_back(std::move(p));
        }
    }
    return true;
}

template <Numeric T>
void BasicDataset<T>::standardize() requires RealNumber<T> {
    if (data_.empty()) return;

    const std::size_t numFeatures = data_[0].size();
    const T n = static_cast<T>(data_.size());
    point_type means(numFeatures, T(0));
    point_type stdDevs(numFeatures, T(0));

    // Calculate means
    for (const auto& point : data_) {
        for (std::size_t i = 0; i < numFeatures; ++i) {
            means[i] += point[i];
        }
    }
    for (std::size_t i = 0; i < numFeatures; ++i) {
        means[i] /= n;
    }

    // Calculate standard deviations
    for (const auto& point : data_) {
        for (std::size_t i = 0; i < numFeatures; ++i) {
            const T diff = point[i] - means[i];
            stdDevs[i] += diff * diff;
        }
    }
    for (std::size_t i = 0; i < numFeatures; ++i) {
        stdDevs[i] = std::sqrt(stdDevs[i] / n);
    }

    // Standardize
    for (auto& point : data_) {
        for (std::size_t i = 0; i < numFeatures; ++i) {
            if (stdDevs[i] != T(0)) {
                point[i] = (point[i] - means[i]) / stdDevs[i];
            }
        }
    }
}

template <Numeric T>
void BasicDataset<T>::normalize() requires RealNumber<T> {
    if (data_.empty()) return;

    const std::size_t numFeatures = data_[0].size();
    point_type mins = data_[0];
    point_type maxs = data_[0];

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
            const T range = maxs[i] - mins[i];
            if (range != T(0)) {
                point[i] = (point[i] - mins[i]) / range;
            }
        }
    }
}

// Backward-compatible name.
using Dataset = BasicDataset<double>;

// Compiled once in src/Dataset.cpp; other translation units reuse it.
extern template class BasicDataset<int>;
extern template class BasicDataset<float>;
extern template class BasicDataset<double>;

} // namespace meanshift

#endif // DATASET_HPP
