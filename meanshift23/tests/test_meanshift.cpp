// ---------------------------------------------------------------------------
// Tests for the templated Mean Shift library.
//
// Two layers:
//   1. Compile-time tests (static_assert): check the concepts accept every
//      numeric type and reject bool, char, std::string, pointers, ...
//      If one of these fails, the test program does not even compile.
//   2. Run-time "typed" tests: every test is a function template that is run
//      once for each type in NumericTypes (short ... long double).
//
// The program prints one line per test and type, and returns a non-zero exit
// code if anything failed, so CTest reports the failure.
// ---------------------------------------------------------------------------

#include "meanshift/Concepts.hpp"
#include "meanshift/Dataset.hpp"
#include "meanshift/Kernels.hpp"
#include "meanshift/MeanShift.hpp"

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

using namespace meanshift;

// ===========================================================================
// Minimal test harness
// ===========================================================================

static int g_failures = 0;
static int g_checks   = 0;

template <class T>
constexpr std::string_view typeName() {
    if constexpr (std::is_same_v<T, short>)                   return "short";
    else if constexpr (std::is_same_v<T, unsigned short>)     return "unsigned short";
    else if constexpr (std::is_same_v<T, int>)                return "int";
    else if constexpr (std::is_same_v<T, unsigned int>)       return "unsigned int";
    else if constexpr (std::is_same_v<T, long>)               return "long";
    else if constexpr (std::is_same_v<T, unsigned long>)      return "unsigned long";
    else if constexpr (std::is_same_v<T, long long>)          return "long long";
    else if constexpr (std::is_same_v<T, unsigned long long>) return "unsigned long long";
    else if constexpr (std::is_same_v<T, signed char>)        return "int8_t";
    else if constexpr (std::is_same_v<T, unsigned char>)      return "uint8_t";
    else if constexpr (std::is_same_v<T, float>)              return "float";
    else if constexpr (std::is_same_v<T, double>)             return "double";
    else if constexpr (std::is_same_v<T, long double>)        return "long double";
    else                                                      return "?";
}

#define CHECK(cond)                                                            \
    do {                                                                       \
        ++g_checks;                                                            \
        if (!(cond)) {                                                         \
            ++g_failures;                                                      \
            std::cerr << "  FAILED [" << __func__ << "] line " << __LINE__     \
                      << ": " #cond << '\n';                                   \
        }                                                                      \
    } while (0)

template <RealNumber R>
bool near(R a, R b, R tol) { return std::abs(a - b) <= tol; }

// Runs `test` once for every type in Ts... and reports per type.
template <class... Ts, class Test>
void forEachType(std::string_view testName, Test test) {
    ([&] {
        const int before = g_failures;
        test.template operator()<Ts>();
        std::cout << (g_failures == before ? "  ok    " : "  FAIL  ")
                  << testName << " <" << typeName<Ts>() << ">\n";
    }(), ...);
}

#define NUMERIC_TYPES                                                          \
    short, unsigned short, int, unsigned int, long, unsigned long, long long,  \
    unsigned long long, std::int8_t, std::uint8_t, float, double, long double

#define REAL_TYPES float, double, long double

// Two well-separated blobs.  Integer coordinates so the same data is valid for
// every type, including uint8_t (max 255) and int8_t (max 127).
template <Numeric T>
BasicData<T> twoBlobs() {
    return {
        {T(0),   T(0)},   {T(1),   T(1)},   {T(2),   T(2)},
        {T(100), T(100)}, {T(101), T(101)}, {T(102), T(102)},
    };
}

// Writes a small CSV file and returns its name (unique per type).
template <Numeric T>
std::string writeTempCSV(std::string_view tag, std::string_view content) {
    std::string name = "test_tmp_" + std::string(tag) + "_" + std::string(typeName<T>()) + ".csv";
    for (char& c : name) if (c == ' ') c = '_';
    std::ofstream(name) << content;
    return name;
}

// ===========================================================================
// 1. Compile-time tests
// ===========================================================================

// --- Numeric accepts every built-in number type -----------------------------
static_assert(Numeric<short> && Numeric<unsigned short>);
static_assert(Numeric<int> && Numeric<unsigned int>);
static_assert(Numeric<long> && Numeric<unsigned long>);
static_assert(Numeric<long long> && Numeric<unsigned long long>);
static_assert(Numeric<std::int8_t> && Numeric<std::uint8_t>);
static_assert(Numeric<float> && Numeric<double> && Numeric<long double>);
static_assert(Numeric<const int>);

// --- ... and rejects things that are not feature values ---------------------
static_assert(!Numeric<bool>);
static_assert(!Numeric<char> && !Numeric<wchar_t>);
static_assert(!Numeric<char8_t> && !Numeric<char16_t> && !Numeric<char32_t>);
static_assert(!Numeric<std::string>);
static_assert(!Numeric<int*>);
static_assert(!Numeric<std::complex<double>>);

// --- RealNumber -------------------------------------------------------------
static_assert(RealNumber<float> && RealNumber<double> && RealNumber<long double>);
static_assert(!RealNumber<int> && !RealNumber<short>);

// --- default_real_t: integers compute in double, floats keep their type -----
static_assert(std::is_same_v<default_real_t<short>, double>);
static_assert(std::is_same_v<default_real_t<int>, double>);
static_assert(std::is_same_v<default_real_t<unsigned long long>, double>);
static_assert(std::is_same_v<default_real_t<float>, float>);
static_assert(std::is_same_v<default_real_t<long double>, long double>);

// --- Class templates refuse non-numeric types -------------------------------
template <class T> concept CanMakeDataset   = requires { typename BasicDataset<T>; };
template <class T> concept CanMakeMeanShift = requires { typename BasicMeanShift<T>; };
static_assert(CanMakeDataset<int> && CanMakeDataset<float>);
static_assert(!CanMakeDataset<std::string> && !CanMakeDataset<bool>);
static_assert(CanMakeMeanShift<short> && CanMakeMeanShift<double>);
static_assert(!CanMakeMeanShift<std::string> && !CanMakeMeanShift<char>);
// The compute type R must be floating point.
template <class T, class R> concept CanMakeMeanShiftR = requires { typename BasicMeanShift<T, R>; };
static_assert(CanMakeMeanShiftR<int, float>);
static_assert(!CanMakeMeanShiftR<int, int>);

// --- Constrained member functions: scaling only for floating datasets -------
template <class D> concept CanStandardize = requires(D d) { d.standardize(); };
template <class D> concept CanNormalize   = requires(D d) { d.normalize(); };
static_assert(CanStandardize<BasicDataset<double>> && CanNormalize<BasicDataset<float>>);
static_assert(!CanStandardize<BasicDataset<int>> && !CanNormalize<BasicDataset<short>>);

// --- KernelFunction ---------------------------------------------------------
struct TriangularKernel {            // a user-defined kernel, no base class
    template <RealNumber R>
    R operator()(R d, R h) const { return d < h ? R(1) - d / h : R(0); }
};
struct NotAKernel {};
struct BadReturnKernel { std::string operator()(double, double) const { return ""; } };
static_assert(KernelFunction<GaussianKernel, double>);
static_assert(KernelFunction<TriangularKernel, float>);
static_assert(!KernelFunction<NotAKernel, double>);
static_assert(!KernelFunction<BadReturnKernel, double>);
static_assert(!KernelFunction<GaussianKernel, int>);   // R must be RealNumber
template <class K> concept CanUseKernel = requires { typename BasicMeanShift<double, double, K>; };
static_assert(CanUseKernel<FlatKernel> && !CanUseKernel<NotAKernel>);

// --- Backward-compatible aliases --------------------------------------------
static_assert(std::is_same_v<Dataset,   BasicDataset<double>>);
static_assert(std::is_same_v<MeanShift, BasicMeanShift<double, double, GaussianKernel>>);
static_assert(std::is_same_v<Point,     std::vector<double>>);
static_assert(std::is_same_v<Data,      std::vector<std::vector<double>>>);
static_assert(std::is_same_v<Cluster,   BasicCluster<double>>);

// --- Result types follow R, not T -------------------------------------------
static_assert(std::is_same_v<BasicMeanShift<int>::cluster_type, BasicCluster<double>>);
static_assert(std::is_same_v<decltype(BasicMeanShift<float>{}.evaluate({})), float>);
static_assert(std::is_same_v<decltype(BasicMeanShift<short, float>{}.evaluate({})), float>);

// ===========================================================================
// 2. Run-time typed tests: MeanShift
// ===========================================================================

template <Numeric T>
void testEuclideanDistance() {
    using MS = BasicMeanShift<T>;
    using R  = typename MS::real_type;
    const BasicPoint<T> p1 = {T(0), T(0)};
    const BasicPoint<T> p2 = {T(3), T(4)};
    CHECK(near(MS::euclideanDistance(p1, p2), R(5), R(1e-6)));
    CHECK(near(MS::euclideanDistance(p2, p1), R(5), R(1e-6)));   // symmetric
    CHECK(MS::euclideanDistance(p1, p1) == R(0));
}

// Regression test for the bug templates exposed: for unsigned T, 2 - 5 wraps.
template <Numeric T>
void testDistanceNoUnsignedWrap() {
    using MS = BasicMeanShift<T>;
    using R  = typename MS::real_type;
    const BasicPoint<T> a = {T(2)};
    const BasicPoint<T> b = {T(5)};
    CHECK(near(MS::euclideanDistance(a, b), R(3), R(1e-6)));
}

// Mixed types: int data point vs. double centroid.
template <Numeric T>
void testDistanceMixedTypes() {
    using MS = BasicMeanShift<T>;
    using R  = typename MS::real_type;
    const BasicPoint<T> a = {T(1), T(1)};
    const BasicPoint<R> c = {R(1.5), R(1.5)};
    CHECK(near(MS::euclideanDistance(a, c), std::sqrt(R(0.5)), R(1e-5)));
}

template <Numeric T>
void testGaussianKernel() {
    using R = default_real_t<T>;
    BasicMeanShift<T> ms(R(2), R(1e-3), 100);
    CHECK(near(ms.gaussianKernel(R(0)), R(1), R(1e-6)));
    CHECK(near(ms.gaussianKernel(R(2)), std::exp(R(-0.5)), R(1e-6)));
    CHECK(near(ms.kernelWeight(R(2)), ms.gaussianKernel(R(2)), R(1e-6)));  // default policy
}

template <Numeric T>
void testShiftPoint() {
    using R = default_real_t<T>;
    BasicMeanShift<T> ms(R(2), R(1e-3), 100);
    const BasicData<T> data = {{T(0), T(0)}, {T(2), T(2)}};
    const BasicPoint<R> mid = {R(1), R(1)};
    const BasicPoint<R> shifted = ms.shiftPoint(mid, data);
    CHECK(shifted.size() == 2);
    CHECK(near(shifted[0], R(1), R(1e-5)));
    CHECK(near(shifted[1], R(1), R(1e-5)));
}

template <Numeric T>
void testMeanShiftClustering() {
    using R = default_real_t<T>;
    BasicMeanShift<T> ms(R(20), R(1e-3), 300);
    ms.fit(twoBlobs<T>());
    const auto& clusters = ms.getClusters();
    CHECK(clusters.size() == 2);
    if (clusters.size() != 2) return;
    // Centroids are real-valued means even for integer input: (1,1) and (101,101).
    CHECK(near(clusters[0].centroid[0], R(1),   R(0.05)));
    CHECK(near(clusters[0].centroid[1], R(1),   R(0.05)));
    CHECK(near(clusters[1].centroid[0], R(101), R(0.05)));
    CHECK(near(clusters[1].centroid[1], R(101), R(0.05)));
    CHECK(clusters[0].pointIndices.size() == 3);
    CHECK(clusters[1].pointIndices.size() == 3);
}

template <Numeric T>
void testGetLabels() {
    using R = default_real_t<T>;
    BasicMeanShift<T> ms(R(20), R(1e-3), 300);
    ms.fit(twoBlobs<T>());
    const auto& labels = ms.getLabels();
    CHECK(labels.size() == 6);
    if (labels.size() != 6) return;
    CHECK(labels[0] == labels[1] && labels[1] == labels[2]);
    CHECK(labels[3] == labels[4] && labels[4] == labels[5]);
    CHECK(labels[0] != labels[3]);
}

template <Numeric T>
void testEstimateBandwidth() {
    BasicMeanShift<T> ms;                       // bandwidth <= 0 -> estimate
    using R = typename BasicMeanShift<T>::real_type;
    CHECK(ms.getBandwidth() <= R(0));
    ms.fit(twoBlobs<T>());
    CHECK(ms.getBandwidth() > R(0));
    CHECK(ms.getClusters().size() == 2);
}

template <Numeric T>
void testEvaluate() {
    using R = default_real_t<T>;
    BasicMeanShift<T> ms(R(20), R(1e-3), 300);
    const auto data = twoBlobs<T>();
    ms.fit(data);
    const R score = ms.evaluate(data);
    // Each blob: distances sqrt2, 0, sqrt2 to its centre -> 4*sqrt2/6 ~ 0.943.
    CHECK(near(score, R(4) * std::sqrt(R(2)) / R(6), R(0.05)));
}

template <Numeric T>
void testEmptyData() {
    BasicMeanShift<T> ms;
    const BasicData<T> empty;
    ms.fit(empty);
    CHECK(ms.getClusters().empty());
    CHECK(ms.getLabels().empty());
    CHECK(ms.evaluate(empty) == 0);
}

// Calling fit() twice must not keep clusters from the first call.
template <Numeric T>
void testRefitResetsState() {
    using R = default_real_t<T>;
    BasicMeanShift<T> ms(R(20), R(1e-3), 300);
    ms.fit(twoBlobs<T>());
    ms.fit(BasicData<T>{{T(5), T(5)}, {T(6), T(6)}});
    CHECK(ms.getClusters().size() == 1);
    CHECK(ms.getLabels().size() == 2);
    ms.fit(BasicData<T>{});
    CHECK(ms.getClusters().empty());
}

// Every kernel policy should find the same two blobs.
template <Numeric T>
void testKernelPolicies() {
    using R = default_real_t<T>;
    BasicMeanShift<T, R, FlatKernel>         flat(R(20), R(1e-3), 300);
    BasicMeanShift<T, R, EpanechnikovKernel> epan(R(20), R(1e-3), 300);
    BasicMeanShift<T, R, TriangularKernel>   tri (R(20), R(1e-3), 300);
    flat.fit(twoBlobs<T>());
    epan.fit(twoBlobs<T>());
    tri.fit(twoBlobs<T>());
    CHECK(flat.getClusters().size() == 2);
    CHECK(epan.getClusters().size() == 2);
    CHECK(tri.getClusters().size() == 2);
    CHECK(flat.kernelWeight(R(25)) == R(0));   // outside bandwidth
    CHECK(flat.kernelWeight(R(5))  == R(1));
}

// Overriding the compute type: int input, float arithmetic.
template <Numeric T>
void testCustomComputeType() {
    BasicMeanShift<T, float> ms(20.0f, 1e-3f, 300);
    ms.fit(twoBlobs<T>());
    CHECK(ms.getClusters().size() == 2);
    const float score = ms.evaluate(twoBlobs<T>());
    CHECK(score > 0.0f && score < 2.0f);
}

// ===========================================================================
// 3. Run-time typed tests: Dataset
// ===========================================================================

template <Numeric T>
void testLoadFromCSV() {
    const std::string file = writeTempCSV<T>("load", "Feature1,Feature2\n1,2\n3,4\n");
    BasicDataset<T> dataset;
    const bool loaded = dataset.loadFromCSV(file, true);
    std::remove(file.c_str());

    const auto& data = dataset.getData();
    CHECK(loaded);
    CHECK(data.size() == 2);
    if (data.size() == 2) {
        CHECK(data[0] == (BasicPoint<T>{T(1), T(2)}));
        CHECK(data[1] == (BasicPoint<T>{T(3), T(4)}));
    }
    CHECK(!dataset.loadFromCSV("non_existent_file.csv", true));
}

// Name columns are skipped; spaces, '+' signs and Windows line endings are fine.
template <Numeric T>
void testLoadFromCSVMessyInput() {
    const std::string file = writeTempCSV<T>(
        "messy", "Name,A,B\r\nJohn, 10 ,+20\r\n\r\nSarah,30,40\r\n");
    BasicDataset<T> dataset;
    CHECK(dataset.loadFromCSV(file, true));
    std::remove(file.c_str());
    const auto& data = dataset.getData();
    CHECK(data.size() == 2);
    if (data.size() == 2) {
        CHECK(data[0] == (BasicPoint<T>{T(10), T(20)}));
        CHECK(data[1] == (BasicPoint<T>{T(30), T(40)}));
    }
}

template <Numeric T>
void testLoadFromCSVNoHeader() {
    const std::string file = writeTempCSV<T>("nohdr", "5,6\n7,8\n");
    BasicDataset<T> dataset;
    CHECK(dataset.loadFromCSV(file, false));
    std::remove(file.c_str());
    CHECK(dataset.getData().size() == 2);
}

// parseNumber is where the type actually matters.
template <Numeric T>
void testParseNumber() {
    T v{};
    CHECK(parseNumber<T>("42", v) && v == T(42));
    CHECK(parseNumber<T>("  7\r", v) && v == T(7));
    CHECK(!parseNumber<T>("", v));
    CHECK(!parseNumber<T>("abc", v));
    CHECK(!parseNumber<T>("12abc", v));          // partial numbers are rejected

    if constexpr (std::integral<T>) {
        CHECK(!parseNumber<T>("2.5", v));        // no silent truncation
        // One past the type's maximum must be rejected, not wrapped.
        const std::string tooBig = std::to_string(
            static_cast<unsigned long long>(std::numeric_limits<T>::max())) + "0";
        CHECK(!parseNumber<T>(tooBig, v));
    } else {
        CHECK(parseNumber<T>("2.5", v) && v == T(2.5));
        CHECK(parseNumber<T>("-1e2", v) && v == T(-100));
        CHECK(!parseNumber<T>("nan", v));
        CHECK(!parseNumber<T>("inf", v));
    }

    if constexpr (std::unsigned_integral<T>) {
        CHECK(!parseNumber<T>("-1", v));         // negative into unsigned
    } else {
        CHECK(parseNumber<T>("-3", v) && v == T(-3));
    }
}

template <Numeric T>
void testStandardize() {   // T is a RealNumber here
    BasicDataset<T> dataset(BasicData<T>{{T(1), T(10)}, {T(3), T(30)}});
    dataset.standardize();
    const auto& d = dataset.getData();
    CHECK(d.size() == 2);
    CHECK(near(d[0][0], T(-1), T(1e-5)) && near(d[1][0], T(1), T(1e-5)));
    CHECK(near(d[0][1], T(-1), T(1e-5)) && near(d[1][1], T(1), T(1e-5)));
}

template <Numeric T>
void testNormalize() {     // T is a RealNumber here
    BasicDataset<T> dataset(BasicData<T>{{T(10), T(50)}, {T(20), T(100)}, {T(30), T(150)}});
    dataset.normalize();
    const auto& d = dataset.getData();
    CHECK(d.size() == 3);
    for (std::size_t col = 0; col < 2; ++col) {
        CHECK(near(d[0][col], T(0),   T(1e-5)));
        CHECK(near(d[1][col], T(0.5), T(1e-5)));
        CHECK(near(d[2][col], T(1),   T(1e-5)));
    }
}

// A constant column must not divide by zero.
template <Numeric T>
void testScalingConstantColumn() {
    BasicDataset<T> a(BasicData<T>{{T(4), T(1)}, {T(4), T(2)}});
    BasicDataset<T> b = a;
    a.standardize();
    b.normalize();
    CHECK(std::isfinite(a.getData()[0][0]) && a.getData()[0][0] == T(4));
    CHECK(std::isfinite(b.getData()[0][0]) && b.getData()[0][0] == T(4));
}

// Integer data -> cast<double>() -> scale.  This is the intended path, since
// standardize() is deliberately not available on BasicDataset<int>.
template <Numeric T>
void testCastThenScale() {
    BasicDataset<T> raw(BasicData<T>{{T(1), T(10)}, {T(3), T(30)}});
    BasicDataset<double> real = raw.template cast<double>();
    CHECK(real.getData()[1][1] == 30.0);
    real.standardize();
    CHECK(near(real.getData()[0][0], -1.0, 1e-9));
    CHECK(near(real.getData()[1][1],  1.0, 1e-9));
}

// End-to-end: CSV of type T -> MeanShift<T>.
template <Numeric T>
void testPipelineFromCSV() {
    const std::string file = writeTempCSV<T>(
        "pipe", "x,y\n0,0\n1,1\n2,2\n100,100\n101,101\n102,102\n");
    BasicDataset<T> dataset;
    CHECK(dataset.loadFromCSV(file));
    std::remove(file.c_str());
    BasicMeanShift<T> ms(default_real_t<T>(20));
    ms.fit(dataset.getData());
    CHECK(ms.getClusters().size() == 2);
}

// ===========================================================================
// 4. The original (non-template) API still works exactly as before
// ===========================================================================

void testLegacyAPI() {
    const int before = g_failures;
    MeanShift ms(2.0, 1e-3, 100);
    Data data = {
        {0.0, 0.0}, {0.1, 0.1}, {0.2, 0.2},
        {10.0, 10.0}, {10.1, 10.1}, {10.2, 10.2}
    };
    ms.fit(data);
    CHECK(ms.getClusters().size() == 2);
    CHECK(MeanShift::euclideanDistance(Point{0.0, 0.0}, Point{3.0, 4.0}) == 5.0);
    const Cluster& c = ms.getClusters().front();
    CHECK(c.pointIndices.size() == 3);
    const double score = ms.evaluate(data);
    CHECK(score > 0.0 && score <= 0.5);
    std::cout << "  " << (g_failures == before ? "ok    " : "FAIL  ") << "testLegacyAPI\n";
}

// ===========================================================================

int main() {
    std::cout << "MeanShift tests\n";
    forEachType<NUMERIC_TYPES>("testEuclideanDistance",      []<class T>() { testEuclideanDistance<T>(); });
    forEachType<NUMERIC_TYPES>("testDistanceNoUnsignedWrap", []<class T>() { testDistanceNoUnsignedWrap<T>(); });
    forEachType<NUMERIC_TYPES>("testDistanceMixedTypes",     []<class T>() { testDistanceMixedTypes<T>(); });
    forEachType<NUMERIC_TYPES>("testGaussianKernel",         []<class T>() { testGaussianKernel<T>(); });
    forEachType<NUMERIC_TYPES>("testShiftPoint",             []<class T>() { testShiftPoint<T>(); });
    forEachType<NUMERIC_TYPES>("testMeanShiftClustering",    []<class T>() { testMeanShiftClustering<T>(); });
    forEachType<NUMERIC_TYPES>("testGetLabels",              []<class T>() { testGetLabels<T>(); });
    forEachType<NUMERIC_TYPES>("testEstimateBandwidth",      []<class T>() { testEstimateBandwidth<T>(); });
    forEachType<NUMERIC_TYPES>("testEvaluate",               []<class T>() { testEvaluate<T>(); });
    forEachType<NUMERIC_TYPES>("testEmptyData",              []<class T>() { testEmptyData<T>(); });
    forEachType<NUMERIC_TYPES>("testRefitResetsState",       []<class T>() { testRefitResetsState<T>(); });
    forEachType<NUMERIC_TYPES>("testKernelPolicies",         []<class T>() { testKernelPolicies<T>(); });
    forEachType<NUMERIC_TYPES>("testCustomComputeType",      []<class T>() { testCustomComputeType<T>(); });

    std::cout << "Dataset tests\n";
    forEachType<NUMERIC_TYPES>("testLoadFromCSV",            []<class T>() { testLoadFromCSV<T>(); });
    forEachType<NUMERIC_TYPES>("testLoadFromCSVMessyInput",  []<class T>() { testLoadFromCSVMessyInput<T>(); });
    forEachType<NUMERIC_TYPES>("testLoadFromCSVNoHeader",    []<class T>() { testLoadFromCSVNoHeader<T>(); });
    forEachType<NUMERIC_TYPES>("testParseNumber",            []<class T>() { testParseNumber<T>(); });
    forEachType<REAL_TYPES>   ("testStandardize",            []<class T>() { testStandardize<T>(); });
    forEachType<REAL_TYPES>   ("testNormalize",              []<class T>() { testNormalize<T>(); });
    forEachType<REAL_TYPES>   ("testScalingConstantColumn",  []<class T>() { testScalingConstantColumn<T>(); });
    forEachType<NUMERIC_TYPES>("testCastThenScale",          []<class T>() { testCastThenScale<T>(); });
    forEachType<NUMERIC_TYPES>("testPipelineFromCSV",        []<class T>() { testPipelineFromCSV<T>(); });

    std::cout << "Backward compatibility\n";
    testLegacyAPI();

    std::cout << '\n' << g_checks << " checks, " << g_failures << " failures\n";
    if (g_failures == 0) std::cout << "All tests passed!\n";
    return g_failures == 0 ? 0 : 1;
}
