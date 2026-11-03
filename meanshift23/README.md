# Mean Shift Clustering Library (C++)

**Group 21**· Machine Learning 13: **Mean Shift Clustering in C++**

## 1. Overview

This is a reusable C++ library that implements **Mean Shift clustering**. Clustering means grouping similar data points together without being told the groups in advance. Unlike k-means, Mean Shift does not need you to say how many clusters you want. It finds that number from the data.

You give the library a set of points (for example, rows from a CSV file). It gives back the cluster centres, which points belong to which cluster, and a simple score for how tight the clusters are.

## 2. Objectives

- Implement Mean Shift clustering using C++ classes.
- Keep a clear public interface (headers in `include/`, code in `src/`).
- Load CSV data and scale it (Z-score or Min-Max).
- Test the library with automated tests, including bad input.
- Show how to use it with small example programs.
- Build from a clean machine with CMake.
- Work as a team using Git and GitHub.

## 3. Main features

| Feature | Function |
| :--- | :--- |
| Load a CSV file (with or without a header) | `Dataset::loadFromCSV` |
| Z-score scaling (mean 0, standard deviation 1) | `Dataset::standardize` |
| Min-Max scaling (values between 0 and 1) | `Dataset::normalize` |
| Distance between two points | `MeanShift::euclideanDistance` |
| Gaussian kernel weight | `MeanShift::gaussianKernel` |
| Automatic bandwidth estimate | `MeanShift::estimateBandwidth` |
| One mean shift step | `MeanShift::shiftPoint` |
| Run the whole clustering | `MeanShift::fit` |
| Read the clusters and labels | `getClusters`, `getLabels` |
| Cluster quality score | `MeanShift::evaluate` |
| Any numeric type (`short`, `int`, `long`, `unsigned`, `int8_t`, `float`, `double`, `long double`, ...) | `BasicDataset<T>`, `BasicMeanShift<T, R, K>` |

## 4. Project structure

```
GROUP-21-PROJECT/
├── README.md
├── CMakeLists.txt        builds the library, examples and tests
├── LICENSE
├── .gitignore
├── include/meanshift/    public headers: Dataset.hpp, MeanShift.hpp
├── src/                  explicit instantiations for int, float, double
├── tests/                unit tests (test_meanshift.cpp)
├── examples/             example1.cpp, example2.cpp, example3.cpp
├── data/input/           sample CSV files
├── data/output/          clustering results are written here
└── reports/              weekly reports, individual reports, final report
```

## 5. Requirements

- Git
- CMake 3.20 or newer
- A C++ compiler that supports C++23 (we checked with GCC 13)

There are no other dependencies. The library uses only the C++ standard library.

## 6. How to build

```
git clone https://github.com/josephmusiitwa/GROUP-21-PROJECT.git
cd GROUP-21-PROJECT
cmake -S . -B build
cmake --build build
```

This creates the library `libmeanshift.a` (or `meanshift.lib` on Windows), the three example programs and the test program inside the `build` folder.

## 7. How to use the library

```cpp
#include "meanshift/Dataset.hpp"
#include "meanshift/MeanShift.hpp"
#include <iostream>

int main() {
    meanshift::Dataset dataset;
    if (!dataset.loadFromCSV("data/input/my_data.csv")) {   // false if the file cannot be opened
        return 1;
    }
    // dataset.standardize();   // optional: Z-score scaling
    // dataset.normalize();     // optional: Min-Max scaling

    meanshift::MeanShift ms(2.0, 1e-3, 300);   // bandwidth, epsilon, max iterations
    ms.fit(dataset.getData());

    for (const auto& cluster : ms.getClusters()) {
        std::cout << "Cluster with " << cluster.pointIndices.size() << " points\n";
    }
    std::cout << "Score: " << ms.evaluate(dataset.getData()) << "\n";
}
```

**What the settings mean:**

- **bandwidth**: how far a point "looks" for neighbours. Small values give many small clusters, large values give a few big ones. If you pass `-1` (the default), the library estimates it.
- **epsilon**: a point stops moving when its step is smaller than this.
- **max iterations**: the most steps a point may take.

**After `fit()`:**

- `getClusters()` gives a list of `Cluster`, each with a `centroid` (the centre) and `pointIndices` (which rows belong to it).
- `getLabels()` gives one cluster number per input row.
- `evaluate()` gives the average distance from each point to its centre. Lower means tighter clusters.

### Using other numeric types

```cpp
meanshift::BasicDataset<int> raw;                 // integer CSV values
raw.loadFromCSV("data/input/students.csv");
meanshift::BasicMeanShift<int> ms(5.0);           // int data, double maths
ms.fit(raw.getData());                            // centroids are BasicPoint<double>

auto scaled = raw.cast<double>();                 // standardize() needs a floating type
scaled.standardize();

meanshift::BasicMeanShift<float, float, meanshift::FlatKernel> flat(0.5f);
flat.fit(scaled.cast<float>().getData());
```

`BasicDataset<int>::standardize()` and `BasicMeanShift<std::string>` are compile errors, and the error message names the concept that failed.

### Data format

CSV files with one point per row and numbers separated by commas. A header row is skipped by default (pass `false` as the second argument if there is no header). Empty lines and values that are not numbers (such as names) are skipped, so `students.csv` can keep a name column.

```
Feature1,Feature2
1.0,1.2
1.5,1.4
8.0,8.1
```

## 8. Examples

| Program | What it does |
| :--- | :--- |
| `example1` | Loads a CSV, clusters it, prints the clusters and writes `Cluster_ID` results to a CSV. |
| `example2` | Clusters 9 points written in the code (no file needed). |
| `example3` | Same as `example1`, but asks you to type the input and output file paths. |

Run them from the project folder:

```
./build/examples/example2
./build/examples/example1 data/input/my_data.csv data/output/results.csv
./build/examples/example3
```

On Windows the programs may be inside `build/examples/Debug/` and end in `.exe`.

Output of `example2`:

```
Number of clusters found: 3
Cluster 0 centroid: (1.50006, 1.35186) with 3 points.
Cluster 1 centroid: (8.47972, 8.33061) with 3 points.
Cluster 2 centroid: (1.52013, 8.30793) with 3 points.
Clustering evaluation score (cohesion): 0.622962
```

Sample data in `data/input/`: `my_data.csv`, `biology_data.csv`, `finance.csv`, `students.csv`, `text_data.csv`.

## 9. How to run the tests

```
cd build
ctest --output-on-failure
```

`tests/test_meanshift.cpp` has two layers:

- **Compile-time tests** (`static_assert`) check that the concepts accept every numeric type and reject `bool`, `char`, `std::string` and pointers. They also check that `standardize()` is unavailable for integer datasets.
- **Typed tests**: each test is a function template run for 13 types (`short` ... `long double`). Together they cover distance (including the unsigned wrap-around case), kernels, shifting, clustering, labels, bandwidth estimation, evaluation, empty data, CSV parsing per type, scaling and the old `double` API. In total there are 865 checks.

Tests that need files create and delete their own temporary CSV, so they can be run from any folder.

## 10. Limitations

- Each step compares a point with every other point, so it gets slow on very large data.
- Kernels are chosen at compile time (template parameter), not at run time.
- The automatic bandwidth is a simple guess (half the average distance between points, using at most the first 500 points).
- `evaluate()` measures only how tight clusters are, not how well separated they are.
- The library does not check that every row has the same number of values. Rows of different length can cause errors.
- The examples write the header `Feature1,Feature2,Cluster_ID`, so the header is wrong if your file has a different number of features.
- `example3` does not accept file paths with spaces.
- The project needs a compiler with C++23 support (concepts and `std::from_chars` for floating-point types are used).
- `long long` values above 2^53 lose precision, because the default compute type for integers is `double`.

## 11. Contributions by group members

| Member | Main contribution |
| :--- | :--- |
| Musiitwa Joseph | Repository set-up, `CMakeLists.txt`, `.gitignore`, test set-up and first tests, merging pull requests, coordinating integration and testing |
| Muhindo Nazarene | First stage of `fit()` (repeated shifting and stop flag), `MeanShift.hpp` declarations, a test, `finance.csv` |
| Mark Jordan | Repository skeleton, `example2.cpp` (with Jonathan), `my_data.csv`, class diagrams, reviewing and merging pull requests, Week 4 report |
| Atuzarirwe Happy | `standardize()` scaling step, `normalize()`, `testLoadFromCSV`, `testStandardize`, `students.csv` |
| Kizza Johnson | `Dataset` class, `loadFromCSV`, `getData`, mean and standard deviation in `standardize`, `testEstimateBandwidth`, `testGetLabels`, `biology_data.csv` |
| Atugonza Simon | `euclideanDistance`, `example1.cpp`, CSV data files including `text_data.csv` |
| Muzaki Desire | Cluster formation stage of `fit()`, `testNormalize`, `example3.cpp` |
| Clive | `getClusters`, `getLabels`, `evaluate`, `testEvaluate`, `testEmptyData` |
| Sserunkuma Jonathan | `estimateBandwidth`, `shiftPoint`, `example2.cpp` (with Mark) |

More detail is in `reports/Individual reports/`.

## 12. How we collaborated

We were 9 members who had never worked on a shared code project before. Each person took one function or file, worked on their own branch, and opened a pull request on GitHub. A different member reviewed and merged it. We wrote a short report every week in `reports/`. The Git history shows every member's commits.

## AI use

AI tools (Gemini, ChatGPT and Claude) were used for learning and checking, and this README was drafted with Claude and checked by the group. The full declaration, week by week, is in `reports/final-report.md` (Section 12).