# Final Report

**Group 21 · Mean Shift Clustering Library in C++ (`meanshift`)**
Topic: Machine Learning 13 — Design and Implementation of the Mean Shift Clustering Algorithm in C++
Repository: https://github.com/josephmusiitwa/GROUP-21-PROJECT
Submission date: 5 November 2026

---

## 1. The problem

Clustering means putting similar data points into the same group without being told the groups in advance. Many clustering methods (for example k-means) need you to say how many clusters you want before you start. Mean Shift does not. It finds the number of clusters from the data itself.

Our task was to build this algorithm as a **reusable C++ library**, not a one-off program, so that any C++ programmer can add it to their project, give it numbers, and get clusters back.

## 2. Objectives

1. Implement Mean Shift clustering in standard C++ using classes.
2. Provide a clear public interface (headers in `include/`, code in `src/`).
3. Load data from CSV files and scale it so that big-number columns do not dominate small-number columns.
4. Test the library with automated tests, including bad inputs.
5. Show how to use it with small example programs.
6. Build a project that compiles from a clean machine using CMake.
7. Work as a team of 9 using Git and GitHub (branches and pull requests).

## 3. How Mean Shift works (plain language)

1. Every data point is treated as a "walker".
2. Each walker looks at the points around it. Closer points count more, farther points count less. The weights come from a Gaussian (bell-curve) kernel, and the **bandwidth** controls how far "around" reaches.
3. The walker moves to the weighted average position of those points. This move is the **mean shift step**.
4. Steps 2–3 repeat. A walker stops when its move is smaller than a small number (`epsilon`), or after `maxIterations` rounds.
5. Walkers that end up in nearly the same place (closer than `bandwidth / 2`) are put in one cluster. That place is the cluster **centroid**.
6. Every original point gets the number of the cluster its walker ended in.

Small bandwidth gives many small clusters. Large bandwidth gives a few big ones. If the user does not give a bandwidth, the library estimates one from the data.

## 4. Library design

### 4.1 Structure

```
GROUP-21-PROJECT/
├── CMakeLists.txt            builds the static library, examples and tests
├── include/meanshift/
│   ├── Dataset.hpp           Dataset class: load and scale data
│   └── MeanShift.hpp         MeanShift class and Cluster struct
├── src/
│   ├── Dataset.cpp
│   └── meanshift.cpp
├── tests/test_meanshift.cpp  10 unit tests, run with ctest
├── examples/                 example1, example2, example3
├── data/input/               5 sample CSV files
├── data/output/              results written here
└── reports/                  weekly reports, individual reports, this report
```

All code is inside `namespace meanshift`. Two type names are used everywhere: `Point` (a `vector<double>`) and `Data` (a `vector<Point>`).

### 4.2 The classes (the lecturer's five questions)

**Class 1: `Dataset`**

| Question | Answer |
| :--- | :--- |
| What is the class? | `meanshift::Dataset` (`Dataset.hpp`, `Dataset.cpp`). |
| Why do we need it? | To keep reading and preparing data in one place, separate from the clustering. |
| What data does it hold? | One member: `data_`, all the points read from the CSV file. |
| What functions operate on it? | `loadFromCSV()` fills it. `standardize()` and `normalize()` rescale it. `getData()` lets others read it. |
| Private or public, and how is it used? | `data_` is private so nothing outside the class can change the points by accident. The functions are public because they are how users work with the class. `getData()` returns a `const` reference, so callers can read but not change the data. Usage: create a `Dataset`, call `loadFromCSV()`, optionally scale, then pass `getData()` to `MeanShift`. |

**Class 2: `MeanShift`**

| Question | Answer |
| :--- | :--- |
| What is the class? | `meanshift::MeanShift` (`MeanShift.hpp`, `meanshift.cpp`). |
| Why do we need it? | To keep the algorithm and its settings together. The user sets the settings once, calls `fit()`, then reads the results. |
| What data does it hold? | Settings: `bandwidth_`, `epsilon_`, `maxIterations_`. Results: `clusters_`, `labels_`. |
| What functions operate on it? | The constructor sets the settings. `estimateBandwidth()` picks a bandwidth. `gaussianKernel()` weights a neighbour. `shiftPoint()` does one mean shift step. `fit()` runs the whole algorithm. `getClusters()`, `getLabels()` and `evaluate()` read the results. |
| Private or public, and how is it used? | Settings and results are private, so the results always match the settings that produced them. The small maths functions are public so they can be tested one by one. `euclideanDistance()` is `static` because it needs only two points and no object data. Usage: construct, `fit(data)`, read results. `MeanShift` needs only the `Data` type, not `Dataset`, so it can cluster data from any source. |

**Struct: `Cluster`** — a plain struct with two public members, `centroid` (a `Point`) and `pointIndices` (which rows belong to it). It is a simple bundle of result data with no behaviour, so a struct is enough. Callers get it through a `const` reference.

### 4.3 Encapsulation, inheritance and polymorphism

- **Encapsulation:** all data members of `Dataset` and `MeanShift` are private. Outside code can only use the public functions. Read access returns `const` references.
- **Inheritance:** our own classes do not inherit from each other. `Dataset` and `MeanShift` do different jobs, and one is not a kind of the other, so we use composition (the program uses both side by side). Inheritance does appear in the standard library parts we rely on: `ifstream` and `stringstream` derive from `istream`, and `std::stod` throws `invalid_argument` and `out_of_range`, which derive from `std::exception`.
- **Polymorphism:** we use default arguments (`MeanShift ms;` or `MeanShift ms(2.0, 1e-3, 300);`, and `loadFromCSV(file)` or `loadFromCSV(file, false)`), templates (`vector<Point>`, `vector<Cluster>`), and base-class references from the standard library (`catch (const exception&)`, `istream&` in `getline`). We did not write our own `virtual` functions, because there are no derived classes.

Hand-drawn diagrams (photographed, in `reports/images/`):

![Dataset and MeanShift class diagrams](images/encapsulation_classes.jpeg)
![Namespace meanshift](images/encapsulation_namespace.jpeg)
![Standard library inheritance](images/inheritance_stdlib.jpeg)
![Polymorphism overview](images/polymorphism_overview.jpeg)

### 4.4 Important implementation decisions

| Decision | Reason |
| :--- | :--- |
| Separate `Dataset` and `MeanShift`. | Loading files and clustering are different jobs. This made each one easier to write, test and give to different people. |
| Gaussian kernel. | Smooth weights, no sharp cut-off, one parameter (bandwidth). |
| Bandwidth guessed as half the average distance between points, using at most the first 500 points. | Avoids comparing every pair on large data, which would be very slow. |
| Per-point stop flag (`stopMoving`). | Once a point has settled it is no longer recomputed. This saves a lot of time because every step compares a point with all others. |
| `fit()` works on a copy of the data. | The caller's data is `const` and is never changed. |
| Merge converged points that are within `bandwidth / 2`. | Floating-point numbers are almost never exactly equal, so we compare with a distance, not with `==`. |
| `evaluate()` returns the mean distance from each point to its centroid. | It is quick to compute and easy to explain. Lower means tighter clusters. |
| `loadFromCSV()` skips the header, empty lines and values that are not numbers. | Lets us use files with name or category columns (for example `students.csv`) without extra code. |
| Static library `libmeanshift.a` with `install()` rules. | Other CMake projects can use it as `MeanShift::meanshift`. |

## 5. Main functionality

| Feature | Where |
| :--- | :--- |
| Load CSV data (with or without header) | `Dataset::loadFromCSV` |
| Z-score scaling (mean 0, standard deviation 1) | `Dataset::standardize` |
| Min-Max scaling (0 to 1) | `Dataset::normalize` |
| Distance between two points | `MeanShift::euclideanDistance` |
| Gaussian kernel weight | `MeanShift::gaussianKernel` |
| Automatic bandwidth | `MeanShift::estimateBandwidth` |
| One mean shift step | `MeanShift::shiftPoint` |
| Full clustering | `MeanShift::fit` |
| Read clusters and labels | `getClusters`, `getLabels` |
| Quality score | `MeanShift::evaluate` |

Minimal use:

```cpp
#include "meanshift/Dataset.hpp"
#include "meanshift/MeanShift.hpp"

meanshift::Dataset ds;
ds.loadFromCSV("data/input/my_data.csv");   // returns false if the file cannot be opened
meanshift::MeanShift ms(2.0, 1e-3, 300);    // bandwidth, epsilon, max iterations
ms.fit(ds.getData());
for (const auto& c : ms.getClusters()) { /* c.centroid, c.pointIndices */ }
```

## 6. Demonstration (real output)

Output below was produced by building the repository and running the programs.

**example2** (9 points written in the code, three groups):
```text
Number of clusters found: 3
Cluster 0 centroid: (1.50006, 1.35186) with 3 points.
Cluster 1 centroid: (8.47972, 8.33061) with 3 points.
Cluster 2 centroid: (1.52013, 8.30793) with 3 points.
Clustering evaluation score (cohesion): 0.622962
```

**example1** on each sample file (`./build/examples/example1 data/input/<file>.csv data/output/results.csv`):

| File | Clusters | Points per cluster | Score |
| :--- | :---: | :--- | :---: |
| `my_data.csv` | 3 | 10, 10, 10 | 0.259014 |
| `biology_data.csv` | 3 | 10, 10, 10 | 0.259014 |
| `finance.csv` | 3 | 10, 10, 10 | 0.259014 |
| `students.csv` (Math, Science) | 4 | 4, 3, 2, 3 | 1.97891 |
| `text_data.csv` (text columns skipped) | 3 | 3, 3, 3 | 0.101065 |

`example3` does the same as `example1`, but asks the user to type the input and output file paths.

Note: `my_data.csv`, `biology_data.csv` and `finance.csv` give identical results because they contain the same 30 points. They are separate files to show that the library works on different file names, not different data.

## 7. Testing

### 7.1 Approach
We started with 2 tests in Week 3 and grew to 10 in Week 4. Tests are in `tests/test_meanshift.cpp` and run through CTest. Tests that need files create a small temporary CSV, use it, and delete it, so they pass from any folder.

### 7.2 Unit tests (all 10 pass)

| # | Test | Type | What it checks |
| :---: | :--- | :--- | :--- |
| 1 | `testEuclideanDistance` | Normal | Distance (0,0) to (3,4) is 5 |
| 2 | `testMeanShiftClustering` | Normal | 6 points in 2 far-apart groups give 2 clusters |
| 3 | `testShiftPoint` | Normal | A point halfway between two points stays in the middle |
| 4 | `testEstimateBandwidth` | Normal | Automatic bandwidth still finds 2 clusters |
| 5 | `testGetLabels` | Normal | 6 labels; same group has the same label, different groups differ |
| 6 | `testEvaluate` | Normal | Score is above 0 and at most 0.5 on tight clusters |
| 7 | `testEmptyData` | Edge case | Empty data gives no clusters and score 0 |
| 8 | `testLoadFromCSV` | Normal and invalid | A valid file loads; a missing file returns `false` |
| 9 | `testStandardize` | Normal | Values become -1 and +1 |
| 10 | `testNormalize` | Normal | Values become 0, 0.5 and 1 |

```text
1/1 Test #1: MeanShiftTests ...................   Passed
100% tests passed out of 1
```

### 7.3 Testing by students outside the group

As the project instructions require, we gave the project to three students from other groups. Each one followed only the steps in the README on their own computer, with no help from us.

| Tester | Computer | Built? | Tests passed? | Problems they found | What we did |
| :--- | :--- | :---: | :---: | :--- | :--- |
| John | Windows | Yes | No problems reported | None | None needed |
| Chris | Windows | Yes | No problems reported | None | None needed |
| Peter | Linux | Yes | No problems reported | None | None needed |

All three testers built the project and ran the tests by following the README, and none reported any problem. Together they covered both Windows and Linux.

Separately from these testers, we found that building on a case-sensitive system (Linux) can fail if `CMakeLists.txt` names `src/MeanShift.cpp` while the file is `src/meanshift.cpp`. Our own Windows computers hid this. The file name in `CMakeLists.txt` is now `src/meanshift.cpp` (see Section 8).

## 8. Challenges and how we solved them

| Challenge | What we did |
| :--- | :--- |
| Nobody knew machine learning at the start. Words like "kernel" and "bandwidth" were new. | Week 1 went on learning. We used AI tools as tutors, then wrote the algorithm in plain steps (Section 3) so everyone could follow it. |
| Nobody had used Git as a team. | Members worked on their own branches and opened pull requests (numbered up to #46). The repository has 130 commits. |
| Slow network sometimes delayed GitHub's compare and pull request pages. | Work went on normally once everyone was used to the steps. |
| Hard to see how our classes use inheritance and polymorphism. | We found where they appear (standard library, default arguments, templates) and drew diagrams. |
| `CMakeLists.txt` named `src/MeanShift.cpp` but the file is `src/meanshift.cpp`. This works on Windows and fails on Linux. | Found when building on a Linux system. Fixed by using `src/meanshift.cpp` in `CMakeLists.txt`. |
| Tests that read `data/input/` failed when run from another folder. | Those tests now create temporary files. |
| Running every point against every other point is slow. | Added the per-point stop flag and the 500-point limit on bandwidth estimation. |

## 9. Individual contributions

Commit counts are from `git log` (each person's different Git names and emails added together). Detailed write-ups are in `reports/Individual reports/`.

| Member | Commits | Main contribution |
| :--- | :---: | :--- |
| Musiitwa Joseph | 39 | Repository set-up, root `CMakeLists.txt` (library, install rules, examples, tests), `.gitignore`, test file set-up and first tests, merging pull requests, aligning code style, coordinating integration and testing |
| Muhindo Nazarene | 18 | First stage of `fit()` (bandwidth trigger, repeated shifting, `stopMoving`), `MeanShift.hpp` declarations, a test function, `finance.csv` |
| Mark Jordan | 21 | Initial repository skeleton, `example2.cpp` (with Jonathan), `my_data.csv`, OOP diagrams, reviewing and merging pull requests, Week 4 report |
| Atuzarirwe Happy | 12 | `standardize()` scaling step, `normalize()`, `testLoadFromCSV`, `testStandardize`, `students.csv` |
| Kizza Johnson | 11 | `Dataset` class, `loadFromCSV`, `getData`, mean and standard deviation in `standardize`, `testEstimateBandwidth`, `testGetLabels`, `biology_data.csv` |
| Atugonza Simon | 10 | `euclideanDistance`, `example1.cpp`, CSV data files including `text_data.csv` |
| Muzaki Desire | 9 | Cluster formation stage of `fit()` (merging within `bandwidth/2`), `testNormalize`, interactive `example3.cpp` |
| Clive | 5 | `getClusters`, `getLabels`, `evaluate`, `testEvaluate`, `testEmptyData` |
| Sserunkuma Jonathan | 5 | `estimateBandwidth`, `shiftPoint`, `example2.cpp` (with Mark) |

Every member has visible commits. Every member has to be able to explain the whole library, not only their own part.

How we worked together: one person (or a pair) per function, one branch per task, a pull request for every change, and a different member reviewing and merging. Weekly reports were written from the commit history.

## 10. Limitations (honest list)

1. **Speed.** Each step compares a point with every other point, so running time grows quickly with the number of points. It is fine for hundreds of points, not for very large data.
2. **One kernel.** Only the Gaussian kernel is available.
3. **Bandwidth guess is simple.** It is half the average distance between the first (up to) 500 points. A hand-chosen bandwidth is often better.
4. **Quality score is simple.** `evaluate()` measures tightness only. It does not measure how well separated the clusters are.
5. **CSV loading is forgiving, not strict.** Values that are not numbers are skipped silently. If a row has fewer numbers than the first row, scaling or distance calculation can read outside the row. The library does not check that all points have the same number of features.
6. **Output header is fixed.** The examples write the header `Feature1,Feature2,Cluster_ID`, so the header is wrong for files with a different number of features.
7. **Example 3.** It reads paths with `cin >>`, so paths with spaces do not work.
8. **Compiler.** `CMakeLists.txt` asks for C++23, so an older compiler will not build it.

## 11. Lessons learned

1. Keeping data loading (`Dataset`) separate from the algorithm (`MeanShift`) made testing and teamwork much easier.
2. Decimal numbers should not be compared with `==`, so we merge points that are close enough.
3. Testing on another computer found a bug that our own Windows computers hid (file name capital letters).
4. Most of our tests were added in Week 4 (2 tests in Week 3, 10 in Week 4). Writing each test together with its function would have caught problems earlier.
5. AI was useful for learning and checking, but we only kept what we understood and could explain.

## 12. AI-use declaration

AI was used as a helper for learning and checking. It was not used to write the whole project. Members reviewed and understood all code, tests and reports they contributed.

| Week | Tool | Purpose | Reason |
| :--- | :--- | :--- | :--- |
| 1 | Gemini, ChatGPT, Claude, Google Search | Explaining clustering, Gaussian kernels and bandwidth; finding beginner resources | We were new to machine learning |
| 2 | Gemini, ChatGPT, Claude | A little help with header/source layout and CSV reading | Minor help; we already knew what to do |
| 3 | Gemini, Claude | Finding where inheritance and polymorphism appear in our code | Our code had no direct inheritance like the class examples |
| 4 | Gemini, ChatGPT, Claude | Checking CMake settings and test coverage | To check the build and tests before submission |
| Final | Claude | Drafting this final report and the `README.md` from the repository files, reports and git history | To save time; the group checked every number and statement against the repository and supplied the external-testing results |

Per-member declarations are in each individual report:

| Member | Tools | Purpose |
| :--- | :--- | :--- |
| Happy | Gemini, ChatGPT | Z-score versus Min-Max scaling, zero-variance columns |
| Jonathan | Gemini, ChatGPT | How the kernel-weighted average relates to density gradient ascent |
| Joseph | Gemini | Modern CMake target export syntax; designing self-contained tests |
| Desire | Gemini, ChatGPT | Why points are merged within `bandwidth / 2` |
| Nazarene | Gemini, ChatGPT | Convergence stopping rules and per-point caching |
| Kizza | Gemini, ChatGPT | `stringstream` splitting and `std::stod` exceptions |
| Clive | Gemini, ChatGPT | Cohesion metric and silhouette score |
| Mark | Gemini, Claude | Inheritance and polymorphism in our code, before drawing diagrams |
| Simon | *(no individual report found; add your declaration)* | |

## 13. How to build and test (clean machine)

Requirements: Git, CMake 3.20 or newer, and a C++ compiler that supports C++23 (we checked with GCC 13).

```text
git clone https://github.com/josephmusiitwa/GROUP-21-PROJECT.git
cd GROUP-21-PROJECT
cmake -S . -B build
cmake --build build
cd build && ctest --output-on-failure
cd ..
./build/examples/example2
./build/examples/example1 data/input/my_data.csv data/output/results.csv
```