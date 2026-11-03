# Group 21: Mean Shift Clustering Library in C++ (`meanshift`)

**Course Project:** Machine Learning 13 — Design and Implementation of the Mean Shift Clustering Algorithm in C++  
**Repository:** `GROUP-21-PROJECT` | **Language Standard:** C++23 | **Build System:** CMake & CTest

> **How to Use This Guide in VS Code:**  
> Open this file in VS Code and press `Ctrl+Shift+V` (or click **Open Preview to the Side**) to render the interactive documentation, mathematical formulas, and clickable hyperlinks. Click any file link below to jump directly to the source code in the VS Code editor.

---

## Quick Navigation Hub

### Presentation Sections
1. [1. Project Overview & Repository Structure](#1-project-overview--repository-structure)
2. [2. Design Decisions & OOP Principles](#2-design-decisions--oop-principles)
3. [3. Actual Code & Implementation Walkthrough](#3-actual-code--implementation-walkthrough)
   - [3.1 Module A: Data Ingestion & Feature Scaling (`Dataset`)](#31-module-a-data-ingestion--feature-scaling-dataset)
   - [3.2 Module B: Mean Shift Clustering Engine (`MeanShift` & `Cluster`)](#32-module-b-mean-shift-clustering-engine-meanshift--cluster)
4. [4. Individual Team Contributions](#4-individual-team-contributions)
5. [5. Practical Demonstrations & Automated Testing Playbook](#5-practical-demonstrations--automated-testing-playbook)
6. [6. Live Coding Walkthroughs](#6-live-coding-walkthroughs)

### Direct Repository File Links (Click to Open in VS Code)
* **Build & Configuration:** [`CMakeLists.txt`](./CMakeLists.txt) | [`examples/CMakeLists.txt`](./examples/CMakeLists.txt) | [`tests/CMakeLists.txt`](./tests/CMakeLists.txt) | [`.gitignore`](./.gitignore) | [`README.md`](./README.md)
* **Public Headers (`include/meanshift/`):** [`Dataset.hpp`](./include/meanshift/Dataset.hpp) | [`MeanShift.hpp`](./include/meanshift/MeanShift.hpp)
* **Core Implementations (`src/`):** [`Dataset.cpp`](./src/Dataset.cpp) | [`meanshift.cpp`](./src/meanshift.cpp)
* **Example Programs (`examples/`):** [`example1.cpp` (CLI CSV Workflow)](./examples/example1.cpp) | [`example2.cpp` (In-Memory Demo)](./examples/example2.cpp) | [`example3.cpp` (Interactive Tool)](./examples/example3.cpp)
* **Automated Test Suite (`tests/`):** [`test_meanshift.cpp` (10 Unit Tests)](./tests/test_meanshift.cpp)
* **Sample Datasets (`data/input/`):** [`my_data.csv`](./data/input/my_data.csv) | [`students.csv`](./data/input/students.csv) | [`biology_data.csv`](./data/input/biology_data.csv) | [`finance.csv`](./data/input/finance.csv) | [`text_data.csv`](./data/input/text_data.csv)
* **Project Reports (`reports/`):** [`finalreport.md`](./reports/finalreport.md) | [`week-01-report.md`](./reports/week-01-report.md) | [`week-2-report.md`](./reports/week-2-report.md) | [`week-3-report.md`](./reports/week-3-report.md) | [`week-4-report.md`](./reports/week-4-report.md)

---

## 1. Project Overview & Repository Structure

### 1.1 Problem Statement & Algorithmic Overview
Unlike parametric clustering methods such as K-Means—which require the user to specify the number of clusters $K$ in advance and assume spherical cluster shapes—**Mean Shift** is a **non-parametric, mode-seeking clustering algorithm**.
1. Each data point $x_i \in \mathbb{R}^D$ is treated as a particle ("walker") on an underlying kernel density surface controlled by a neighborhood radius (**bandwidth** $h$).
2. At each step, every active point computes the **Gaussian kernel-weighted mean** of all dataset points around it and shifts toward the region of highest local density.
3. Points stops moving once their step displacement drops below a convergence tolerance $\epsilon$ (`epsilon_`) or reaches `maxIterations_`.
4. Converged points landing within `bandwidth / 2.0` of one another are merged into a single `Cluster` centroid, automatically discovering the number of clusters $K$ from the data geometry.

Our objective was to engineer this algorithm as a **reusable, standard C++ static library (`libmeanshift.a`)** with clean header/implementation separation, feature preprocessing, fault-tolerant CSV I/O, automated unit tests, and example applications.

### 1.2 Repository Structure
```text
GROUP-21-PROJECT/
├── CMakeLists.txt                     # Root build config: static library (meanshift), install rules, subdirs
├── README.md                          # Project overview and usage documentation
├── Presentation.md                    # Technical presentation guide (this document)
├── LICENSE                            # Project license
├── .gitignore                         # Ignores build artifacts, binaries, and temporary files
├── include/
│   └── meanshift/
│       ├── Dataset.hpp                # Public interface: Point, Data aliases & Dataset class
│       └── MeanShift.hpp              # Public interface: Cluster struct & MeanShift class
├── src/
│   ├── Dataset.cpp                    # Implementation: CSV ingestion, standardize(), normalize()
│   └── meanshift.cpp                  # Implementation: KDE math, estimateBandwidth(), shiftPoint(), fit(), evaluate()
├── examples/
│   ├── CMakeLists.txt                 # Builds example1, example2, example3 linked against meanshift
│   ├── example1.cpp                   # CLI workflow: loads CSV, clusters, exports labeled CSV
│   ├── example2.cpp                   # In-memory workflow: clusters 9 synthetic 2D points directly
│   └── example3.cpp                   # Interactive CLI tool: prompts user for input/output CSV paths
├── tests/
│   ├── CMakeLists.txt                 # Registers test_meanshift executable with CTest
│   └── test_meanshift.cpp             # 10 automated unit tests (normal, edge, and invalid input cases)
├── data/
│   ├── input/                         # 5 test datasets: my_data.csv, students.csv, biology_data.csv, finance.csv, text_data.csv
│   └── output/                        # Destination folder for exported clustering results (results.csv)
└── reports/
    ├── finalreport.md                 # Complete project report
    ├── week-01-report.md ... week-4-report.md
    ├── images/                        # Architectural & OOP diagrams
    └── Individual reports/            # 9 detailed individual technical reports
```

### 1.3 Build System Walkthrough (`CMakeLists.txt`)
* **Inspect Files:** [`CMakeLists.txt`](./CMakeLists.txt) | [`examples/CMakeLists.txt`](./examples/CMakeLists.txt) | [`tests/CMakeLists.txt`](./tests/CMakeLists.txt)
* **Technical Highlights:**
  * **Static Library Target (Lines 9–12):** Compiles [`src/Dataset.cpp`](./src/Dataset.cpp) and [`src/meanshift.cpp`](./src/meanshift.cpp) into `libmeanshift.a` (`add_library(meanshift STATIC ...)`), decoupling library compilation from consumer executables.
  * **Target-Scoped Include Propagation (Lines 14–18):** Uses `target_include_directories(meanshift PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)` so any target linking `meanshift` automatically resolves `#include "meanshift/Dataset.hpp"` and `#include "meanshift/MeanShift.hpp"`.
  * **Modern C++ Standard & Package Export (Lines 20–38):** Enforces `cxx_std_23` and exports `MeanShift::meanshift` via `install(EXPORT MeanShiftTargets ...)` for external CMake projects.
  * **Downstream Executables & CTest Integration (Lines 41–44):** Links `example1`, `example2`, `example3`, and `test_meanshift` via `target_link_libraries(<target> PRIVATE meanshift)`.

[⬆ Back to Top](#quick-navigation-hub)

---

## 2. Design Decisions & OOP Principles

### 2.1 Core Engineering & Algorithmic Design Decisions

| Design Decision | Implementation Location | Technical Justification |
| :--- | :--- | :--- |
| **1. Strict Header / Source Separation** | [`include/meanshift/`](./include/meanshift/) $\rightarrow$ [`src/`](./src/) | Decouples the public API contract from mathematical implementation details, enabling clean static library packaging (`libmeanshift.a`). |
| **2. Decoupling `Dataset` from `MeanShift`** | [`Dataset.hpp`](./include/meanshift/Dataset.hpp) & [`MeanShift.hpp`](./include/meanshift/MeanShift.hpp) | `MeanShift::fit(const Data& data)` operates on the `Data` (`vector<vector<double>>`) type rather than `Dataset`. This allows clustering both CSV files ([`example1.cpp`](./examples/example1.cpp)) and in-memory vectors ([`example2.cpp`](./examples/example2.cpp)). |
| **3. Feature Scaling Pipeline (`standardize` & `normalize`)** | [`src/Dataset.cpp`](./src/Dataset.cpp) (Lines 48–103) | Mean Shift uses isotropic Euclidean distance inside a single scalar bandwidth $h$. Feature scaling prevents large-magnitude columns from dominating small-magnitude columns. |
| **4. Subsampled Bandwidth Estimation ($M \le 500$)** | [`src/meanshift.cpp`](./src/meanshift.cpp) (Lines 24–39) | Computing all pairwise distances costs $O(N^2)$. Capping the sample at `min<size_t>(500, data.size())` bounds estimation to at most $124,750$ pairs, running in milliseconds on large datasets. |
| **5. Per-Point Freeze Mask (`stopMoving`)** | [`src/meanshift.cpp`](./src/meanshift.cpp) (Lines 73–91) | Each `shiftPoint()` call costs $O(N \cdot D)$. Freezing points as soon as their step size drops below `epsilon_` (`stopMoving[i] = true`) skips redundant kernel evaluations in later iterations. |
| **6. Non-Mutating `fit(const Data&)` Execution** | [`src/meanshift.cpp`](./src/meanshift.cpp) (Line 72) | Copies input coordinates into a local working vector `vector<Point> shiftedPoints = data;` so the caller's original dataset is never mutated. |
| **7. Radius-Based Mode Merging ($h / 2$)** | [`src/meanshift.cpp`](./src/meanshift.cpp) (Lines 98–124) | Converged floating-point trajectories around the same density peak differ by up to $\sim \epsilon$. Grouping points within `bandwidth_ / 2.0` merges numerical jitter without collapsing distinct peaks. |
| **8. Fault-Tolerant Mixed-Type CSV Parser** | [`src/Dataset.cpp`](./src/Dataset.cpp) (Lines 12–42) | Skips optional headers, ignores blank lines, and wraps `std::stod(token)` in `try-catch` so mixed text/numeric datasets (like [`students.csv`](./data/input/students.csv)) automatically extract numeric columns without crashing. |
| **9. $O(N \cdot D)$ Intra-Cluster Cohesion Metric** | [`src/meanshift.cpp`](./src/meanshift.cpp) (Lines 134–146) | Computes the mean Euclidean distance from each original point `data[idx]` to its assigned `cluster.centroid`, avoiding $O(N^2)$ pairwise Silhouette overhead. |

### 2.2 Object-Oriented Programming (OOP) Principles
*(Visual architecture diagrams available in [`reports/images/`](./reports/images/): [`encapsulation_classes.jpeg`](./reports/images/encapsulation_classes.jpeg), [`encapsulation_namespace.jpeg`](./reports/images/encapsulation_namespace.jpeg), [`inheritance_stdlib.jpeg`](./reports/images/inheritance_stdlib.jpeg), [`polymorphism_overview.jpeg`](./reports/images/polymorphism_overview.jpeg))*

1. **Encapsulation & Data Hiding:**
   * All internal state is `private`: `data_` in [`Dataset`](./include/meanshift/Dataset.hpp) (Line 28), and `bandwidth_`, `epsilon_`, `maxIterations_`, `clusters_`, and `labels_` in [`MeanShift`](./include/meanshift/MeanShift.hpp) (Lines 36–42).
   * Read access is exposed strictly through `const` member functions returning `const` references (`const Data& getData() const`, `const vector<Cluster>& getClusters() const`, `const vector<int>& getLabels() const`), achieving **zero-copy read efficiency** while preventing external state corruption.
   * [`Cluster`](./include/meanshift/MeanShift.hpp) (Lines 12–15) is a passive data `struct` (`centroid` and `pointIndices`) safely exposed via `const vector<Cluster>&`.
2. **Composition & Standard Library Inheritance:**
   * Our domain classes (`Dataset` and `MeanShift`) have a *"uses-a"* workflow relationship rather than an *"is-a"* hierarchy, so we favor **composition over inheritance**.
   * **Standard Library Inheritance:** [`Dataset::loadFromCSV()`](./src/Dataset.cpp) leverages the C++ stream hierarchy (`std::ifstream` and `std::stringstream` inheriting from `std::istream`) and exception hierarchy (`std::invalid_argument` and `std::out_of_range` inheriting from `std::exception`).
3. **Compile-Time & Runtime Polymorphism:**
   * **Function Overloading via Default Arguments:** `MeanShift(double bandwidth = -1.0, double epsilon = 1e-3, int maxIterations = 300)` can be instantiated with 0, 1, 2, or 3 arguments; `loadFromCSV(const string& filename, bool hasHeader = true)` works with 1 or 2 arguments.
   * **Parametric Polymorphism (Templates):** Uses C++ STL templates (`std::vector<double>`, `std::vector<Point>`, `std::vector<Cluster>`, `std::min<size_t>`) to support arbitrary $D$-dimensional datasets.
   * **Runtime Polymorphism:** Uses base-class references in `std::getline(std::istream&, ...)` and `catch (const std::exception& e)`.

[⬆ Back to Top](#quick-navigation-hub)

---

## 3. Actual Code & Implementation Walkthrough

### 3.1 Module A: Data Ingestion & Feature Scaling (`Dataset`)
* **Header File:** [`include/meanshift/Dataset.hpp`](./include/meanshift/Dataset.hpp) (Lines 1–33)
* **Implementation File:** [`src/Dataset.cpp`](./src/Dataset.cpp) (Lines 1–105)

#### 1. Type Aliases & Class Interface ([`include/meanshift/Dataset.hpp`](./include/meanshift/Dataset.hpp))
```cpp
namespace meanshift {

using Point = vector<double>;
using Data = vector<Point>;

class Dataset {
public:
    Dataset();
    bool loadFromCSV(const string& filename, bool hasHeader = true);
    const Data& getData() const;
    void standardize(); // Z-score normalization (mean = 0, std = 1)
    void normalize();   // Min-Max scaling to [0, 1]
private:
    Data data_;
};
} // namespace meanshift
```

#### 2. Fault-Tolerant CSV Ingestion — `Dataset::loadFromCSV()` ([`src/Dataset.cpp`](./src/Dataset.cpp), Lines 12–42)
* Validates file stream opening (`if (!file.is_open())`), printing an error to `std::cerr` and returning `false` on missing files.
* Consumes the header row when `hasHeader == true`.
* Tokenizes each line by comma `','` using `std::stringstream` and parses numeric tokens via `std::stod(token)` inside a `try-catch` block:
  ```cpp
  while (getline(ss, token, ',')) {
      try {
          p.push_back(stod(token));
      } catch (const exception& e) {
      }
  }
  if (!p.empty()) {
      data_.push_back(p);
  }
  ```
* **Why this matters:** In [`data/input/students.csv`](./data/input/students.csv), rows contain `StudentName,Math_Score,Science_Score` (e.g., `John Smith,85,92`). `stod("John Smith")` throws `std::invalid_argument`, which is safely caught and ignored, extracting `{85.0, 92.0}` automatically.

#### 3. Z-Score Standardization — `Dataset::standardize()` ([`src/Dataset.cpp`](./src/Dataset.cpp), Lines 48–80)
* Rescales each feature column $j$ across $N$ points to zero mean ($\mu_j = 0$) and unit population standard deviation ($\sigma_j = 1$):
  $$\mu_j = \frac{1}{N}\sum_{i=0}^{N-1} x_{ij}, \qquad \sigma_j = \sqrt{\frac{1}{N}\sum_{i=0}^{N-1}(x_{ij} - \mu_j)^2}, \qquad z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$
* **Defensive Guards:** Checks `if (data_.empty()) return;` and guards against division by zero on constant columns with `if (stdDevs[i] != 0)`.

#### 4. Min-Max Normalization — `Dataset::normalize()` ([`src/Dataset.cpp`](./src/Dataset.cpp), Lines 82–103)
* Linearly maps each feature column $j$ into $[0.0, 1.0]$:
  $$x'_{ij} = \frac{x_{ij} - \min(x_j)}{\max(x_j) - \min(x_j)}$$
* **Defensive Guards:** Initializes `Point mins = data_[0]; Point maxs = data_[0];` (avoiding the common bug of initializing `mins` to `0.0` when all data values are positive) and checks `if (maxs[i] - mins[i] != 0)` before dividing.

---

### 3.2 Module B: Mean Shift Clustering Engine (`MeanShift` & `Cluster`)
* **Header File:** [`include/meanshift/MeanShift.hpp`](./include/meanshift/MeanShift.hpp) (Lines 1–47)
* **Implementation File:** [`src/meanshift.cpp`](./src/meanshift.cpp) (Lines 1–148)

#### 1. Euclidean Distance & Radial Gaussian Kernel ([`src/meanshift.cpp`](./src/meanshift.cpp), Lines 12–22)
* **`euclideanDistance(const Point& a, const Point& b)`:** Declared `static` because it is a pure mathematical function of two points:
  $$d(a, b) = \|a - b\|_2 = \sqrt{\sum_{i=0}^{D-1} (a_i - b_i)^2}$$
* **`gaussianKernel(double distance) const`:** Computes the smooth radial weight using private member `bandwidth_` ($h$):
  $$K(d) = \exp\left(-\frac{d^2}{2h^2}\right)$$
  ```cpp
  double MeanShift::gaussianKernel(double distance) const {
      return exp(-0.5 * (distance * distance) / (bandwidth_ * bandwidth_));
  }
  ```

#### 2. Data-Driven Bandwidth Estimation — `estimateBandwidth()` ([`src/meanshift.cpp`](./src/meanshift.cpp), Lines 24–39)
* Triggered automatically inside `fit()` whenever `bandwidth_ <= 0` (the default constructor sets `bandwidth = -1.0`).
* Samples up to $M = \min(500, N)$ points, averages all upper-triangular pairwise distances ($j > i$), and sets $h = 0.5 \times \bar{d}$ (with a fallback to `1.0` if `count == 0` or `bandwidth_ <= 0`):
  ```cpp
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
  ```

#### 3. Single-Point Density Gradient Step — `shiftPoint()` ([`src/meanshift.cpp`](./src/meanshift.cpp), Lines 41–63)
* Computes the kernel-weighted center of mass around query point $p$:
  $$m(p) = \frac{\sum_{j=0}^{N-1} K(\|p - x_j\|_2) \, x_j}{\sum_{j=0}^{N-1} K(\|p - x_j\|_2)}$$
* Accumulates `numerator[i] += other_p[i] * weight` and `denominator += weight` across all dataset points, guarding against underflow with `if (denominator > 0)`.

#### 4. Iterative Convergence & Mode Merging — `fit()` ([`src/meanshift.cpp`](./src/meanshift.cpp), Lines 65–124)
* **Stage 1: Iterative Mode-Seeking Loop (Lines 65–96):**
  * Guards against empty input (`if (data.empty()) return;`) and auto-estimates bandwidth if `bandwidth_ <= 0`.
  * Maintains `vector<Point> shiftedPoints = data;` and `vector<bool> stopMoving(data.size(), false);`.
  * Shifts active points until `euclideanDistance(shiftedPoints[i], newPoint) < epsilon_` or `maxIterations_` is reached, breaking early when `allStopped == true`.
* **Stage 2: Post-Convergence Cluster Merging & Labeling (Lines 98–124):**
  * Resets `clusters_.clear()` and `labels_.assign(data.size(), -1)`.
  * For each converged point `shiftedPoints[i]`, checks if it lies within `bandwidth_ / 2.0` of any existing `cluster.centroid`:
    * If **found**, appends index `i` to `cluster.pointIndices` and sets `labels_[i] = clusterIdx`.
    * If **not found**, creates a new `Cluster` with `centroid = shiftedPoints[i]`, records index `i`, and sets `labels_[i] = clusters_.size() - 1`.

#### 5. Clustering Quality Metric — `evaluate()` ([`src/meanshift.cpp`](./src/meanshift.cpp), Lines 134–146)
* Computes the **Mean Intra-Cluster Distance (Cohesion Score)** in $O(N \cdot D)$ time:
  $$\text{Cohesion}(X, \mathcal{C}) = \frac{1}{N} \sum_{k=0}^{K-1} \sum_{i \in C_k.\text{pointIndices}} \|x_i - c_k\|_2$$
* Lower values indicate tighter, more compact clusters around their discovered modes.

[⬆ Back to Top](#quick-navigation-hub)

---

## 4. Individual Team Contributions

> **Tip for Assessors:** Click any file link or Individual Report link below to inspect the exact code and documentation written by each member in VS Code.

| # | Team Member | Primary Role & Module Ownership | Code Files, Methods & Tests to Highlight | Individual Report |
| :---: | :--- | :--- | :--- | :--- |
| 1 | **Kizza Johnson Ssozi** | `Dataset` Class Architecture, CSV Ingestion & Statistical Moments | [`Dataset.hpp`](./include/meanshift/Dataset.hpp), [`src/Dataset.cpp`](./src/Dataset.cpp) (`Dataset()`, `loadFromCSV`, `getData`, mean/stdDev passes in `standardize`), `testEstimateBandwidth()`, `testGetLabels()`, [`biology_data.csv`](./data/input/biology_data.csv) | [`kizza.md`](./reports/Individual%20reports/kizza.md) |
| 2 | **Atuzarirwe Happy** | Feature Scaling & Preprocessing Algorithms | [`src/Dataset.cpp`](./src/Dataset.cpp) (Z-score scaling in `standardize` Lines 73–80, full `normalize` Lines 82–103), `testLoadFromCSV()`, `testStandardize()`, [`students.csv`](./data/input/students.csv) | [`Happy.md`](./reports/Individual%20reports/Happy.md) |
| 3 | **Atugonza Simon** | `Cluster` Struct, `MeanShift` Interface, Distance/Kernel Math & CLI App | [`MeanShift.hpp`](./include/meanshift/MeanShift.hpp), [`src/meanshift.cpp`](./src/meanshift.cpp) (`MeanShift` ctor, `euclideanDistance`, `gaussianKernel`), [`examples/example1.cpp`](./examples/example1.cpp), [`text_data.csv`](./data/input/text_data.csv) | [`simon_atugonza.md`](./reports/Individual%20reports/simon_atugonza.md) |
| 4 | **Sserunkuma Jonathan G.** | Automatic Bandwidth Estimation & Kernel Shift Update | [`src/meanshift.cpp`](./src/meanshift.cpp) (`estimateBandwidth` Lines 24–39, `shiftPoint` Lines 41–63), `testShiftPoint()`, co-developed [`examples/example2.cpp`](./examples/example2.cpp) | [`Jonathan.md`](./reports/Individual%20reports/Jonathan.md) |
| 5 | **Muhindo Nazarene Kamabu** | Iterative Mode-Seeking Engine & Convergence Control (`fit` Stage 1) | [`src/meanshift.cpp`](./src/meanshift.cpp) (`fit` Stage 1 Lines 65–96: `stopMoving` optimization, `epsilon_` convergence, `allStopped` early exit), [`finance.csv`](./data/input/finance.csv) | [`nazarene.md`](./reports/Individual%20reports/nazarene.md) |
| 6 | **Muzaki Desire Wadelo.F** | Post-Convergence Mode Merging & Label Assignment (`fit` Stage 2) | [`src/meanshift.cpp`](./src/meanshift.cpp) (`fit` Stage 2 Lines 98–124: mode merging within `bandwidth_ / 2.0`, populating `clusters_` & `labels_`), `testNormalize()`, [`examples/example3.cpp`](./examples/example3.cpp) | [`desire_muzaki.md`](./reports/Individual%20reports/desire_muzaki.md) |
| 7 | **Clive** | Output Accessors, Cohesion Evaluation Metric & Interactive Tool | [`src/meanshift.cpp`](./src/meanshift.cpp) (`getClusters`, `getLabels`, `evaluate` Lines 126–146), `testEvaluate()`, `testEmptyData()`, [`examples/example3.cpp`](./examples/example3.cpp) | [`Clive.md`](./reports/Individual%20reports/Clive.md) |
| 8 | **Mark Jordan** | In-Memory Demonstration, Default Dataset & OOP Architecture Diagrams | [`examples/example2.cpp`](./examples/example2.cpp), [`data/input/my_data.csv`](./data/input/my_data.csv), OOP diagrams in [`reports/images/`](./reports/images/), PR reviews & merges | [`Mark_Jordan.md`](./reports/Individual%20reports/Mark_Jordan.md) |
| 9 | **Musiitwa Joseph** | CMake Build System, Repository Integration & Automated Unit Test Suite | [`CMakeLists.txt`](./CMakeLists.txt), [`examples/CMakeLists.txt`](./examples/CMakeLists.txt), [`tests/CMakeLists.txt`](./tests/CMakeLists.txt), [`.gitignore`](./.gitignore), [`tests/test_meanshift.cpp`](./tests/test_meanshift.cpp) (10-test suite alignment) | [`Musiitwa_Joseph.md`](./reports/Individual%20reports/Musiitwa_Joseph.md) |

### Live Git Attribution Commands (Run in VS Code Terminal `Ctrl + ~`)
```bash
# 1. View visual commit history, branch structure, and Pull Request merges
git log --oneline --graph --decorate -n 25

# 2. View commit counts per team contributor
git shortlog -sn --all

# 3. Inspect line-by-line authorship on core implementation files
git blame src/Dataset.cpp
git blame src/meanshift.cpp
```

[⬆ Back to Top](#quick-navigation-hub)

---

## 5. Practical Demonstrations & Automated Testing Playbook

Run these commands live in the VS Code integrated terminal (`Ctrl + ~`) to demonstrate compilation, unit test verification, and practical execution on real datasets.

### Step 1: Configure & Build the Library and Executables
```bash
cmake -B build
cmake --build build
```
* **What to point out:** CMake compiles `libmeanshift.a` first, then links the 3 example binaries (`example1`, `example2`, `example3`) and the test runner (`test_meanshift`).

---

### Step 2: Execute the 10 Automated Unit Tests (`CTest`)
* **Inspect Test File:** [`tests/test_meanshift.cpp`](./tests/test_meanshift.cpp)
```bash
ctest --test-dir build --output-on-failure --verbose
```

#### Summary of the 10 Unit Tests in [`tests/test_meanshift.cpp`](./tests/test_meanshift.cpp)

| # | Test Function | Lines | Category | What It Verifies |
| :---: | :--- | :---: | :--- | :--- |
| 1 | `testEuclideanDistance()` | 10–16 | Normal Use | 3-4-5 right triangle distance between `{0.0, 0.0}` and `{3.0, 4.0}` equals `5.0`. |
| 2 | `testMeanShiftClustering()` | 18–28 | Normal Use | 6 points in 2 separated groups (`{0,0}..{0.2,0.2}` and `{10,10}..{10.2,10.2}`) converge into `2` clusters. |
| 3 | `testShiftPoint()` | 30–41 | Symmetry / Math | Midpoint `{1.0, 1.0}` equidistant between `{0.0, 0.0}` and `{2.0, 2.0}` stays stationary at `{1.0, 1.0}`. |
| 4 | `testEstimateBandwidth()` | 42–52 | Auto-Config | Default `MeanShift ms;` (`bandwidth = -1.0`) estimates a positive bandwidth and finds `2` clusters. |
| 5 | `testGetLabels()` | 54–69 | Output Integrity | Returns `6` labels where points `0..2` share one label, `3..5` share another, and `labels[0] != labels[3]`. |
| 6 | `testEvaluate()` | 70–80 | Cohesion Metric | Verifies intra-cluster distance score satisfies `0.0 < score <= 0.5` on compact clusters. |
| 7 | `testEmptyData()` | 81–89 | Edge / Invalid Input | Passing empty `Data` (`{}`) leaves `clusters` and `labels` empty and returns `evaluate() == 0.0` safely. |
| 8 | `testLoadFromCSV()` | 90–110 | I/O & Invalid File | Loads self-cleaning temp CSV (`2` rows) and verifies `"non_existent_file.csv"` returns `false` without crashing. |
| 9 | `testStandardize()` | 112–132 | Preprocessing | Rescales `{1.0, 10.0}` and `{3.0, 30.0}` ($10\times$ scale gap) to exact Z-scores `{-1.0, +1.0}`. |
| 10 | `testNormalize()` | 133–154 | Preprocessing | Rescales `{10, 50}`, `{20, 100}`, `{30, 150}` to exact Min-Max values `{0.0, 0.5, 1.0}`. |

---

### Step 3: Run the 3 Practical Example Programs

#### Demo 3A: Command-Line CSV Clustering & Export ([`examples/example1.cpp`](./examples/example1.cpp))
1. **Default Run on [`data/input/my_data.csv`](./data/input/my_data.csv) (30 points in 3 groups):**
   ```bash
   ./build/examples/example1 data/input/my_data.csv data/output/results.csv
   ```
   * **Expected Output:** Discovers **3 clusters** (10 points each) around centroids `(2.00108, 2.0977)`, `(7.92302, 8.01874)`, and `(2.07534, 7.94359)` with cohesion score **`0.259014`**, and writes labeled rows (`Feature1,Feature2,Cluster_ID`) to `data/output/results.csv`.
2. **Mixed Text & Numeric CSV Run on [`data/input/students.csv`](./data/input/students.csv):**
   ```bash
   ./build/examples/example1 data/input/students.csv data/output/students_results.csv
   ```
   * **What to highlight:** `loadFromCSV()` automatically ignores the `StudentName` text column, clusters the 12 students by `(Math_Score, Science_Score)` into **4 performance groups**, and exports the labeled results.
3. **Empty File Edge Case on [`data/input/text_data.csv`](./data/input/text_data.csv):**
   ```bash
   ./build/examples/example1 data/input/text_data.csv data/output/empty_results.csv
   ```
   * **What to highlight:** Gracefully reports `Number of clusters found: 0` and `Clustering evaluation score: 0` without crashing.

#### Demo 3B: In-Memory Synthetic Data Clustering ([`examples/example2.cpp`](./examples/example2.cpp))
```bash
./build/examples/example2
```
* **What to highlight:** Demonstrates low coupling—`MeanShift` clusters a `meanshift::Data` vector defined directly in C++ memory (9 points in 3 groups) without needing `Dataset` or CSV files, finding **3 clusters** with cohesion score **`0.622962`**.

#### Demo 3C: Interactive User Console Tool ([`examples/example3.cpp`](./examples/example3.cpp))
```bash
./build/examples/example3
```
* When prompted in the terminal, enter:
  * Input path: `data/input/biology_data.csv` (or `data/input/finance.csv`)
  * Output path: `data/output/bio_results.csv`

[⬆ Back to Top](#quick-navigation-hub)

---

## 6. Live Coding Walkthroughs

During the live coding segment, open one of the files below in VS Code, make the live edit, recompile with `cmake --build build`, and run the binary to show how the algorithm responds in real time.

### Option A: Live Bandwidth Sensitivity & Feature Scaling Experiment (Recommended)
* **Target File:** Open [`examples/example1.cpp`](./examples/example1.cpp) (around Line 35) or [`examples/example2.cpp`](./examples/example2.cpp) (around Line 15).
* **Goal:** Show live how changing the bandwidth $h$ (small vs. auto-estimated vs. large) and applying `dataset.standardize()` affects cluster discovery and cohesion scores.

Replace Lines 35–50 in [`examples/example1.cpp`](./examples/example1.cpp) with this live experiment block:

```cpp
    // --- LIVE CODING DEMO: Feature Scaling & Bandwidth Sensitivity ---
    dataset.standardize(); // Rescale features to mean = 0, stdDev = 1

    vector<double> testBandwidths = {0.3, -1.0, 5.0}; // Small, Auto-Estimated (-1.0), and Large h
    for (double bw : testBandwidths) {
        meanshift::MeanShift ms(bw, 1e-3, 300);
        ms.fit(dataset.getData());

        cout << "\n[Live Demo] Bandwidth arg = " << bw
             << " | Clusters Found = " << ms.getClusters().size()
             << " | Cohesion Score = " << ms.evaluate(dataset.getData()) << endl;
    }
```

**Recompile and run in terminal:**
```bash
cmake --build build
./build/examples/example1 data/input/my_data.csv data/output/results.csv
```

**Technical Explanation for Assessors:**
* **Small Bandwidth (`bw = 0.3`):** Narrow Gaussian kernel window $\rightarrow$ each point only sees immediate neighbors, fragmenting data into many small local modes (**over-clustering**).
* **Auto-Estimated Bandwidth (`bw = -1.0`):** Triggers `estimateBandwidth()`, setting $h$ to half the average pairwise distance and recovering the **3 true underlying clusters**.
* **Large Bandwidth (`bw = 5.0`):** Wide Gaussian kernel window $\rightarrow$ smooths over separate density peaks and pulls all points into **1 global cluster** (**under-clustering**).

---

### Option B: Live Single-Point Gradient Ascent Trace (`shiftPoint` Trajectory)
* **Target File:** Open [`src/meanshift.cpp`](./src/meanshift.cpp) inside `MeanShift::fit()` at **Line 84** (right after `double dist = euclideanDistance(shiftedPoints[i], newPoint);`).
* **Goal:** Prove how a single data point (`i == 0`) climbs uphill step-by-step toward its cluster mode until `dist < epsilon_`.

Insert this 5-line diagnostic trace inside [`src/meanshift.cpp`](./src/meanshift.cpp) at Line 84:

```cpp
            // --- LIVE CODING DEMO: Trace Point 0 Trajectory ---
            if (i == 0) {
                cout << "  [Trace Point 0] Iter " << iter
                     << " -> Pos: (" << newPoint[0] << ", " << newPoint[1] << ")"
                     << " | Step Delta: " << dist << endl;
            }
```

*(Note: Ensure `#include <iostream>` is present at the top of [`src/meanshift.cpp`](./src/meanshift.cpp).)*

**Recompile and run in terminal:**
```bash
cmake --build build
./build/examples/example2
```

**Technical Explanation for Assessors:**
* Point out how `Step Delta` (`dist`) decreases monotonically on every iteration as Point 0 approaches the local density peak $(1.50006, 1.35186)$.
* Once `Step Delta` drops below `epsilon_ = 0.001`, `stopMoving[0] = true` freezes Point 0 so no further iterations are wasted on it.

---

### Option C: Live Unit Test Addition in `tests/test_meanshift.cpp`
* **Target File:** Open [`tests/test_meanshift.cpp`](./tests/test_meanshift.cpp) at the bottom of the file (Line 155).
* **Goal:** Demonstrate how individual mathematical properties of `MeanShift` are unit-tested with CTest.

Add a new test verifying Gaussian kernel boundary properties ($K(0) = 1.0$ and $K(d)$ decreases with distance) and call it in `main()`:

```cpp
void testGaussianKernelDecay() {
    meanshift::MeanShift ms(2.0, 1e-3, 100);
    double w0 = ms.gaussianKernel(0.0);
    double w1 = ms.gaussianKernel(1.0);
    double w5 = ms.gaussianKernel(5.0);
    if (abs(w0 - 1.0) > 1e-9 || !(w0 > w1 && w1 > w5)) {
        throw runtime_error("testGaussianKernelDecay failed");
    }
    cout << "testGaussianKernelDecay passed." << endl;
}
```

**Recompile and run CTest:**
```bash
cmake --build build
ctest --test-dir build --output-on-failure --verbose
```

[⬆ Back to Top](#quick-navigation-hub)
