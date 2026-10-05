# MeanShift Clustering Library

## Project Objectives
The objective of this project is to provide a robust, reusable C++ library for Mean Shift clustering. The algorithm automatically discovers the number of clusters and identifies dense regions of data. It addresses the Group 21 task: "Machine Learning 13 (Design and Implementation of the Mean Shift Clustering Algorithm in C++)".

## Main Features
- Dataset Loading (CSV support)
- Data Preprocessing (Standardization, Normalization)
- Mean Shift Clustering (Gaussian Kernel, Euclidean Distance)
- Automatic Bandwidth Estimation
- Cluster Evaluation

## Project Structure
- `include/`: Public headers (`Dataset.hpp`, `MeanShift.hpp`)
- `src/`: Implementation files
- `tests/`: Unit tests for the library
- `examples/`: Sample usage program
- `data/`: Sample input and output data

## Requirements
- C++17 compatible compiler
- CMake 3.10+

## Installation and Build Instructions
```bash
mkdir build
cd build
cmake ..
cmake --build .
```

## How to use the library
Include the library headers `meanshift/Dataset.hpp` and `meanshift/MeanShift.hpp` in your project and link against the `meanshift` library.
See `examples/example1.cpp` for a working example.

## How to run the tests
From the build directory, simply run:
```bash
ctest
# or run the executable directly
./test_meanshift
```

## Contributions
- Group Member 1: Dataset processing and scaling.
- Group Member 2: Distance calculation, Kernel function, Mean Shift algorithm.
- Group Member 3: Tests, Examples, Documentation.
- (Adjust as needed for your team).

## Collaboration
Our team divided tasks modularly, communicated regularly on technical decisions, and performed code reviews via GitHub Pull Requests to ensure integration of the Mean Shift components went smoothly.
