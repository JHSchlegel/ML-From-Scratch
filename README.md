# Machine Learning From Scratch

A collection of machine learning algorithms implemented from scratch in various programming languages.

## Overview

This repository contains fundamental machine learning algorithms built without relying on high-level ML libraries, providing educational implementations that demonstrate core concepts and mathematical foundations.

## Current Implementation Status

### Completed
- **Linear Regression** (C++)
  - Ordinary Least Squares
  - Ridge Regression
  - Lasso Regression
- **K-Means Clustering** (C++)
- **Simulated Annealing** (Julia)

### Work in Progress
- **Random Forest** (C++)
- **Gradient Boosting** (Rust)

## Structure

```
00_supervised_learning/
├── linear_regression/
│   ├── ordinary_least_squares/
│   └── ridge_regression/
└── random_forest/

01_unsupervised_learning/
└── clustering/
    └── kmeans/
```

## Building with CMake

Each algorithm includes its own CMakeLists.txt. To build:

```bash
cd <algorithm_directory>
mkdir build
cd build
cmake ..
make
```

Example:
```bash
cd 00_supervised_learning/linear_regression/ordinary_least_squares
mkdir build
cd build
cmake ..
make
./linear_regression
```

## Code Formatting

This project uses clang-format for consistent C++ code style. To format code:

```bash
# Format a single file
clang-format -i <filename>

# Format all C++ files recursively
find . -name "*.cpp" -o -name "*.hpp" -o -name "*.h" | xargs clang-format -i
```
