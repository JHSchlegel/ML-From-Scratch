# Machine Learning From Scratch

A modular library of the most important machine-learning algorithms, implemented
**from scratch in Rust** with a **Python** interface.

## Overview

Everything lives in the [`mlfs/`](mlfs/) crate: ~18 classic ML algorithms built
without any ML libraries, exposed to Python (via PyO3 + maturin) behind a
scikit-learn-style API. "From scratch" means the only borrowed primitives are
`ndarray` for tensors and `linfa-linalg` (pure Rust, no LAPACK) for a handful of
matrix decompositions — every model's fitting logic is written by hand.

| Category | Algorithms |
| --- | --- |
| Linear models | LinearRegression, Ridge, Lasso, LogisticRegression |
| Neighbours / Bayes | KNeighbors{Classifier,Regressor}, GaussianNB |
| Trees & ensembles | DecisionTree{Classifier,Regressor}, RandomForest{…}, GradientBoosting{…}, AdaBoost |
| Kernel & neural | SVC (SMO; linear/RBF), MLP{Classifier,Regressor} |
| Unsupervised | KMeans, GaussianMixture, DBSCAN, AgglomerativeClustering |
| Dimensionality reduction | PCA, t-SNE |
| Utilities | StandardScaler, train_test_split, metrics |

## Quick start

```python
import numpy as np
from mlfs import RandomForestClassifier, StandardScaler, train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
scaler = StandardScaler().fit(X_train)
clf = RandomForestClassifier(n_estimators=100).fit(scaler.transform(X_train), y_train)
print(clf.score(scaler.transform(X_test), y_test))
```

See [`mlfs/examples/showcase.ipynb`](mlfs/examples/showcase.ipynb) for a full,
plotted tour of every algorithm.

## Building

```bash
cd mlfs
cargo test                     # run the Rust unit tests
python3 -m venv ../.venv && source ../.venv/bin/activate
pip install maturin numpy
maturin develop --release      # build + install the `mlfs` Python package
python -c "import mlfs; print(mlfs.__version__)"
```

For the notebook, also install the extras: `pip install matplotlib scikit-learn
pandas jupyter` (scikit-learn is used only to load toy datasets and sanity-check
results — never inside the library). See [`mlfs/README.md`](mlfs/README.md) for
the full module layout.
