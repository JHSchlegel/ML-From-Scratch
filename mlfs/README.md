# mlfs — Machine Learning From Scratch

A modular collection of classic machine-learning algorithms implemented from
scratch in **Rust**, with a **Python** interface generated via PyO3 + maturin.

"From scratch" means no ML libraries: the only borrowed primitives are
[`ndarray`](https://crates.io/crates/ndarray) for tensors and
[`linfa-linalg`](https://crates.io/crates/linfa-linalg) (pure-Rust, no LAPACK)
for a handful of matrix decompositions. Every algorithm — the fitting, the
optimisation, the tree building — is written by hand.

## Layout

```
mlfs/
  src/            Rust core
    common/       estimator traits, linear-algebra helpers, metrics, preprocessing
    linear/       linear / ridge / lasso / logistic regression
    neighbors/    k-nearest-neighbours
    naive_bayes/  gaussian naive Bayes
    tree/         CART decision trees
    ensemble/     random forest, gradient boosting, adaboost
    svm/          support vector classifier (SMO)
    nn/           multi-layer perceptron
    decomposition/ PCA, t-SNE
    cluster/      k-means, gaussian mixture, DBSCAN, agglomerative
    pybind/       PyO3 wrappers
  python/mlfs/    pure-Python package layer (re-exports the compiled module)
  examples/       showcase.ipynb
```

## Building

Rust only:

```bash
cargo test          # run the unit / integration tests
cargo build --release
```

Python extension (from a virtualenv with `maturin` installed):

```bash
maturin develop --release      # build + install `mlfs` into the active venv
python -c "import mlfs; print(mlfs.__version__)"
```

## Usage (Python)

```python
import numpy as np
from mlfs import LinearRegression

X = np.random.randn(100, 3)
y = X @ np.array([2.0, -1.0, 0.5]) + 3.0
model = LinearRegression().fit(X, y)
print(model.coef_, model.intercept_)
print(model.score(X, y))
```

See `examples/showcase.ipynb` for a full tour of every algorithm.
