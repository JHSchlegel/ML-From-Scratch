"""mlfs — Machine Learning From Scratch.

A collection of classic ML algorithms implemented from scratch in Rust and
exposed to Python. The estimators follow a scikit-learn-like API::

    from mlfs import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100).fit(X, y)
    y_pred = model.predict(X_new)
    acc = model.score(X, y)

All estimators accept and return ``float64`` numpy arrays. Class labels are
passed and returned as floats.
"""

from ._mlfs import __version__  # noqa: F401
from ._mlfs import LinearRegression, Ridge, Lasso

__all__ = [
    "LinearRegression",
    "Ridge",
    "Lasso",
]
