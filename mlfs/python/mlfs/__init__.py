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
# Linear models
from ._mlfs import LinearRegression, Ridge, Lasso, LogisticRegression

# Neighbours
from ._mlfs import KNeighborsClassifier, KNeighborsRegressor

# Naive Bayes
from ._mlfs import GaussianNB

# Trees
from ._mlfs import DecisionTreeClassifier, DecisionTreeRegressor

# Ensembles
from ._mlfs import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    AdaBoostClassifier,
)

# SVM
from ._mlfs import SVC

# Neural networks
from ._mlfs import MLPClassifier, MLPRegressor

# Dimensionality reduction
from ._mlfs import PCA, TSNE

# Clustering
from ._mlfs import KMeans, GaussianMixture, DBSCAN, AgglomerativeClustering

__all__ = [
    "LinearRegression",
    "Ridge",
    "Lasso",
    "LogisticRegression",
    "KNeighborsClassifier",
    "KNeighborsRegressor",
    "GaussianNB",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "AdaBoostClassifier",
    "SVC",
    "MLPClassifier",
    "MLPRegressor",
    "PCA",
    "TSNE",
    "KMeans",
    "GaussianMixture",
    "DBSCAN",
    "AgglomerativeClustering",
]
