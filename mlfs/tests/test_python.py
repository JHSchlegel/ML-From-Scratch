"""End-to-end smoke tests for every estimator exposed through the Python API.

Datasets are generated with numpy only (no sklearn dependency) so the test
suite runs against the library in isolation.
"""

import numpy as np
import pytest

import mlfs


def make_blobs(n_per=40, centers=((0, 0), (5, 5), (0, 5)), spread=0.6, seed=0):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for label, c in enumerate(centers):
        pts = rng.normal(loc=c, scale=spread, size=(n_per, 2))
        xs.append(pts)
        ys.append(np.full(n_per, float(label)))
    X = np.vstack(xs)
    y = np.concatenate(ys)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def make_regression(n=120, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    w = rng.standard_normal(d)
    y = X @ w + 0.1 * rng.standard_normal(n) + 2.0
    return X, y


# ----------------------------- regression --------------------------------- #

@pytest.mark.parametrize(
    "model",
    [
        mlfs.LinearRegression(),
        mlfs.Ridge(alpha=1.0),
        mlfs.Lasso(alpha=0.01),
        mlfs.KNeighborsRegressor(n_neighbors=5),
        mlfs.DecisionTreeRegressor(max_depth=6),
        mlfs.RandomForestRegressor(n_estimators=30, random_state=0),
        mlfs.GradientBoostingRegressor(n_estimators=80, learning_rate=0.1),
        mlfs.MLPRegressor(hidden_layer_sizes=[32], activation="tanh",
                          learning_rate=0.01, max_iter=400, random_state=0),
    ],
)
def test_regressors(model):
    X, y = make_regression()
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == y.shape
    assert model.score(X, y) > 0.5  # explains a good chunk of variance


# --------------------------- classification ------------------------------- #

@pytest.mark.parametrize(
    "model",
    [
        mlfs.LogisticRegression(lr=0.1, max_iter=500),
        mlfs.KNeighborsClassifier(n_neighbors=5),
        mlfs.GaussianNB(),
        mlfs.DecisionTreeClassifier(max_depth=8),
        mlfs.RandomForestClassifier(n_estimators=30, random_state=0),
        mlfs.GradientBoostingClassifier(n_estimators=50, learning_rate=0.2),
        mlfs.AdaBoostClassifier(n_estimators=30),
        mlfs.SVC(kernel="rbf", gamma=0.3, max_passes=5),
        mlfs.MLPClassifier(hidden_layer_sizes=[32], activation="relu",
                           learning_rate=0.02, max_iter=400, random_state=0),
    ],
)
def test_classifiers(model):
    X, y = make_blobs()
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == y.shape
    acc = model.score(X, y)
    assert acc > 0.9, f"accuracy too low: {acc}"


# ------------------------------ clustering -------------------------------- #

def test_kmeans():
    X, y = make_blobs()
    km = mlfs.KMeans(n_clusters=3, random_state=0)
    labels = km.fit_predict(X)
    assert labels.shape == y.shape
    assert km.cluster_centers_.shape == (3, 2)
    assert km.inertia_ > 0


def test_gmm():
    X, _ = make_blobs()
    gmm = mlfs.GaussianMixture(n_components=3, random_state=0)
    labels = gmm.fit_predict(X)
    assert set(np.unique(labels)).issubset({0.0, 1.0, 2.0})
    assert gmm.means_.shape == (3, 2)


def test_dbscan():
    X, _ = make_blobs(spread=0.3)
    db = mlfs.DBSCAN(eps=1.0, min_samples=5)
    labels = db.fit_predict(X)
    assert labels.shape[0] == X.shape[0]


def test_agglomerative():
    X, _ = make_blobs()
    ac = mlfs.AgglomerativeClustering(n_clusters=3, linkage="average")
    labels = ac.fit_predict(X)
    assert len(np.unique(labels)) == 3


# -------------------------- dimensionality -------------------------------- #

def test_pca():
    X, _ = make_blobs()
    pca = mlfs.PCA(n_components=2)
    Z = pca.fit_transform(X)
    assert Z.shape == (X.shape[0], 2)
    assert pca.explained_variance_ratio_.sum() <= 1.0 + 1e-9
    assert pca.components_.shape == (2, 2)


def test_tsne():
    X, _ = make_blobs(n_per=15)
    ts = mlfs.TSNE(n_components=2, perplexity=10.0, n_iter=250, random_state=0)
    Z = ts.fit_transform(X)
    assert Z.shape == (X.shape[0], 2)


# ------------------------------ utilities --------------------------------- #

def test_standard_scaler():
    X, _ = make_regression()
    scaler = mlfs.StandardScaler()
    Xs = scaler.fit_transform(X)
    assert np.allclose(Xs.mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(Xs.std(axis=0), 1.0, atol=1e-6)


def test_train_test_split():
    X, y = make_regression(n=100)
    Xtr, Xte, ytr, yte = mlfs.train_test_split(X, y, test_size=0.25, random_state=0)
    assert Xtr.shape[0] == 75
    assert Xte.shape[0] == 25
    assert ytr.shape[0] == 75 and yte.shape[0] == 25
