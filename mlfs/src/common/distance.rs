//! Distance and kernel helpers shared by KNN, SVM, clustering, etc.

use ndarray::ArrayView1;

/// Squared Euclidean distance (cheaper than the full norm; monotonic in it).
pub fn squared_euclidean(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Euclidean (L2) distance.
pub fn euclidean(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    squared_euclidean(a, b).sqrt()
}

/// Manhattan (L1) distance.
pub fn manhattan(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

/// Gaussian RBF kernel `exp(-gamma * ||a - b||^2)`.
pub fn rbf_kernel(a: ArrayView1<f64>, b: ArrayView1<f64>, gamma: f64) -> f64 {
    (-gamma * squared_euclidean(a, b)).exp()
}
