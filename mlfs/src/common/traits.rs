//! Common estimator traits.
//!
//! The design mirrors scikit-learn's fit / predict / transform contract while
//! staying idiomatic Rust: `fit` takes borrowed views and returns a `Result`,
//! predictors and transformers are separate capabilities a model may implement.

use crate::error::Result;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// A model that can be trained on a feature matrix `X` and target vector `y`.
///
/// `X` has shape `(n_samples, n_features)` and `y` has length `n_samples`.
/// For classifiers, `y` holds class indices encoded as `f64`.
pub trait Estimator {
    fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<()>;
}

/// A model that can predict targets for new samples.
pub trait Predictor {
    fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>>;
}

/// A model that maps a feature matrix to a new feature space
/// (e.g. PCA, StandardScaler).
pub trait Transformer {
    fn transform(&self, x: ArrayView2<f64>) -> Result<Array2<f64>>;
}
