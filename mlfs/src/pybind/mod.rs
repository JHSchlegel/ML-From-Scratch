//! Python bindings (PyO3). Each estimator family gets a submodule of thin
//! `#[pyclass]` wrappers that convert numpy arrays to `ndarray` views (zero
//! copy) and delegate to the pure-Rust implementations.

use crate::error::MlError;
use crate::tree::TreeParams;
use ndarray::{Array1, ArrayView1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

pub mod cluster;
pub mod decomposition;
pub mod ensemble;
pub mod linear;
pub mod naive_bayes;
pub mod neighbors;
pub mod nn;
pub mod svm;
pub mod tree;

/// Convert a crate error into an appropriate Python exception.
pub(crate) fn to_py_err(e: MlError) -> PyErr {
    match e {
        MlError::ShapeMismatch(_) | MlError::InvalidParameter(_) => {
            PyValueError::new_err(e.to_string())
        }
        MlError::NotFitted(_) | MlError::Numerical(_) => PyRuntimeError::new_err(e.to_string()),
    }
}

/// R^2 score of predictions against targets (regression `.score`).
pub(crate) fn r2(pred: &Array1<f64>, y: ArrayView1<f64>) -> f64 {
    let mean = y.mean().unwrap_or(0.0);
    let ss_res: f64 = y.iter().zip(pred.iter()).map(|(t, p)| (t - p).powi(2)).sum();
    let ss_tot: f64 = y.iter().map(|t| (t - mean).powi(2)).sum();
    if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    }
}

/// Classification accuracy (classifier `.score`).
pub(crate) fn accuracy(pred: &Array1<f64>, y: ArrayView1<f64>) -> f64 {
    if y.is_empty() {
        return 0.0;
    }
    let c = y
        .iter()
        .zip(pred.iter())
        .filter(|(t, p)| (*t - *p).abs() < 1e-9)
        .count();
    c as f64 / y.len() as f64
}

/// Build `TreeParams` from Python-friendly optional arguments.
pub(crate) fn make_tree_params(
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features: Option<usize>,
) -> TreeParams {
    TreeParams {
        max_depth: max_depth.unwrap_or(usize::MAX),
        min_samples_split,
        min_samples_leaf,
        max_features,
    }
}

/// Register every `#[pyclass]` on the top-level module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<linear::LinearRegression>()?;
    m.add_class::<linear::Ridge>()?;
    m.add_class::<linear::Lasso>()?;
    m.add_class::<linear::LogisticRegression>()?;

    m.add_class::<neighbors::KNeighborsClassifier>()?;
    m.add_class::<neighbors::KNeighborsRegressor>()?;

    m.add_class::<naive_bayes::GaussianNB>()?;

    m.add_class::<tree::DecisionTreeClassifier>()?;
    m.add_class::<tree::DecisionTreeRegressor>()?;

    m.add_class::<ensemble::RandomForestClassifier>()?;
    m.add_class::<ensemble::RandomForestRegressor>()?;
    m.add_class::<ensemble::GradientBoostingClassifier>()?;
    m.add_class::<ensemble::GradientBoostingRegressor>()?;
    m.add_class::<ensemble::AdaBoostClassifier>()?;

    m.add_class::<svm::SVC>()?;

    m.add_class::<nn::MLPClassifier>()?;
    m.add_class::<nn::MLPRegressor>()?;

    m.add_class::<decomposition::PCA>()?;
    m.add_class::<decomposition::TSNE>()?;

    m.add_class::<cluster::KMeans>()?;
    Ok(())
}
