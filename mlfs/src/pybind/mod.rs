//! Python bindings (PyO3). Each estimator family gets a submodule of thin
//! `#[pyclass]` wrappers that convert numpy arrays to `ndarray` views (zero
//! copy) and delegate to the pure-Rust implementations.

use crate::error::MlError;
use ndarray::{Array1, ArrayView1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

pub mod linear;

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
#[allow(dead_code)]
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
#[allow(dead_code)]
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

/// Register every `#[pyclass]` on the top-level module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<linear::LinearRegression>()?;
    m.add_class::<linear::Ridge>()?;
    Ok(())
}
