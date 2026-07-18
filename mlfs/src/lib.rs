//! # mlfs — Machine Learning From Scratch
//!
//! A modular collection of classic machine-learning algorithms implemented in
//! Rust with no ML-library dependencies (only `ndarray` for tensors and
//! `linfa-linalg` for a handful of matrix decompositions). The same crate is
//! compiled into a Python extension module via PyO3 + maturin.
//!
//! Rust users depend on the library crate directly; Python users `import mlfs`.

pub mod common;
pub mod error;
pub mod linear;
pub mod naive_bayes;
pub mod neighbors;
pub mod tree;

pub use common::{Estimator, Predictor, Transformer};
pub use error::{MlError, Result};

// The Python-facing layer is only compiled into the extension module.
mod pybind;

use pyo3::prelude::*;

/// The compiled Python module (`mlfs._mlfs`).
#[pymodule]
fn _mlfs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    pybind::register(m)?;
    Ok(())
}
