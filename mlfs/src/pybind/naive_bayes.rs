//! Python wrapper for Gaussian Naive Bayes.

use super::{accuracy, to_py_err};
use crate::common::traits::{Estimator, Predictor};
use crate::naive_bayes::GaussianNB as RustGaussianNB;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Gaussian Naive Bayes classifier.
#[pyclass(name = "GaussianNB", module = "mlfs")]
pub struct GaussianNB {
    inner: RustGaussianNB,
}

#[pymethods]
impl GaussianNB {
    #[new]
    #[pyo3(signature = (var_smoothing = 1e-9))]
    fn new(var_smoothing: f64) -> Self {
        Self {
            inner: RustGaussianNB::new(var_smoothing),
        }
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner
            .fit(x.as_array(), y.as_array())
            .map_err(to_py_err)?;
        Ok(slf)
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self
            .inner
            .predict(x.as_array())
            .map_err(to_py_err)?
            .into_pyarray(py))
    }

    fn score(&self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
        let pred = self.inner.predict(x.as_array()).map_err(to_py_err)?;
        Ok(accuracy(&pred, y.as_array()))
    }
    fn __repr__(&self) -> String {
        "GaussianNB()".into()
    }
}
