//! Python wrapper for the support vector classifier.

use super::{accuracy, to_py_err};
use crate::common::traits::{Estimator, Predictor};
use crate::svm::{Kernel, SVC as RustSvc};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Support Vector Classifier (SMO solver; linear or RBF kernel).
#[pyclass(name = "SVC", module = "mlfs")]
pub struct SVC {
    inner: RustSvc,
}

#[pymethods]
impl SVC {
    #[new]
    #[pyo3(signature = (
        c = 1.0,
        kernel = "rbf".to_string(),
        gamma = 0.5,
        tol = 1e-3,
        max_passes = 10,
        random_state = 0
    ))]
    fn new(
        c: f64,
        kernel: String,
        gamma: f64,
        tol: f64,
        max_passes: usize,
        random_state: u64,
    ) -> PyResult<Self> {
        let k = match kernel.as_str() {
            "linear" => Kernel::Linear,
            "rbf" => Kernel::Rbf { gamma },
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown kernel '{other}', expected 'linear' or 'rbf'"
                )))
            }
        };
        Ok(Self {
            inner: RustSvc::new(c, k, tol, max_passes, random_state).map_err(to_py_err)?,
        })
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
        "SVC()".into()
    }
}
