//! Python wrappers for linear models.

use super::{r2, to_py_err};
use crate::common::traits::{Estimator, Predictor};
use crate::linear::LinearRegression as RustLinearRegression;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Ordinary least squares linear regression.
#[pyclass(name = "LinearRegression", module = "mlfs")]
pub struct LinearRegression {
    inner: RustLinearRegression,
}

#[pymethods]
impl LinearRegression {
    #[new]
    #[pyo3(signature = (fit_intercept = true))]
    fn new(fit_intercept: bool) -> Self {
        Self {
            inner: RustLinearRegression::new(fit_intercept),
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
        Ok(r2(&pred, y.as_array()))
    }

    #[getter]
    fn coef_<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.coef().map(|c| c.to_owned().into_pyarray(py))
    }
    #[getter]
    fn intercept_(&self) -> f64 {
        self.inner.intercept()
    }
    fn __repr__(&self) -> String {
        "LinearRegression()".into()
    }
}
