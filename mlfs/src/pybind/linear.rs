//! Python wrappers for linear models.

use super::{r2, to_py_err};
use crate::common::traits::{Estimator, Predictor};
use crate::linear::{LinearRegression as RustLinearRegression, Ridge as RustRidge};
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

/// Ridge (L2-penalised) regression.
#[pyclass(name = "Ridge", module = "mlfs")]
pub struct Ridge {
    inner: RustRidge,
}

#[pymethods]
impl Ridge {
    #[new]
    #[pyo3(signature = (alpha = 1.0, fit_intercept = true))]
    fn new(alpha: f64, fit_intercept: bool) -> PyResult<Self> {
        Ok(Self {
            inner: RustRidge::new(alpha, fit_intercept).map_err(to_py_err)?,
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
        "Ridge()".into()
    }
}
