//! Python wrappers for nearest-neighbour models.

use super::{accuracy, r2, to_py_err};
use crate::common::traits::{Estimator, Predictor};
use crate::neighbors::{KNeighborsClassifier as RustKnnC, KNeighborsRegressor as RustKnnR};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// k-nearest-neighbours classifier.
#[pyclass(name = "KNeighborsClassifier", module = "mlfs")]
pub struct KNeighborsClassifier {
    inner: RustKnnC,
}

#[pymethods]
impl KNeighborsClassifier {
    #[new]
    #[pyo3(signature = (n_neighbors = 5))]
    fn new(n_neighbors: usize) -> PyResult<Self> {
        Ok(Self {
            inner: RustKnnC::new(n_neighbors).map_err(to_py_err)?,
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
        "KNeighborsClassifier()".into()
    }
}

/// k-nearest-neighbours regressor.
#[pyclass(name = "KNeighborsRegressor", module = "mlfs")]
pub struct KNeighborsRegressor {
    inner: RustKnnR,
}

#[pymethods]
impl KNeighborsRegressor {
    #[new]
    #[pyo3(signature = (n_neighbors = 5))]
    fn new(n_neighbors: usize) -> PyResult<Self> {
        Ok(Self {
            inner: RustKnnR::new(n_neighbors).map_err(to_py_err)?,
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
    fn __repr__(&self) -> String {
        "KNeighborsRegressor()".into()
    }
}
