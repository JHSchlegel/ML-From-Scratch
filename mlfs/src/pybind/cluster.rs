//! Python wrappers for clustering algorithms.

use super::to_py_err;
use crate::cluster::KMeans as RustKMeans;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

/// K-Means clustering.
#[pyclass(name = "KMeans", module = "mlfs")]
pub struct KMeans {
    inner: RustKMeans,
}

#[pymethods]
impl KMeans {
    #[new]
    #[pyo3(signature = (n_clusters = 8, max_iter = 300, tol = 1e-4, random_state = 0))]
    fn new(n_clusters: usize, max_iter: usize, tol: f64, random_state: u64) -> PyResult<Self> {
        Ok(Self {
            inner: RustKMeans::new(n_clusters, max_iter, tol, random_state).map_err(to_py_err)?,
        })
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner.fit(x.as_array()).map_err(to_py_err)?;
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

    fn fit_predict<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self
            .inner
            .fit_predict(x.as_array())
            .map_err(to_py_err)?
            .into_pyarray(py))
    }

    #[getter]
    fn cluster_centers_<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.inner
            .centroids()
            .map(|c| c.to_owned().into_pyarray(py))
    }
    #[getter]
    fn labels_<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .labels()
            .map(|l| l.mapv(|v| v as f64).into_pyarray(py))
    }
    #[getter]
    fn inertia_(&self) -> f64 {
        self.inner.inertia()
    }
    fn __repr__(&self) -> String {
        "KMeans()".into()
    }
}
