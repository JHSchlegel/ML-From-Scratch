//! Python wrappers for clustering algorithms.

use super::to_py_err;
use crate::cluster::{
    AgglomerativeClustering as RustAgg, GaussianMixture as RustGmm, KMeans as RustKMeans, Linkage,
    DBSCAN as RustDbscan,
};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
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

/// Gaussian Mixture Model.
#[pyclass(name = "GaussianMixture", module = "mlfs")]
pub struct GaussianMixture {
    inner: RustGmm,
}

#[pymethods]
impl GaussianMixture {
    #[new]
    #[pyo3(signature = (
        n_components = 1,
        max_iter = 100,
        tol = 1e-3,
        reg_covar = 1e-6,
        random_state = 0
    ))]
    fn new(
        n_components: usize,
        max_iter: usize,
        tol: f64,
        reg_covar: f64,
        random_state: u64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: RustGmm::new(n_components, max_iter, tol, reg_covar, random_state)
                .map_err(to_py_err)?,
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
    fn means_<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.inner.means().map(|c| c.to_owned().into_pyarray(py))
    }
    #[getter]
    fn weights_<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.weights().map(|c| c.to_owned().into_pyarray(py))
    }
    fn __repr__(&self) -> String {
        "GaussianMixture()".into()
    }
}

/// DBSCAN density-based clustering.
#[pyclass(name = "DBSCAN", module = "mlfs")]
pub struct DBSCAN {
    inner: RustDbscan,
}

#[pymethods]
impl DBSCAN {
    #[new]
    #[pyo3(signature = (eps = 0.5, min_samples = 5))]
    fn new(eps: f64, min_samples: usize) -> PyResult<Self> {
        Ok(Self {
            inner: RustDbscan::new(eps, min_samples).map_err(to_py_err)?,
        })
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner.fit(x.as_array()).map_err(to_py_err)?;
        Ok(slf)
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
    fn labels_<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .labels()
            .map(|l| l.mapv(|v| v as f64).into_pyarray(py))
    }
    fn __repr__(&self) -> String {
        "DBSCAN()".into()
    }
}

/// Agglomerative (hierarchical) clustering.
#[pyclass(name = "AgglomerativeClustering", module = "mlfs")]
pub struct AgglomerativeClustering {
    inner: RustAgg,
}

#[pymethods]
impl AgglomerativeClustering {
    #[new]
    #[pyo3(signature = (n_clusters = 2, linkage = "average".to_string()))]
    fn new(n_clusters: usize, linkage: String) -> PyResult<Self> {
        let link = match linkage.as_str() {
            "single" => Linkage::Single,
            "complete" => Linkage::Complete,
            "average" => Linkage::Average,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown linkage '{other}', expected single/complete/average"
                )))
            }
        };
        Ok(Self {
            inner: RustAgg::new(n_clusters, link).map_err(to_py_err)?,
        })
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner.fit(x.as_array()).map_err(to_py_err)?;
        Ok(slf)
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
    fn labels_<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .labels()
            .map(|l| l.mapv(|v| v as f64).into_pyarray(py))
    }
    fn __repr__(&self) -> String {
        "AgglomerativeClustering()".into()
    }
}
