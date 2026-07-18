//! Python wrappers for dimensionality reduction.

use super::to_py_err;
use crate::decomposition::{PCA as RustPca, TSNE as RustTsne};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

/// Principal Component Analysis.
#[pyclass(name = "PCA", module = "mlfs")]
pub struct PCA {
    inner: RustPca,
}

#[pymethods]
impl PCA {
    #[new]
    #[pyo3(signature = (n_components = 2))]
    fn new(n_components: usize) -> Self {
        Self {
            inner: RustPca::new(n_components),
        }
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner.fit(x.as_array()).map_err(to_py_err)?;
        Ok(slf)
    }

    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(self
            .inner
            .transform(x.as_array())
            .map_err(to_py_err)?
            .into_pyarray(py))
    }

    fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(self
            .inner
            .fit_transform(x.as_array())
            .map_err(to_py_err)?
            .into_pyarray(py))
    }

    #[getter]
    fn components_<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.inner
            .components()
            .map(|c| c.to_owned().into_pyarray(py))
    }
    #[getter]
    fn explained_variance_<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .explained_variance()
            .map(|c| c.to_owned().into_pyarray(py))
    }
    #[getter]
    fn explained_variance_ratio_<'py>(
        &self,
        py: Python<'py>,
    ) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .explained_variance_ratio()
            .map(|c| c.to_owned().into_pyarray(py))
    }
    fn __repr__(&self) -> String {
        "PCA()".into()
    }
}

/// t-SNE embedding.
#[pyclass(name = "TSNE", module = "mlfs")]
pub struct TSNE {
    inner: RustTsne,
}

#[pymethods]
impl TSNE {
    #[new]
    #[pyo3(signature = (
        n_components = 2,
        perplexity = 30.0,
        learning_rate = 200.0,
        n_iter = 1000,
        random_state = 0
    ))]
    fn new(
        n_components: usize,
        perplexity: f64,
        learning_rate: f64,
        n_iter: usize,
        random_state: u64,
    ) -> Self {
        Self {
            inner: RustTsne::new(n_components, perplexity, learning_rate, n_iter, random_state),
        }
    }

    fn fit_transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(self
            .inner
            .fit_transform(x.as_array())
            .map_err(to_py_err)?
            .into_pyarray(py))
    }
    fn __repr__(&self) -> String {
        "TSNE()".into()
    }
}
