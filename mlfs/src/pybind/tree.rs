//! Python wrappers for decision trees.

use super::{accuracy, make_tree_params, r2, to_py_err};
use crate::common::traits::{Estimator, Predictor};
use crate::tree::{DecisionTreeClassifier as RustDtc, DecisionTreeRegressor as RustDtr};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// CART decision tree classifier.
#[pyclass(name = "DecisionTreeClassifier", module = "mlfs")]
pub struct DecisionTreeClassifier {
    inner: RustDtc,
}

#[pymethods]
impl DecisionTreeClassifier {
    #[new]
    #[pyo3(signature = (
        max_depth = None,
        min_samples_split = 2,
        min_samples_leaf = 1,
        criterion = "gini".to_string(),
        random_state = 0
    ))]
    fn new(
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
        criterion: String,
        random_state: u64,
    ) -> Self {
        let params = make_tree_params(max_depth, min_samples_split, min_samples_leaf, None);
        let entropy = criterion == "entropy";
        Self {
            inner: RustDtc::new(params, entropy, random_state),
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
        "DecisionTreeClassifier()".into()
    }
}

/// CART decision tree regressor.
#[pyclass(name = "DecisionTreeRegressor", module = "mlfs")]
pub struct DecisionTreeRegressor {
    inner: RustDtr,
}

#[pymethods]
impl DecisionTreeRegressor {
    #[new]
    #[pyo3(signature = (
        max_depth = None,
        min_samples_split = 2,
        min_samples_leaf = 1,
        random_state = 0
    ))]
    fn new(
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
        random_state: u64,
    ) -> Self {
        let params = make_tree_params(max_depth, min_samples_split, min_samples_leaf, None);
        Self {
            inner: RustDtr::new(params, random_state),
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
    fn __repr__(&self) -> String {
        "DecisionTreeRegressor()".into()
    }
}
