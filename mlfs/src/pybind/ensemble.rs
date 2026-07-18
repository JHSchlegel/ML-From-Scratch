//! Python wrappers for ensemble methods.

use super::{accuracy, make_tree_params, r2, to_py_err};
use crate::common::traits::{Estimator, Predictor};
use crate::ensemble::{
    AdaBoostClassifier as RustAda, GradientBoostingClassifier as RustGbc,
    GradientBoostingRegressor as RustGbr, RandomForestClassifier as RustRfc,
    RandomForestRegressor as RustRfr,
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Random forest classifier.
#[pyclass(name = "RandomForestClassifier", module = "mlfs")]
pub struct RandomForestClassifier {
    inner: RustRfc,
}

#[pymethods]
impl RandomForestClassifier {
    #[new]
    #[pyo3(signature = (
        n_estimators = 100,
        max_depth = None,
        min_samples_split = 2,
        min_samples_leaf = 1,
        max_features = None,
        criterion = "gini".to_string(),
        random_state = 0
    ))]
    fn new(
        n_estimators: usize,
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
        max_features: Option<usize>,
        criterion: String,
        random_state: u64,
    ) -> Self {
        let params = make_tree_params(max_depth, min_samples_split, min_samples_leaf, max_features);
        Self {
            inner: RustRfc::new(n_estimators, params, criterion == "entropy", random_state),
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
        "RandomForestClassifier()".into()
    }
}

/// Random forest regressor.
#[pyclass(name = "RandomForestRegressor", module = "mlfs")]
pub struct RandomForestRegressor {
    inner: RustRfr,
}

#[pymethods]
impl RandomForestRegressor {
    #[new]
    #[pyo3(signature = (
        n_estimators = 100,
        max_depth = None,
        min_samples_split = 2,
        min_samples_leaf = 1,
        max_features = None,
        random_state = 0
    ))]
    fn new(
        n_estimators: usize,
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
        max_features: Option<usize>,
        random_state: u64,
    ) -> Self {
        let params = make_tree_params(max_depth, min_samples_split, min_samples_leaf, max_features);
        Self {
            inner: RustRfr::new(n_estimators, params, random_state),
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
        "RandomForestRegressor()".into()
    }
}

/// Gradient boosting classifier.
#[pyclass(name = "GradientBoostingClassifier", module = "mlfs")]
pub struct GradientBoostingClassifier {
    inner: RustGbc,
}

#[pymethods]
impl GradientBoostingClassifier {
    #[new]
    #[pyo3(signature = (
        n_estimators = 100,
        learning_rate = 0.1,
        max_depth = 3,
        min_samples_split = 2,
        min_samples_leaf = 1,
        random_state = 0
    ))]
    fn new(
        n_estimators: usize,
        learning_rate: f64,
        max_depth: usize,
        min_samples_split: usize,
        min_samples_leaf: usize,
        random_state: u64,
    ) -> Self {
        let params = make_tree_params(Some(max_depth), min_samples_split, min_samples_leaf, None);
        Self {
            inner: RustGbc::new(n_estimators, learning_rate, params, random_state),
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
        "GradientBoostingClassifier()".into()
    }
}

/// Gradient boosting regressor.
#[pyclass(name = "GradientBoostingRegressor", module = "mlfs")]
pub struct GradientBoostingRegressor {
    inner: RustGbr,
}

#[pymethods]
impl GradientBoostingRegressor {
    #[new]
    #[pyo3(signature = (
        n_estimators = 100,
        learning_rate = 0.1,
        max_depth = 3,
        min_samples_split = 2,
        min_samples_leaf = 1,
        random_state = 0
    ))]
    fn new(
        n_estimators: usize,
        learning_rate: f64,
        max_depth: usize,
        min_samples_split: usize,
        min_samples_leaf: usize,
        random_state: u64,
    ) -> Self {
        let params = make_tree_params(Some(max_depth), min_samples_split, min_samples_leaf, None);
        Self {
            inner: RustGbr::new(n_estimators, learning_rate, params, random_state),
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
        "GradientBoostingRegressor()".into()
    }
}

/// AdaBoost classifier (SAMME).
#[pyclass(name = "AdaBoostClassifier", module = "mlfs")]
pub struct AdaBoostClassifier {
    inner: RustAda,
}

#[pymethods]
impl AdaBoostClassifier {
    #[new]
    #[pyo3(signature = (
        n_estimators = 50,
        learning_rate = 1.0,
        max_depth = 1,
        random_state = 0
    ))]
    fn new(n_estimators: usize, learning_rate: f64, max_depth: usize, random_state: u64) -> Self {
        let params = make_tree_params(Some(max_depth), 2, 1, None);
        Self {
            inner: RustAda::new(n_estimators, learning_rate, params, random_state),
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
        "AdaBoostClassifier()".into()
    }
}
