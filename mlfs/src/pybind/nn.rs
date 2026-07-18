//! Python wrappers for the multi-layer perceptron.

use super::{accuracy, r2, to_py_err};
use crate::common::traits::{Estimator, Predictor};
use crate::nn::{Activation, MLPClassifier as RustMlpC, MLPRegressor as RustMlpR};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn parse_activation(s: &str) -> PyResult<Activation> {
    match s {
        "relu" => Ok(Activation::Relu),
        "tanh" => Ok(Activation::Tanh),
        other => Err(PyValueError::new_err(format!(
            "unknown activation '{other}', expected 'relu' or 'tanh'"
        ))),
    }
}

/// Multi-layer perceptron classifier.
#[pyclass(name = "MLPClassifier", module = "mlfs")]
pub struct MLPClassifier {
    inner: RustMlpC,
}

#[pymethods]
impl MLPClassifier {
    #[new]
    #[pyo3(signature = (
        hidden_layer_sizes = vec![64],
        activation = "relu".to_string(),
        learning_rate = 0.01,
        max_iter = 300,
        batch_size = 32,
        l2 = 0.0,
        random_state = 0
    ))]
    fn new(
        hidden_layer_sizes: Vec<usize>,
        activation: String,
        learning_rate: f64,
        max_iter: usize,
        batch_size: usize,
        l2: f64,
        random_state: u64,
    ) -> PyResult<Self> {
        let act = parse_activation(&activation)?;
        Ok(Self {
            inner: RustMlpC::new(
                hidden_layer_sizes,
                act,
                learning_rate,
                max_iter,
                batch_size,
                l2,
                random_state,
            ),
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
        "MLPClassifier()".into()
    }
}

/// Multi-layer perceptron regressor.
#[pyclass(name = "MLPRegressor", module = "mlfs")]
pub struct MLPRegressor {
    inner: RustMlpR,
}

#[pymethods]
impl MLPRegressor {
    #[new]
    #[pyo3(signature = (
        hidden_layer_sizes = vec![64],
        activation = "relu".to_string(),
        learning_rate = 0.01,
        max_iter = 300,
        batch_size = 32,
        l2 = 0.0,
        random_state = 0
    ))]
    fn new(
        hidden_layer_sizes: Vec<usize>,
        activation: String,
        learning_rate: f64,
        max_iter: usize,
        batch_size: usize,
        l2: f64,
        random_state: u64,
    ) -> PyResult<Self> {
        let act = parse_activation(&activation)?;
        Ok(Self {
            inner: RustMlpR::new(
                hidden_layer_sizes,
                act,
                learning_rate,
                max_iter,
                batch_size,
                l2,
                random_state,
            ),
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
        "MLPRegressor()".into()
    }
}
