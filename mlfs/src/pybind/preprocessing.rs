//! Python wrappers for preprocessing utilities.

use super::to_py_err;
use crate::common::preprocessing::{train_test_split_indices, StandardScaler as RustScaler};
use crate::common::traits::Transformer;
use ndarray::Axis;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Standardise features to zero mean and unit variance.
#[pyclass(name = "StandardScaler", module = "mlfs")]
pub struct StandardScaler {
    inner: RustScaler,
}

#[pymethods]
impl StandardScaler {
    #[new]
    fn new() -> Self {
        Self {
            inner: RustScaler::new(),
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
    fn __repr__(&self) -> String {
        "StandardScaler()".into()
    }
}

/// Split `X` and `y` into random train and test subsets.
///
/// Returns `(X_train, X_test, y_train, y_test)`.
#[pyfunction]
#[pyo3(signature = (x, y, test_size = 0.25, random_state = 0))]
pub fn train_test_split<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    test_size: f64,
    random_state: u64,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let x = x.as_array();
    let y = y.as_array();
    let (train, test) =
        train_test_split_indices(x.nrows(), test_size, random_state).map_err(to_py_err)?;

    let x_train = x.select(Axis(0), &train);
    let x_test = x.select(Axis(0), &test);
    let y_train = y.select(Axis(0), &train);
    let y_test = y.select(Axis(0), &test);

    Ok((
        x_train.into_pyarray(py),
        x_test.into_pyarray(py),
        y_train.into_pyarray(py),
        y_test.into_pyarray(py),
    ))
}
