//! Small input-validation helpers shared by estimators.

use crate::error::{MlError, Result};
use ndarray::{ArrayView1, ArrayView2};

/// Ensure `X` (n_samples, n_features) and `y` (n_samples) agree on sample count.
pub fn check_xy(x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<()> {
    if x.nrows() != y.len() {
        return Err(MlError::ShapeMismatch(format!(
            "X has {} samples but y has {}",
            x.nrows(),
            y.len()
        )));
    }
    if x.nrows() == 0 {
        return Err(MlError::InvalidParameter("X has no samples".into()));
    }
    Ok(())
}

/// Ensure a prediction-time `X` has the number of features the model expects.
pub fn check_n_features(x: ArrayView2<f64>, expected: usize) -> Result<()> {
    if x.ncols() != expected {
        return Err(MlError::ShapeMismatch(format!(
            "expected {expected} features, got {}",
            x.ncols()
        )));
    }
    Ok(())
}
