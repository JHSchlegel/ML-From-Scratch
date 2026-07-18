//! Ordinary least squares linear regression.

use crate::common::linalg::lstsq;
use crate::common::traits::{Estimator, Predictor};
use crate::common::validation::{check_n_features, check_xy};
use crate::error::{MlError, Result};
use ndarray::{Array1, ArrayView1, ArrayView2, Axis};

/// Ordinary least squares linear regression: fits `y ~ X @ coef_ + intercept_`
/// by minimising the residual sum of squares.
///
/// The fit is computed with an SVD-based least-squares solve (see
/// [`crate::common::linalg::lstsq`]), which is numerically stable even when
/// features are collinear.
#[derive(Debug, Clone)]
pub struct LinearRegression {
    fit_intercept: bool,
    coef: Option<Array1<f64>>,
    intercept: f64,
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            coef: None,
            intercept: 0.0,
        }
    }
}

impl LinearRegression {
    /// Create a new model. `fit_intercept` controls whether a bias term is fit;
    /// when `false` the data is assumed already centered.
    pub fn new(fit_intercept: bool) -> Self {
        Self {
            fit_intercept,
            ..Default::default()
        }
    }

    /// Fitted coefficients, one per feature. `None` until [`Estimator::fit`] runs.
    pub fn coef(&self) -> Option<&Array1<f64>> {
        self.coef.as_ref()
    }

    /// Fitted intercept (bias) term.
    pub fn intercept(&self) -> f64 {
        self.intercept
    }
}

impl Estimator for LinearRegression {
    fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<()> {
        check_xy(x, y)?;

        if self.fit_intercept {
            // Center X and y so the intercept can be recovered afterwards.
            let x_mean = x.mean_axis(Axis(0)).unwrap();
            let y_mean = y.mean().unwrap();
            let xc = &x - &x_mean;
            let yc = &y - y_mean;
            let coef = lstsq(xc.view(), yc.view())?;
            self.intercept = y_mean - x_mean.dot(&coef);
            self.coef = Some(coef);
        } else {
            self.coef = Some(lstsq(x, y)?);
            self.intercept = 0.0;
        }
        Ok(())
    }
}

impl Predictor for LinearRegression {
    fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        let coef = self
            .coef
            .as_ref()
            .ok_or_else(|| MlError::NotFitted("LinearRegression".into()))?;
        check_n_features(x, coef.len())?;
        Ok(x.dot(coef) + self.intercept)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array2};

    #[test]
    fn recovers_known_linear_relationship() {
        // y = 3 + 2*x0 - 1*x1
        let x: Array2<f64> = array![[0., 0.], [1., 0.], [0., 1.], [1., 1.], [2., 1.], [3., 2.]];
        let y: Array1<f64> = x.column(0).mapv(|v| 2.0 * v) - x.column(1).mapv(|v| v) + 3.0;

        let mut model = LinearRegression::new(true);
        model.fit(x.view(), y.view()).unwrap();

        let coef = model.coef().unwrap();
        assert_abs_diff_eq!(coef[0], 2.0, epsilon = 1e-9);
        assert_abs_diff_eq!(coef[1], -1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(model.intercept(), 3.0, epsilon = 1e-9);

        let preds = model.predict(x.view()).unwrap();
        for (p, t) in preds.iter().zip(y.iter()) {
            assert_abs_diff_eq!(p, t, epsilon = 1e-9);
        }
    }
}
