//! Ridge regression (L2-penalised least squares).

use crate::common::linalg::lstsq;
use crate::common::traits::{Estimator, Predictor};
use crate::common::validation::{check_n_features, check_xy};
use crate::error::{MlError, Result};
use ndarray::{concatenate, Array1, Array2, ArrayView1, ArrayView2, Axis};

/// Ridge regression: minimises `||X beta - y||^2 + alpha * ||beta||^2`.
///
/// Solved by augmenting the design matrix with `sqrt(alpha) * I` and running the
/// same stable least-squares solver used for OLS — so `alpha = 0` reduces to OLS.
#[derive(Debug, Clone)]
pub struct Ridge {
    alpha: f64,
    fit_intercept: bool,
    coef: Option<Array1<f64>>,
    intercept: f64,
}

impl Ridge {
    pub fn new(alpha: f64, fit_intercept: bool) -> Result<Self> {
        if alpha < 0.0 {
            return Err(MlError::InvalidParameter("alpha must be >= 0".into()));
        }
        Ok(Self {
            alpha,
            fit_intercept,
            coef: None,
            intercept: 0.0,
        })
    }

    pub fn coef(&self) -> Option<&Array1<f64>> {
        self.coef.as_ref()
    }
    pub fn intercept(&self) -> f64 {
        self.intercept
    }

    fn solve(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<Array1<f64>> {
        let p = x.ncols();
        // Augment: [X; sqrt(alpha) I] beta ~= [y; 0].
        let reg = Array2::<f64>::eye(p) * self.alpha.sqrt();
        let x_aug = concatenate![Axis(0), x, reg.view()];
        let mut y_aug = Array1::<f64>::zeros(x.nrows() + p);
        y_aug.slice_mut(ndarray::s![..x.nrows()]).assign(&y);
        lstsq(x_aug.view(), y_aug.view())
    }
}

impl Estimator for Ridge {
    fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<()> {
        check_xy(x, y)?;
        if self.fit_intercept {
            let x_mean = x.mean_axis(Axis(0)).unwrap();
            let y_mean = y.mean().unwrap();
            let xc = &x - &x_mean;
            let yc = &y - y_mean;
            let coef = self.solve(xc.view(), yc.view())?;
            self.intercept = y_mean - x_mean.dot(&coef);
            self.coef = Some(coef);
        } else {
            self.coef = Some(self.solve(x, y)?);
            self.intercept = 0.0;
        }
        Ok(())
    }
}

impl Predictor for Ridge {
    fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        let coef = self
            .coef
            .as_ref()
            .ok_or_else(|| MlError::NotFitted("Ridge".into()))?;
        check_n_features(x, coef.len())?;
        Ok(x.dot(coef) + self.intercept)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn ridge_shrinks_coefficients_toward_zero() {
        let x = array![[1.0, 1.0], [2.0, 1.5], [3.0, 2.0], [4.0, 3.1], [5.0, 4.0]];
        let y = array![2.0, 3.4, 5.1, 6.9, 8.2];

        let mut ols = Ridge::new(0.0, true).unwrap();
        ols.fit(x.view(), y.view()).unwrap();
        let mut ridge = Ridge::new(10.0, true).unwrap();
        ridge.fit(x.view(), y.view()).unwrap();

        let n_ols: f64 = ols.coef().unwrap().iter().map(|c| c * c).sum();
        let n_ridge: f64 = ridge.coef().unwrap().iter().map(|c| c * c).sum();
        assert!(n_ridge < n_ols, "ridge should shrink coefficients");
    }
}
