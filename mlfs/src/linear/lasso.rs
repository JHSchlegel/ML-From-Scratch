//! Lasso regression (L1-penalised least squares) via coordinate descent.

use crate::common::traits::{Estimator, Predictor};
use crate::common::validation::{check_n_features, check_xy};
use crate::error::{MlError, Result};
use ndarray::{Array1, ArrayView1, ArrayView2, Axis};

/// Lasso regression: minimises `(1/2n)||X beta - y||^2 + alpha * ||beta||_1`.
///
/// Fit with cyclic coordinate descent and soft-thresholding — the standard
/// algorithm behind sklearn's `Lasso`. L1 drives some coefficients exactly to
/// zero, giving a sparse model.
#[derive(Debug, Clone)]
pub struct Lasso {
    alpha: f64,
    fit_intercept: bool,
    max_iter: usize,
    tol: f64,
    coef: Option<Array1<f64>>,
    intercept: f64,
}

fn soft_threshold(z: f64, gamma: f64) -> f64 {
    if z > gamma {
        z - gamma
    } else if z < -gamma {
        z + gamma
    } else {
        0.0
    }
}

impl Lasso {
    pub fn new(alpha: f64, fit_intercept: bool, max_iter: usize, tol: f64) -> Result<Self> {
        if alpha < 0.0 {
            return Err(MlError::InvalidParameter("alpha must be >= 0".into()));
        }
        Ok(Self {
            alpha,
            fit_intercept,
            max_iter,
            tol,
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
}

impl Estimator for Lasso {
    fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<()> {
        check_xy(x, y)?;
        let n = x.nrows();
        let p = x.ncols();

        // Center for the intercept (data is not scaled here, matching sklearn's
        // default of operating on the given feature scale).
        let (xc, yc, x_mean, y_mean) = if self.fit_intercept {
            let x_mean = x.mean_axis(Axis(0)).unwrap();
            let y_mean = y.mean().unwrap();
            (&x - &x_mean, &y - y_mean, x_mean, y_mean)
        } else {
            (x.to_owned(), y.to_owned(), Array1::zeros(p), 0.0)
        };

        // Precompute per-feature squared norms (the coordinate-wise curvature).
        let col_sq: Array1<f64> = xc.map_axis(Axis(0), |col| col.dot(&col));

        let mut beta = Array1::<f64>::zeros(p);
        // Residual r = yc - Xc @ beta (starts at yc since beta = 0).
        let mut r = yc.clone();

        for _ in 0..self.max_iter {
            let mut max_delta = 0.0_f64;
            for j in 0..p {
                if col_sq[j] < 1e-12 {
                    continue;
                }
                let x_j = xc.column(j);
                // rho = X_j . (r + beta_j X_j)  (add back this feature's contribution)
                let rho = x_j.dot(&r) + beta[j] * col_sq[j];
                let new_bj = soft_threshold(rho / n as f64, self.alpha) / (col_sq[j] / n as f64);
                let delta = new_bj - beta[j];
                if delta != 0.0 {
                    // r -= delta * X_j
                    r.scaled_add(-delta, &x_j);
                    beta[j] = new_bj;
                    max_delta = max_delta.max(delta.abs());
                }
            }
            if max_delta < self.tol {
                break;
            }
        }

        self.intercept = if self.fit_intercept {
            y_mean - x_mean.dot(&beta)
        } else {
            0.0
        };
        self.coef = Some(beta);
        Ok(())
    }
}

impl Predictor for Lasso {
    fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        let coef = self
            .coef
            .as_ref()
            .ok_or_else(|| MlError::NotFitted("Lasso".into()))?;
        check_n_features(x, coef.len())?;
        Ok(x.dot(coef) + self.intercept)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, Array2};
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn lasso_zeroes_irrelevant_features() {
        // y depends only on feature 0; features 1..4 are noise.
        let x: Array2<f64> = Array::random((80, 5), Uniform::new(-1.0, 1.0));
        let y = x.column(0).mapv(|v| 3.0 * v) + 1.0;

        let mut model = Lasso::new(0.05, true, 1000, 1e-7).unwrap();
        model.fit(x.view(), y.view()).unwrap();
        let coef = model.coef().unwrap();

        assert!(coef[0].abs() > 1.0, "relevant feature kept");
        for j in 1..5 {
            assert!(coef[j].abs() < 0.5, "noise feature {j} shrunk toward 0");
        }
    }
}
