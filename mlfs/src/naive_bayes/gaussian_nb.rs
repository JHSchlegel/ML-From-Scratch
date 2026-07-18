//! Gaussian Naive Bayes classifier.

use crate::common::labels::LabelEncoder;
use crate::common::traits::{Estimator, Predictor};
use crate::common::validation::{check_n_features, check_xy};
use crate::error::{MlError, Result};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

/// Gaussian Naive Bayes: assumes each feature is class-conditionally normal and
/// independent. Fitting is a single pass estimating per-class feature means and
/// variances; prediction maximises the log-posterior.
#[derive(Debug, Clone)]
pub struct GaussianNB {
    var_smoothing: f64,
    encoder: Option<LabelEncoder>,
    /// theta[c, j] = mean of feature j for class c.
    theta: Option<Array2<f64>>,
    /// var[c, j] = variance of feature j for class c.
    var: Option<Array2<f64>>,
    log_prior: Option<Array1<f64>>,
}

impl GaussianNB {
    pub fn new(var_smoothing: f64) -> Self {
        Self {
            var_smoothing,
            encoder: None,
            theta: None,
            var: None,
            log_prior: None,
        }
    }

    pub fn classes(&self) -> Option<&[f64]> {
        self.encoder.as_ref().map(|e| e.classes())
    }

    /// Joint log-likelihood per class, shape (n_samples, n_classes).
    fn joint_log_likelihood(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        let theta = self
            .theta
            .as_ref()
            .ok_or_else(|| MlError::NotFitted("GaussianNB".into()))?;
        let var = self.var.as_ref().unwrap();
        let log_prior = self.log_prior.as_ref().unwrap();
        check_n_features(x, theta.ncols())?;
        let k = theta.nrows();

        let mut jll = Array2::<f64>::zeros((x.nrows(), k));
        for (i, row) in x.rows().into_iter().enumerate() {
            for c in 0..k {
                // log N(x | mu, var) summed over independent features.
                let mut ll = log_prior[c];
                for j in 0..row.len() {
                    let v = var[[c, j]];
                    let diff = row[j] - theta[[c, j]];
                    ll += -0.5 * ((2.0 * std::f64::consts::PI * v).ln() + diff * diff / v);
                }
                jll[[i, c]] = ll;
            }
        }
        Ok(jll)
    }
}

impl Estimator for GaussianNB {
    fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<()> {
        check_xy(x, y)?;
        let encoder = LabelEncoder::fit(y)?;
        let k = encoder.n_classes();
        let p = x.ncols();
        let y_idx = encoder.encode(y)?;

        let mut theta = Array2::<f64>::zeros((k, p));
        let mut var = Array2::<f64>::zeros((k, p));
        let mut log_prior = Array1::<f64>::zeros(k);
        let n = x.nrows() as f64;

        // Global variance floor for numerical stability (sklearn convention).
        let global_var = x.var_axis(Axis(0), 0.0);
        let epsilon = self.var_smoothing * global_var.fold(0.0_f64, |a, &b| a.max(b));

        for c in 0..k {
            let rows: Vec<usize> = y_idx
                .iter()
                .enumerate()
                .filter(|&(_, &ci)| ci == c)
                .map(|(i, _)| i)
                .collect();
            let count = rows.len();
            let sub = x.select(Axis(0), &rows);
            let mean = sub.mean_axis(Axis(0)).unwrap();
            let mut v = sub.var_axis(Axis(0), 0.0);
            v.mapv_inplace(|val| val + epsilon);
            theta.row_mut(c).assign(&mean);
            var.row_mut(c).assign(&v);
            log_prior[c] = (count as f64 / n).ln();
        }

        self.encoder = Some(encoder);
        self.theta = Some(theta);
        self.var = Some(var);
        self.log_prior = Some(log_prior);
        Ok(())
    }
}

impl Predictor for GaussianNB {
    fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        let jll = self.joint_log_likelihood(x)?;
        let encoder = self.encoder.as_ref().unwrap();
        let idx: Vec<usize> = jll
            .rows()
            .into_iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap()
            })
            .collect();
        Ok(encoder.decode(&idx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::metrics::accuracy_score;
    use ndarray::array;

    #[test]
    fn gaussian_nb_separates_classes() {
        let x = array![
            [1.0, 1.0],
            [1.2, 0.8],
            [0.9, 1.1],
            [5.0, 5.0],
            [5.1, 4.8],
            [4.9, 5.2]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut m = GaussianNB::new(1e-9);
        m.fit(x.view(), y.view()).unwrap();
        let pred = m.predict(x.view()).unwrap();
        assert_eq!(accuracy_score(y.view(), pred.view()), 1.0);
    }
}
