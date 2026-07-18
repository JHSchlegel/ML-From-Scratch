//! Multinomial logistic regression trained by gradient descent.

use crate::common::labels::LabelEncoder;
use crate::common::traits::{Estimator, Predictor};
use crate::common::validation::{check_n_features, check_xy};
use crate::error::{MlError, Result};
use ndarray::{Array1, Array2, ArrayView2, Axis};

/// Softmax (multinomial) logistic regression. Works for any number of classes;
/// binary problems are just the 2-class case. Optimised with full-batch
/// gradient descent and optional L2 regularisation.
#[derive(Debug, Clone)]
pub struct LogisticRegression {
    lr: f64,
    max_iter: usize,
    l2: f64,
    encoder: Option<LabelEncoder>,
    /// Weights, shape (n_classes, n_features).
    weights: Option<Array2<f64>>,
    /// Biases, length n_classes.
    bias: Option<Array1<f64>>,
}

/// Row-wise numerically stable softmax.
fn softmax_rows(z: &Array2<f64>) -> Array2<f64> {
    let mut out = z.clone();
    for mut row in out.rows_mut() {
        let max = row.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        row.mapv_inplace(|v| (v - max).exp());
        let sum: f64 = row.sum();
        if sum > 0.0 {
            row /= sum;
        }
    }
    out
}

impl LogisticRegression {
    pub fn new(lr: f64, max_iter: usize, l2: f64) -> Result<Self> {
        if lr <= 0.0 {
            return Err(MlError::InvalidParameter("lr must be > 0".into()));
        }
        Ok(Self {
            lr,
            max_iter,
            l2,
            encoder: None,
            weights: None,
            bias: None,
        })
    }

    pub fn classes(&self) -> Option<&[f64]> {
        self.encoder.as_ref().map(|e| e.classes())
    }

    /// Class-probability matrix, shape (n_samples, n_classes).
    pub fn predict_proba(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        let w = self
            .weights
            .as_ref()
            .ok_or_else(|| MlError::NotFitted("LogisticRegression".into()))?;
        let b = self.bias.as_ref().unwrap();
        check_n_features(x, w.ncols())?;
        // Z = X W^T + b
        let mut z = x.dot(&w.t());
        z += &b.view().insert_axis(Axis(0));
        Ok(softmax_rows(&z))
    }
}

impl Estimator for LogisticRegression {
    fn fit(&mut self, x: ArrayView2<f64>, y: ndarray::ArrayView1<f64>) -> Result<()> {
        check_xy(x, y)?;
        let encoder = LabelEncoder::fit(y)?;
        let k = encoder.n_classes();
        let (n, p) = (x.nrows(), x.ncols());
        let y_idx = encoder.encode(y)?;

        // One-hot targets.
        let mut y_oh = Array2::<f64>::zeros((n, k));
        for (i, &c) in y_idx.iter().enumerate() {
            y_oh[[i, c]] = 1.0;
        }

        let mut w = Array2::<f64>::zeros((k, p));
        let mut b = Array1::<f64>::zeros(k);
        let inv_n = 1.0 / n as f64;

        for _ in 0..self.max_iter {
            // Forward: P = softmax(X W^T + b)
            let mut z = x.dot(&w.t());
            z += &b.view().insert_axis(Axis(0));
            let p = softmax_rows(&z);
            // Gradient of cross-entropy: G = P - Y  (n x k)
            let g = &p - &y_oh;
            // grad_w = (1/n) G^T X + l2 * W ; grad_b = (1/n) sum_i G_i
            let mut grad_w = g.t().dot(&x) * inv_n;
            if self.l2 > 0.0 {
                grad_w = grad_w + &(&w * self.l2);
            }
            let grad_b = g.sum_axis(Axis(0)) * inv_n;
            w.scaled_add(-self.lr, &grad_w);
            b.scaled_add(-self.lr, &grad_b);
        }

        self.encoder = Some(encoder);
        self.weights = Some(w);
        self.bias = Some(b);
        Ok(())
    }
}

impl Predictor for LogisticRegression {
    fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        let proba = self.predict_proba(x)?;
        let encoder = self.encoder.as_ref().unwrap();
        let idx: Vec<usize> = proba
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
    fn separates_two_clusters() {
        // Two linearly separable blobs.
        let x = array![
            [-2.0, -2.0],
            [-2.1, -1.8],
            [-1.8, -2.2],
            [2.0, 2.0],
            [2.2, 1.9],
            [1.8, 2.1]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut model = LogisticRegression::new(0.5, 500, 0.0).unwrap();
        model.fit(x.view(), y.view()).unwrap();
        let pred = model.predict(x.view()).unwrap();
        assert_eq!(accuracy_score(y.view(), pred.view()), 1.0);
    }
}
