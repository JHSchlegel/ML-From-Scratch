//! Gradient boosting with regression-tree base learners.
//!
//! Regression uses squared-error loss (fit residuals). Classification uses the
//! multinomial (softmax) deviance: each boosting round fits one regression tree
//! per class to the negative gradient, which reduces to the familiar
//! logistic-loss boosting in the binary case.

use crate::common::labels::LabelEncoder;
use crate::common::traits::{Estimator, Predictor};
use crate::common::validation::{check_n_features, check_xy};
use crate::error::{MlError, Result};
use crate::tree::{DecisionTree, Task, TreeParams};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::rngs::StdRng;
use rand::SeedableRng;

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

/// Gradient boosting regressor (squared-error loss).
#[derive(Debug, Clone)]
pub struct GradientBoostingRegressor {
    n_estimators: usize,
    lr: f64,
    params: TreeParams,
    seed: u64,
    init: f64,
    trees: Vec<DecisionTree>,
    n_features: usize,
}

impl GradientBoostingRegressor {
    pub fn new(n_estimators: usize, lr: f64, params: TreeParams, seed: u64) -> Self {
        Self {
            n_estimators,
            lr,
            params,
            seed,
            init: 0.0,
            trees: Vec::new(),
            n_features: 0,
        }
    }
}

impl Estimator for GradientBoostingRegressor {
    fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<()> {
        check_xy(x, y)?;
        self.n_features = x.ncols();
        self.init = y.mean().unwrap();
        let mut f = Array1::<f64>::from_elem(x.nrows(), self.init);
        self.trees.clear();
        for m in 0..self.n_estimators {
            let residual = &y - &f; // negative gradient of 1/2 (y-F)^2
            let mut tree = DecisionTree::new(Task::Regression, self.params);
            let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(m as u64));
            tree.fit(x, residual.view(), &mut rng);
            let update = tree.predict(x);
            f.scaled_add(self.lr, &update);
            self.trees.push(tree);
        }
        Ok(())
    }
}

impl Predictor for GradientBoostingRegressor {
    fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        if self.trees.is_empty() {
            return Err(MlError::NotFitted("GradientBoostingRegressor".into()));
        }
        check_n_features(x, self.n_features)?;
        let mut f = Array1::<f64>::from_elem(x.nrows(), self.init);
        for tree in &self.trees {
            f.scaled_add(self.lr, &tree.predict(x));
        }
        Ok(f)
    }
}

/// Gradient boosting classifier (multinomial deviance).
#[derive(Debug, Clone)]
pub struct GradientBoostingClassifier {
    n_estimators: usize,
    lr: f64,
    params: TreeParams,
    seed: u64,
    encoder: Option<LabelEncoder>,
    init: Array1<f64>,
    /// `rounds[m][k]` = tree for class k at boosting round m.
    rounds: Vec<Vec<DecisionTree>>,
    n_features: usize,
}

impl GradientBoostingClassifier {
    pub fn new(n_estimators: usize, lr: f64, params: TreeParams, seed: u64) -> Self {
        Self {
            n_estimators,
            lr,
            params,
            seed,
            encoder: None,
            init: Array1::zeros(0),
            rounds: Vec::new(),
            n_features: 0,
        }
    }

    pub fn classes(&self) -> Option<&[f64]> {
        self.encoder.as_ref().map(|e| e.classes())
    }

    fn decision_function(&self, x: ArrayView2<f64>) -> Array2<f64> {
        let k = self.init.len();
        let mut f = Array2::<f64>::zeros((x.nrows(), k));
        // Broadcast init across rows.
        for mut row in f.rows_mut() {
            row.assign(&self.init);
        }
        for round in &self.rounds {
            for (c, tree) in round.iter().enumerate() {
                let upd = tree.predict(x);
                let mut col = f.column_mut(c);
                col.scaled_add(self.lr, &upd);
            }
        }
        f
    }
}

impl Estimator for GradientBoostingClassifier {
    fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<()> {
        check_xy(x, y)?;
        let encoder = LabelEncoder::fit(y)?;
        let k = encoder.n_classes();
        let n = x.nrows();
        self.n_features = x.ncols();
        let y_idx = encoder.encode(y)?;

        // One-hot targets and log-prior initialisation.
        let mut y_oh = Array2::<f64>::zeros((n, k));
        let mut counts = vec![0.0f64; k];
        for (i, &c) in y_idx.iter().enumerate() {
            y_oh[[i, c]] = 1.0;
            counts[c] += 1.0;
        }
        self.init = Array1::from_iter(counts.iter().map(|&c| (c / n as f64).max(1e-6).ln()));

        let mut f = Array2::<f64>::zeros((n, k));
        for mut row in f.rows_mut() {
            row.assign(&self.init);
        }

        self.rounds.clear();
        for m in 0..self.n_estimators {
            let p = softmax_rows(&f);
            let mut round = Vec::with_capacity(k);
            for c in 0..k {
                // Negative gradient of multinomial deviance = y_onehot - p.
                let grad = &y_oh.column(c) - &p.column(c);
                let mut tree = DecisionTree::new(Task::Regression, self.params);
                let mut rng = StdRng::seed_from_u64(
                    self.seed
                        .wrapping_add((m * k + c) as u64)
                        .wrapping_mul(2654435761),
                );
                tree.fit(x, grad.view(), &mut rng);
                let upd = tree.predict(x);
                f.column_mut(c).scaled_add(self.lr, &upd);
                round.push(tree);
            }
            self.rounds.push(round);
        }
        self.encoder = Some(encoder);
        Ok(())
    }
}

impl Predictor for GradientBoostingClassifier {
    fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        if self.rounds.is_empty() {
            return Err(MlError::NotFitted("GradientBoostingClassifier".into()));
        }
        check_n_features(x, self.n_features)?;
        let f = self.decision_function(x);
        let idx: Vec<usize> = f
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
        Ok(self.encoder.as_ref().unwrap().decode(&idx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::metrics::{accuracy_score, r2_score};
    use ndarray::{array, Array, Array2};
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn boosting_regressor_learns_nonlinear() {
        let x: Array2<f64> = Array::random((100, 1), Uniform::new(-3.0, 3.0));
        let y = x.column(0).mapv(|v| v * v); // parabola
        let params = TreeParams {
            max_depth: 3,
            ..Default::default()
        };
        let mut m = GradientBoostingRegressor::new(100, 0.1, params, 0);
        m.fit(x.view(), y.view()).unwrap();
        let pred = m.predict(x.view()).unwrap();
        assert!(r2_score(y.view(), pred.view()) > 0.9);
    }

    #[test]
    fn boosting_classifier_separates() {
        let x = array![
            [0.0, 0.0],
            [0.2, 0.1],
            [0.1, 0.3],
            [5.0, 5.0],
            [5.2, 4.8],
            [4.9, 5.1]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let params = TreeParams {
            max_depth: 2,
            ..Default::default()
        };
        let mut m = GradientBoostingClassifier::new(50, 0.2, params, 0);
        m.fit(x.view(), y.view()).unwrap();
        let pred = m.predict(x.view()).unwrap();
        assert_eq!(accuracy_score(y.view(), pred.view()), 1.0);
    }
}
