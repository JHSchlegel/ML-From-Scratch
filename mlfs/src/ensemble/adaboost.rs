//! AdaBoost classifier using the SAMME algorithm (multi-class).
//!
//! The base learners are shallow decision trees (stumps by default). Because the
//! core tree does not take sample weights, each round trains on a bootstrap
//! sample drawn *according to the current sample weights* — the standard
//! "boosting by resampling" formulation of AdaBoost.

use crate::common::labels::LabelEncoder;
use crate::common::traits::{Estimator, Predictor};
use crate::common::validation::{check_n_features, check_xy};
use crate::error::{MlError, Result};
use crate::tree::{DecisionTree, Task, TreeParams};
use ndarray::{Array1, ArrayView1, ArrayView2, Axis};
use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Multi-class AdaBoost (SAMME).
#[derive(Debug, Clone)]
pub struct AdaBoostClassifier {
    n_estimators: usize,
    lr: f64,
    params: TreeParams,
    seed: u64,
    encoder: Option<LabelEncoder>,
    estimators: Vec<(DecisionTree, f64)>, // (stump, alpha)
    n_classes: usize,
    n_features: usize,
}

impl AdaBoostClassifier {
    pub fn new(n_estimators: usize, lr: f64, params: TreeParams, seed: u64) -> Self {
        Self {
            n_estimators,
            lr,
            params,
            seed,
            encoder: None,
            estimators: Vec::new(),
            n_classes: 0,
            n_features: 0,
        }
    }

    pub fn classes(&self) -> Option<&[f64]> {
        self.encoder.as_ref().map(|e| e.classes())
    }
}

impl Estimator for AdaBoostClassifier {
    fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<()> {
        check_xy(x, y)?;
        let encoder = LabelEncoder::fit(y)?;
        let k = encoder.n_classes();
        let n = x.nrows();
        self.n_classes = k;
        self.n_features = x.ncols();
        let y_idx = encoder.encode(y)?; // Array1<usize>
        let y_f = y_idx.mapv(|v| v as f64);

        let mut w = Array1::<f64>::from_elem(n, 1.0 / n as f64);
        self.estimators.clear();

        for m in 0..self.n_estimators {
            let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(m as u64));
            // Resample indices according to weights.
            let dist = WeightedIndex::new(w.iter().cloned())
                .map_err(|e| MlError::Numerical(format!("weighting failed: {e}")))?;
            let idx: Vec<usize> = (0..n).map(|_| dist.sample(&mut rng)).collect();
            let xb = x.select(Axis(0), &idx);
            let yb = y_f.select(Axis(0), &idx);

            let mut tree = DecisionTree::new(
                Task::Classification {
                    n_classes: k,
                    entropy: false,
                },
                self.params,
            );
            tree.fit(xb.view(), yb.view(), &mut rng);

            let pred = tree.predict(x);
            let incorrect: Array1<f64> = pred
                .iter()
                .zip(y_f.iter())
                .map(|(p, t)| if (p - t).abs() < 1e-9 { 0.0 } else { 1.0 })
                .collect();

            let err = (&w * &incorrect).sum() / w.sum();
            // Stop if the learner is no better than random guessing.
            if err >= 1.0 - 1.0 / k as f64 {
                break;
            }
            if err <= 1e-10 {
                // Perfect learner: give it dominant weight and stop.
                self.estimators.push((tree, 1.0));
                break;
            }

            let alpha = self.lr * ((1.0 - err) / err).ln() + (k as f64 - 1.0).ln();
            // Update and renormalise weights.
            for i in 0..n {
                w[i] *= (alpha * incorrect[i]).exp();
            }
            let s = w.sum();
            w /= s;

            self.estimators.push((tree, alpha));
        }

        if self.estimators.is_empty() {
            return Err(MlError::Numerical(
                "AdaBoost produced no usable estimators".into(),
            ));
        }
        self.encoder = Some(encoder);
        Ok(())
    }
}

impl Predictor for AdaBoostClassifier {
    fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        if self.estimators.is_empty() {
            return Err(MlError::NotFitted("AdaBoostClassifier".into()));
        }
        check_n_features(x, self.n_features)?;
        let n = x.nrows();
        // Accumulate weighted votes per class.
        let mut scores = ndarray::Array2::<f64>::zeros((n, self.n_classes));
        for (tree, alpha) in &self.estimators {
            let pred = tree.predict(x);
            for (i, &p) in pred.iter().enumerate() {
                scores[[i, p as usize]] += *alpha;
            }
        }
        let idx: Vec<usize> = scores
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
    use crate::common::metrics::accuracy_score;
    use ndarray::array;

    #[test]
    fn adaboost_separates_blobs() {
        let x = array![
            [0.0, 0.0],
            [0.3, 0.1],
            [0.1, 0.2],
            [0.2, 0.3],
            [5.0, 5.0],
            [5.3, 4.9],
            [4.8, 5.2],
            [5.1, 4.7]
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let params = TreeParams {
            max_depth: 1,
            ..Default::default()
        };
        let mut m = AdaBoostClassifier::new(50, 1.0, params, 0);
        m.fit(x.view(), y.view()).unwrap();
        let pred = m.predict(x.view()).unwrap();
        assert_eq!(accuracy_score(y.view(), pred.view()), 1.0);
    }
}
