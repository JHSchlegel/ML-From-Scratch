//! Random forests: bagged decision trees with per-split feature subsampling.

use crate::common::labels::LabelEncoder;
use crate::common::traits::{Estimator, Predictor};
use crate::common::validation::{check_n_features, check_xy};
use crate::error::{MlError, Result};
use crate::tree::{DecisionTree, Task, TreeParams};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Draw `n` bootstrap sample indices (with replacement).
fn bootstrap_indices(n: usize, rng: &mut StdRng) -> Vec<usize> {
    (0..n).map(|_| rng.gen_range(0..n)).collect()
}

/// Random forest classifier: majority vote over bagged trees.
#[derive(Debug, Clone)]
pub struct RandomForestClassifier {
    n_estimators: usize,
    params: TreeParams,
    entropy: bool,
    seed: u64,
    trees: Vec<DecisionTree>,
    encoder: Option<LabelEncoder>,
}

impl RandomForestClassifier {
    pub fn new(n_estimators: usize, params: TreeParams, entropy: bool, seed: u64) -> Self {
        Self {
            n_estimators,
            params,
            entropy,
            seed,
            trees: Vec::new(),
            encoder: None,
        }
    }

    pub fn classes(&self) -> Option<&[f64]> {
        self.encoder.as_ref().map(|e| e.classes())
    }
}

impl Estimator for RandomForestClassifier {
    fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<()> {
        check_xy(x, y)?;
        let encoder = LabelEncoder::fit(y)?;
        let n_classes = encoder.n_classes();
        let y_idx = encoder.encode(y)?.mapv(|v| v as f64);

        // Default feature bagging: sqrt(n_features) if unset.
        let mut params = self.params;
        if params.max_features.is_none() {
            params.max_features = Some((x.ncols() as f64).sqrt().ceil() as usize);
        }

        let n = x.nrows();
        self.trees = (0..self.n_estimators)
            .map(|t| {
                let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(t as u64));
                let idx = bootstrap_indices(n, &mut rng);
                let xb = x.select(Axis(0), &idx);
                let yb = y_idx.select(Axis(0), &idx);
                let mut tree = DecisionTree::new(
                    Task::Classification {
                        n_classes,
                        entropy: self.entropy,
                    },
                    params,
                );
                tree.fit(xb.view(), yb.view(), &mut rng);
                tree
            })
            .collect();
        self.encoder = Some(encoder);
        Ok(())
    }
}

impl Predictor for RandomForestClassifier {
    fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        if self.trees.is_empty() {
            return Err(MlError::NotFitted("RandomForestClassifier".into()));
        }
        check_n_features(x, self.trees[0].n_features())?;
        let n_classes = self.encoder.as_ref().unwrap().n_classes();
        let mut votes = Array2::<u32>::zeros((x.nrows(), n_classes));
        for tree in &self.trees {
            let pred = tree.predict(x);
            for (i, &p) in pred.iter().enumerate() {
                votes[[i, p as usize]] += 1;
            }
        }
        let idx: Vec<usize> = votes
            .rows()
            .into_iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by_key(|&(_, &v)| v)
                    .map(|(c, _)| c)
                    .unwrap()
            })
            .collect();
        Ok(self.encoder.as_ref().unwrap().decode(&idx))
    }
}

/// Random forest regressor: average prediction over bagged trees.
#[derive(Debug, Clone)]
pub struct RandomForestRegressor {
    n_estimators: usize,
    params: TreeParams,
    seed: u64,
    trees: Vec<DecisionTree>,
}

impl RandomForestRegressor {
    pub fn new(n_estimators: usize, params: TreeParams, seed: u64) -> Self {
        Self {
            n_estimators,
            params,
            seed,
            trees: Vec::new(),
        }
    }
}

impl Estimator for RandomForestRegressor {
    fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<()> {
        check_xy(x, y)?;
        let mut params = self.params;
        if params.max_features.is_none() {
            // Regression default: ~1/3 of features (at least 1).
            params.max_features = Some(((x.ncols() as f64) / 3.0).ceil().max(1.0) as usize);
        }
        let n = x.nrows();
        self.trees = (0..self.n_estimators)
            .map(|t| {
                let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(t as u64));
                let idx = bootstrap_indices(n, &mut rng);
                let xb = x.select(Axis(0), &idx);
                let yb = y.select(Axis(0), &idx);
                let mut tree = DecisionTree::new(Task::Regression, params);
                tree.fit(xb.view(), yb.view(), &mut rng);
                tree
            })
            .collect();
        Ok(())
    }
}

impl Predictor for RandomForestRegressor {
    fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        if self.trees.is_empty() {
            return Err(MlError::NotFitted("RandomForestRegressor".into()));
        }
        check_n_features(x, self.trees[0].n_features())?;
        let mut sum = Array1::<f64>::zeros(x.nrows());
        for tree in &self.trees {
            sum += &tree.predict(x);
        }
        Ok(sum / self.trees.len() as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::metrics::accuracy_score;
    use ndarray::array;

    #[test]
    fn forest_classifies_blobs() {
        let x = array![
            [0.0, 0.0],
            [0.2, 0.1],
            [0.1, 0.3],
            [0.3, 0.0],
            [5.0, 5.0],
            [5.2, 4.8],
            [4.9, 5.1],
            [5.1, 5.2]
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let mut m = RandomForestClassifier::new(20, TreeParams::default(), false, 42);
        m.fit(x.view(), y.view()).unwrap();
        let pred = m.predict(x.view()).unwrap();
        assert_eq!(accuracy_score(y.view(), pred.view()), 1.0);
    }
}
