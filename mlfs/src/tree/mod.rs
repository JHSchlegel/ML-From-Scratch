//! Decision trees (CART).

pub(crate) mod cart;

pub use cart::{DecisionTree, Task, TreeParams};

use crate::common::labels::LabelEncoder;
use crate::common::traits::{Estimator, Predictor};
use crate::common::validation::{check_n_features, check_xy};
use crate::error::{MlError, Result};
use ndarray::{Array1, ArrayView1, ArrayView2};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// CART decision tree classifier (Gini or entropy criterion).
#[derive(Debug, Clone)]
pub struct DecisionTreeClassifier {
    params: TreeParams,
    entropy: bool,
    seed: u64,
    tree: Option<DecisionTree>,
    encoder: Option<LabelEncoder>,
}

impl DecisionTreeClassifier {
    pub fn new(params: TreeParams, entropy: bool, seed: u64) -> Self {
        Self {
            params,
            entropy,
            seed,
            tree: None,
            encoder: None,
        }
    }

    pub fn classes(&self) -> Option<&[f64]> {
        self.encoder.as_ref().map(|e| e.classes())
    }
}

impl Estimator for DecisionTreeClassifier {
    fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<()> {
        check_xy(x, y)?;
        let encoder = LabelEncoder::fit(y)?;
        let y_idx = encoder.encode(y)?.mapv(|v| v as f64);
        let mut tree = DecisionTree::new(
            Task::Classification {
                n_classes: encoder.n_classes(),
                entropy: self.entropy,
            },
            self.params,
        );
        let mut rng = StdRng::seed_from_u64(self.seed);
        tree.fit(x, y_idx.view(), &mut rng);
        self.tree = Some(tree);
        self.encoder = Some(encoder);
        Ok(())
    }
}

impl Predictor for DecisionTreeClassifier {
    fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        let tree = self
            .tree
            .as_ref()
            .ok_or_else(|| MlError::NotFitted("DecisionTreeClassifier".into()))?;
        check_n_features(x, tree.n_features())?;
        let idx = tree.predict(x).mapv(|v| v as usize);
        Ok(self
            .encoder
            .as_ref()
            .unwrap()
            .decode(idx.as_slice().unwrap()))
    }
}

/// CART decision tree regressor (variance-reduction / MSE criterion).
#[derive(Debug, Clone)]
pub struct DecisionTreeRegressor {
    params: TreeParams,
    seed: u64,
    tree: Option<DecisionTree>,
    n_features: usize,
}

impl DecisionTreeRegressor {
    pub fn new(params: TreeParams, seed: u64) -> Self {
        Self {
            params,
            seed,
            tree: None,
            n_features: 0,
        }
    }
}

impl Estimator for DecisionTreeRegressor {
    fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<()> {
        check_xy(x, y)?;
        let mut tree = DecisionTree::new(Task::Regression, self.params);
        let mut rng = StdRng::seed_from_u64(self.seed);
        tree.fit(x, y, &mut rng);
        self.n_features = x.ncols();
        self.tree = Some(tree);
        Ok(())
    }
}

impl Predictor for DecisionTreeRegressor {
    fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        let tree = self
            .tree
            .as_ref()
            .ok_or_else(|| MlError::NotFitted("DecisionTreeRegressor".into()))?;
        check_n_features(x, self.n_features)?;
        Ok(tree.predict(x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::metrics::{accuracy_score, r2_score};
    use ndarray::array;

    #[test]
    fn classifier_fits_axis_aligned_regions() {
        // Four quadrants, each its own class — CART splits recursively.
        let x = array![
            [0.0, 0.0],
            [0.1, 0.2],
            [0.2, 0.1],
            [0.0, 5.0],
            [0.1, 5.2],
            [0.2, 4.9],
            [5.0, 0.0],
            [5.1, 0.2],
            [4.9, 0.1],
            [5.0, 5.0],
            [5.2, 5.1],
            [4.8, 4.9]
        ];
        let y = array![0., 0., 0., 1., 1., 1., 2., 2., 2., 3., 3., 3.];
        let mut m = DecisionTreeClassifier::new(TreeParams::default(), false, 0);
        m.fit(x.view(), y.view()).unwrap();
        let pred = m.predict(x.view()).unwrap();
        assert_eq!(accuracy_score(y.view(), pred.view()), 1.0);
    }

    #[test]
    fn regressor_fits_step_function() {
        let x = array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![0.0, 0.0, 0.0, 10.0, 10.0, 10.0];
        let mut m = DecisionTreeRegressor::new(TreeParams::default(), 0);
        m.fit(x.view(), y.view()).unwrap();
        let pred = m.predict(x.view()).unwrap();
        assert!(r2_score(y.view(), pred.view()) > 0.99);
    }
}
