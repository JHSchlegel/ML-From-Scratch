//! Core CART (Classification And Regression Tree) used directly by the public
//! tree estimators and reused by the ensemble methods (random forest, boosting).

use ndarray::{Array1, ArrayView1, ArrayView2};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

/// What the tree optimises for.
#[derive(Debug, Clone, Copy)]
pub enum Task {
    /// Classification with `n_classes` classes; leaf value is a class index.
    /// `entropy = true` uses information gain, otherwise Gini impurity.
    Classification { n_classes: usize, entropy: bool },
    /// Regression; leaf value is the mean target.
    Regression,
}

/// Hyper-parameters controlling tree growth.
#[derive(Debug, Clone, Copy)]
pub struct TreeParams {
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    /// Number of features to consider per split (feature bagging). `None` = all.
    pub max_features: Option<usize>,
}

impl Default for TreeParams {
    fn default() -> Self {
        Self {
            max_depth: usize::MAX,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
        }
    }
}

#[derive(Debug, Clone)]
enum Node {
    Leaf {
        value: f64,
    },
    Split {
        feature: usize,
        threshold: f64,
        left: Box<Node>,
        right: Box<Node>,
    },
}

/// A fitted decision tree.
#[derive(Debug, Clone)]
pub struct DecisionTree {
    task: Task,
    params: TreeParams,
    root: Option<Node>,
    n_features: usize,
}

impl DecisionTree {
    pub fn new(task: Task, params: TreeParams) -> Self {
        Self {
            task,
            params,
            root: None,
            n_features: 0,
        }
    }

    /// Fit on `x` and target `y` (class indices as f64 for classification),
    /// using `rng` for feature subsampling when `max_features` is set.
    pub fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<f64>, rng: &mut StdRng) {
        self.n_features = x.ncols();
        let idx: Vec<usize> = (0..x.nrows()).collect();
        self.root = Some(self.build(x, y, &idx, 0, rng));
    }

    /// Number of features seen during `fit`.
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    pub fn predict(&self, x: ArrayView2<f64>) -> Array1<f64> {
        let root = self.root.as_ref().expect("tree not fitted");
        let mut out = Array1::zeros(x.nrows());
        for (i, row) in x.rows().into_iter().enumerate() {
            out[i] = Self::predict_row(root, row);
        }
        out
    }

    fn predict_row(node: &Node, row: ArrayView1<f64>) -> f64 {
        match node {
            Node::Leaf { value } => *value,
            Node::Split {
                feature,
                threshold,
                left,
                right,
            } => {
                if row[*feature] <= *threshold {
                    Self::predict_row(left, row)
                } else {
                    Self::predict_row(right, row)
                }
            }
        }
    }

    /// Leaf value for the samples `idx`: majority class or mean target.
    fn leaf_value(&self, y: ArrayView1<f64>, idx: &[usize]) -> f64 {
        match self.task {
            Task::Classification { n_classes, .. } => {
                let mut counts = vec![0usize; n_classes];
                for &i in idx {
                    counts[y[i] as usize] += 1;
                }
                counts
                    .iter()
                    .enumerate()
                    .max_by_key(|&(_, &c)| c)
                    .map(|(c, _)| c as f64)
                    .unwrap_or(0.0)
            }
            Task::Regression => idx.iter().map(|&i| y[i]).sum::<f64>() / idx.len() as f64,
        }
    }

    /// Impurity of the node containing samples `idx`.
    fn impurity(&self, y: ArrayView1<f64>, idx: &[usize]) -> f64 {
        let n = idx.len() as f64;
        match self.task {
            Task::Classification { n_classes, entropy } => {
                let mut counts = vec![0usize; n_classes];
                for &i in idx {
                    counts[y[i] as usize] += 1;
                }
                if entropy {
                    counts
                        .iter()
                        .filter(|&&c| c > 0)
                        .map(|&c| {
                            let p = c as f64 / n;
                            -p * p.log2()
                        })
                        .sum()
                } else {
                    1.0 - counts.iter().map(|&c| (c as f64 / n).powi(2)).sum::<f64>()
                }
            }
            Task::Regression => {
                let mean = idx.iter().map(|&i| y[i]).sum::<f64>() / n;
                idx.iter().map(|&i| (y[i] - mean).powi(2)).sum::<f64>() / n
            }
        }
    }

    fn build(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        idx: &[usize],
        depth: usize,
        rng: &mut StdRng,
    ) -> Node {
        // Stopping conditions.
        let stop = depth >= self.params.max_depth
            || idx.len() < self.params.min_samples_split
            || self.is_pure(y, idx);
        if stop {
            return Node::Leaf {
                value: self.leaf_value(y, idx),
            };
        }

        match self.best_split(x, y, idx, rng) {
            Some((feature, threshold, left_idx, right_idx)) => Node::Split {
                feature,
                threshold,
                left: Box::new(self.build(x, y, &left_idx, depth + 1, rng)),
                right: Box::new(self.build(x, y, &right_idx, depth + 1, rng)),
            },
            None => Node::Leaf {
                value: self.leaf_value(y, idx),
            },
        }
    }

    fn is_pure(&self, y: ArrayView1<f64>, idx: &[usize]) -> bool {
        if idx.is_empty() {
            return true;
        }
        let first = y[idx[0]];
        idx.iter().all(|&i| (y[i] - first).abs() < 1e-12)
    }

    /// Search for the split minimising the children's weighted impurity.
    /// Returns `(feature, threshold, left_idx, right_idx)`.
    fn best_split(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        idx: &[usize],
        rng: &mut StdRng,
    ) -> Option<(usize, f64, Vec<usize>, Vec<usize>)> {
        let parent_imp = self.impurity(y, idx);
        let n = idx.len() as f64;

        // Candidate features (subsample for feature bagging).
        let mut features: Vec<usize> = (0..self.n_features).collect();
        if let Some(m) = self.params.max_features {
            if m < self.n_features {
                features.shuffle(rng);
                features.truncate(m);
            }
        }

        let mut best_gain = 0.0;
        let mut best: Option<(usize, f64, Vec<usize>, Vec<usize>)> = None;

        for &f in &features {
            // Sort sample indices by this feature's value.
            let mut order: Vec<usize> = idx.to_vec();
            order.sort_by(|&a, &b| x[[a, f]].partial_cmp(&x[[b, f]]).unwrap());

            // Try thresholds between consecutive distinct values.
            for w in 1..order.len() {
                let v_prev = x[[order[w - 1], f]];
                let v_cur = x[[order[w], f]];
                if (v_cur - v_prev).abs() < 1e-12 {
                    continue;
                }
                let left = &order[..w];
                let right = &order[w..];
                if left.len() < self.params.min_samples_leaf
                    || right.len() < self.params.min_samples_leaf
                {
                    continue;
                }
                let il = self.impurity(y, left);
                let ir = self.impurity(y, right);
                let child = (left.len() as f64 / n) * il + (right.len() as f64 / n) * ir;
                let gain = parent_imp - child;
                if gain > best_gain + 1e-12 {
                    best_gain = gain;
                    let threshold = 0.5 * (v_prev + v_cur);
                    best = Some((f, threshold, left.to_vec(), right.to_vec()));
                }
            }
        }
        best
    }
}
