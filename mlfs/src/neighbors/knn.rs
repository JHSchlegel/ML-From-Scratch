//! k-nearest-neighbours classifier and regressor (brute-force search).

use crate::common::distance::squared_euclidean;
use crate::common::labels::LabelEncoder;
use crate::common::traits::{Estimator, Predictor};
use crate::common::validation::{check_n_features, check_xy};
use crate::error::{MlError, Result};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Find the indices of the `k` training rows closest to `q`.
fn k_nearest(train: &Array2<f64>, q: ArrayView1<f64>, k: usize) -> Vec<usize> {
    let mut dists: Vec<(f64, usize)> = train
        .rows()
        .into_iter()
        .enumerate()
        .map(|(i, row)| (squared_euclidean(row, q), i))
        .collect();
    // Partial selection is enough, but n is small for a from-scratch impl.
    dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    dists.iter().take(k).map(|&(_, i)| i).collect()
}

/// k-NN classifier: predicts the majority class among the `k` nearest neighbours.
#[derive(Debug, Clone)]
pub struct KNeighborsClassifier {
    k: usize,
    x_train: Option<Array2<f64>>,
    y_idx: Option<Vec<usize>>,
    encoder: Option<LabelEncoder>,
}

impl KNeighborsClassifier {
    pub fn new(k: usize) -> Result<Self> {
        if k == 0 {
            return Err(MlError::InvalidParameter("k must be >= 1".into()));
        }
        Ok(Self {
            k,
            x_train: None,
            y_idx: None,
            encoder: None,
        })
    }
}

impl Estimator for KNeighborsClassifier {
    fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<()> {
        check_xy(x, y)?;
        let encoder = LabelEncoder::fit(y)?;
        self.y_idx = Some(encoder.encode(y)?.to_vec());
        self.encoder = Some(encoder);
        self.x_train = Some(x.to_owned());
        Ok(())
    }
}

impl Predictor for KNeighborsClassifier {
    fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        let train = self
            .x_train
            .as_ref()
            .ok_or_else(|| MlError::NotFitted("KNeighborsClassifier".into()))?;
        check_n_features(x, train.ncols())?;
        let y_idx = self.y_idx.as_ref().unwrap();
        let encoder = self.encoder.as_ref().unwrap();
        let n_classes = encoder.n_classes();
        let k = self.k.min(train.nrows());

        let mut out = Vec::with_capacity(x.nrows());
        for q in x.rows() {
            let nn = k_nearest(train, q, k);
            let mut votes = vec![0usize; n_classes];
            for idx in nn {
                votes[y_idx[idx]] += 1;
            }
            let best = votes
                .iter()
                .enumerate()
                .max_by_key(|&(_, &v)| v)
                .map(|(i, _)| i)
                .unwrap();
            out.push(best);
        }
        Ok(encoder.decode(&out))
    }
}

/// k-NN regressor: predicts the mean target of the `k` nearest neighbours.
#[derive(Debug, Clone)]
pub struct KNeighborsRegressor {
    k: usize,
    x_train: Option<Array2<f64>>,
    y_train: Option<Array1<f64>>,
}

impl KNeighborsRegressor {
    pub fn new(k: usize) -> Result<Self> {
        if k == 0 {
            return Err(MlError::InvalidParameter("k must be >= 1".into()));
        }
        Ok(Self {
            k,
            x_train: None,
            y_train: None,
        })
    }
}

impl Estimator for KNeighborsRegressor {
    fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<()> {
        check_xy(x, y)?;
        self.x_train = Some(x.to_owned());
        self.y_train = Some(y.to_owned());
        Ok(())
    }
}

impl Predictor for KNeighborsRegressor {
    fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        let train = self
            .x_train
            .as_ref()
            .ok_or_else(|| MlError::NotFitted("KNeighborsRegressor".into()))?;
        check_n_features(x, train.ncols())?;
        let y = self.y_train.as_ref().unwrap();
        let k = self.k.min(train.nrows());

        let mut out = Array1::<f64>::zeros(x.nrows());
        for (i, q) in x.rows().into_iter().enumerate() {
            let nn = k_nearest(train, q, k);
            let mean = nn.iter().map(|&idx| y[idx]).sum::<f64>() / k as f64;
            out[i] = mean;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::metrics::accuracy_score;
    use ndarray::array;

    #[test]
    fn knn_classifies_neighbourhoods() {
        let x = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [5.0, 5.0],
            [5.1, 5.0],
            [5.0, 5.1]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut m = KNeighborsClassifier::new(3).unwrap();
        m.fit(x.view(), y.view()).unwrap();
        let q = array![[0.05, 0.05], [4.9, 5.0]];
        let pred = m.predict(q.view()).unwrap();
        assert_eq!(pred[0], 0.0);
        assert_eq!(pred[1], 1.0);
        assert_eq!(
            accuracy_score(y.view(), m.predict(x.view()).unwrap().view()),
            1.0
        );
    }
}
