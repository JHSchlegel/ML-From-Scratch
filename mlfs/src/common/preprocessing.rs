//! Feature scaling and data-splitting utilities.

use crate::common::traits::Transformer;
use crate::error::{MlError, Result};
use ndarray::{Array1, Array2, ArrayView2, Axis};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

/// Standardise features to zero mean and unit variance (per column).
#[derive(Debug, Clone, Default)]
pub struct StandardScaler {
    mean: Option<Array1<f64>>,
    scale: Option<Array1<f64>>,
}

impl StandardScaler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Learn the per-column mean and standard deviation from `x`.
    pub fn fit(&mut self, x: ArrayView2<f64>) -> Result<()> {
        if x.nrows() == 0 {
            return Err(MlError::InvalidParameter("X has no samples".into()));
        }
        let mean = x.mean_axis(Axis(0)).unwrap();
        // Population standard deviation; guard against zero-variance columns.
        let mut scale = x.std_axis(Axis(0), 0.0);
        scale.mapv_inplace(|s| if s > 1e-12 { s } else { 1.0 });
        self.mean = Some(mean);
        self.scale = Some(scale);
        Ok(())
    }

    /// Fit then transform in one call.
    pub fn fit_transform(&mut self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        self.fit(x)?;
        self.transform(x)
    }
}

impl Transformer for StandardScaler {
    fn transform(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        let mean = self
            .mean
            .as_ref()
            .ok_or_else(|| MlError::NotFitted("StandardScaler".into()))?;
        let scale = self.scale.as_ref().unwrap();
        Ok((&x - mean) / scale)
    }
}

/// Shuffle-and-split indices into train/test partitions.
///
/// Returns `(train_idx, test_idx)`. `test_size` is a fraction in `(0, 1)`.
pub fn train_test_split_indices(
    n_samples: usize,
    test_size: f64,
    seed: u64,
) -> Result<(Vec<usize>, Vec<usize>)> {
    if !(0.0..1.0).contains(&test_size) {
        return Err(MlError::InvalidParameter(
            "test_size must be in (0, 1)".into(),
        ));
    }
    let mut idx: Vec<usize> = (0..n_samples).collect();
    let mut rng = StdRng::seed_from_u64(seed);
    idx.shuffle(&mut rng);
    let n_test = ((n_samples as f64) * test_size).round() as usize;
    let test = idx[..n_test].to_vec();
    let train = idx[n_test..].to_vec();
    Ok((train, test))
}
