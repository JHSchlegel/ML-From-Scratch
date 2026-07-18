//! Principal Component Analysis via the SVD of the centered data.

use crate::error::{MlError, Result};
use linfa_linalg::svd::SVD;
use ndarray::{Array1, Array2, ArrayView2, Axis};

/// Principal Component Analysis. Projects data onto the `n_components`
/// directions of greatest variance, found from the SVD of the centered matrix.
#[derive(Debug, Clone)]
pub struct PCA {
    n_components: usize,
    mean: Option<Array1<f64>>,
    /// Principal axes, shape (n_components, n_features).
    components: Option<Array2<f64>>,
    explained_variance: Option<Array1<f64>>,
    explained_variance_ratio: Option<Array1<f64>>,
}

impl PCA {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            mean: None,
            components: None,
            explained_variance: None,
            explained_variance_ratio: None,
        }
    }

    pub fn components(&self) -> Option<&Array2<f64>> {
        self.components.as_ref()
    }
    pub fn explained_variance(&self) -> Option<&Array1<f64>> {
        self.explained_variance.as_ref()
    }
    pub fn explained_variance_ratio(&self) -> Option<&Array1<f64>> {
        self.explained_variance_ratio.as_ref()
    }

    pub fn fit(&mut self, x: ArrayView2<f64>) -> Result<()> {
        let (n, p) = (x.nrows(), x.ncols());
        if n < 2 {
            return Err(MlError::InvalidParameter("PCA needs >= 2 samples".into()));
        }
        let k = self.n_components.min(p).min(n);
        let mean = x.mean_axis(Axis(0)).unwrap();
        let xc = &x - &mean;

        // X_c = U S V^T ; principal axes are rows of V^T. linfa-linalg does not
        // guarantee an ordering, so sort singular values descending and reorder
        // the corresponding right-singular vectors.
        let (_u, s, vt) = xc
            .svd(false, true)
            .map_err(|e| MlError::Numerical(format!("SVD failed: {e:?}")))?;
        let vt = vt.ok_or_else(|| MlError::Numerical("SVD returned no V^T".into()))?;

        let mut order: Vec<usize> = (0..s.len()).collect();
        order.sort_by(|&a, &b| s[b].partial_cmp(&s[a]).unwrap());

        let denom = (n - 1) as f64;
        let total: f64 = s.iter().map(|sv| sv * sv / denom).sum();

        let mut components = Array2::<f64>::zeros((k, p));
        let mut explained_variance = Array1::<f64>::zeros(k);
        for (rank, &si) in order.iter().take(k).enumerate() {
            components.row_mut(rank).assign(&vt.row(si));
            explained_variance[rank] = s[si] * s[si] / denom;
        }
        let explained_variance_ratio = if total > 0.0 {
            explained_variance.mapv(|v| v / total)
        } else {
            Array1::zeros(k)
        };

        self.mean = Some(mean);
        self.components = Some(components);
        self.explained_variance = Some(explained_variance);
        self.explained_variance_ratio = Some(explained_variance_ratio);
        Ok(())
    }

    /// Project `x` onto the principal axes, shape (n_samples, n_components).
    pub fn transform(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        let mean = self
            .mean
            .as_ref()
            .ok_or_else(|| MlError::NotFitted("PCA".into()))?;
        let components = self.components.as_ref().unwrap();
        if x.ncols() != mean.len() {
            return Err(MlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                mean.len(),
                x.ncols()
            )));
        }
        let xc = &x - mean;
        Ok(xc.dot(&components.t()))
    }

    pub fn fit_transform(&mut self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        self.fit(x)?;
        self.transform(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn pca_captures_dominant_direction() {
        // Points lie mostly along the (1,1) direction.
        let x = array![
            [-2.0, -2.0],
            [-1.0, -1.1],
            [0.0, 0.1],
            [1.0, 0.9],
            [2.0, 2.1]
        ];
        let mut pca = PCA::new(1);
        let z = pca.fit_transform(x.view()).unwrap();
        assert_eq!(z.ncols(), 1);
        // First component should explain almost all variance.
        assert!(pca.explained_variance_ratio().unwrap()[0] > 0.95);
    }
}
