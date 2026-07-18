//! Class-label encoding shared by all classifiers.
//!
//! Targets arrive as `f64` (the Python side passes class labels as floats).
//! Internally classifiers work with contiguous class *indices* `0..n_classes`;
//! this helper maps between the two.

use crate::error::{MlError, Result};
use ndarray::{Array1, ArrayView1};

/// Maps arbitrary `f64` class labels to `0..n_classes` and back.
#[derive(Debug, Clone)]
pub struct LabelEncoder {
    /// Sorted unique class labels; position = encoded index.
    classes: Vec<f64>,
}

impl LabelEncoder {
    /// Build an encoder from the labels present in `y`.
    pub fn fit(y: ArrayView1<f64>) -> Result<Self> {
        if y.is_empty() {
            return Err(MlError::InvalidParameter("y has no samples".into()));
        }
        let mut classes: Vec<f64> = y.to_vec();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        classes.dedup();
        Ok(Self { classes })
    }

    pub fn n_classes(&self) -> usize {
        self.classes.len()
    }

    /// Original class labels, in encoded-index order.
    pub fn classes(&self) -> &[f64] {
        &self.classes
    }

    /// Encode labels to indices `0..n_classes`.
    pub fn encode(&self, y: ArrayView1<f64>) -> Result<Array1<usize>> {
        let mut out = Array1::zeros(y.len());
        for (i, &v) in y.iter().enumerate() {
            let idx = self
                .classes
                .iter()
                .position(|&c| (c - v).abs() < 1e-12)
                .ok_or_else(|| MlError::InvalidParameter(format!("unseen label {v}")))?;
            out[i] = idx;
        }
        Ok(out)
    }

    /// Decode a single index back to its original class label.
    pub fn decode_one(&self, idx: usize) -> f64 {
        self.classes[idx]
    }

    /// Decode a vector of indices back to original labels.
    pub fn decode(&self, idx: &[usize]) -> Array1<f64> {
        idx.iter().map(|&i| self.classes[i]).collect()
    }
}
