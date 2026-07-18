//! Agglomerative (bottom-up) hierarchical clustering.

use crate::common::distance::euclidean;
use crate::error::{MlError, Result};
use ndarray::{Array1, ArrayView2};

/// Linkage criterion: how the distance between two clusters is measured.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Linkage {
    /// Minimum pairwise distance between the clusters.
    Single,
    /// Maximum pairwise distance.
    Complete,
    /// Mean pairwise distance.
    Average,
}

/// Agglomerative clustering: start with every point in its own cluster and
/// repeatedly merge the two closest clusters until `n_clusters` remain.
#[derive(Debug, Clone)]
pub struct AgglomerativeClustering {
    n_clusters: usize,
    linkage: Linkage,
    labels: Option<Array1<usize>>,
}

impl AgglomerativeClustering {
    pub fn new(n_clusters: usize, linkage: Linkage) -> Result<Self> {
        if n_clusters == 0 {
            return Err(MlError::InvalidParameter("n_clusters must be >= 1".into()));
        }
        Ok(Self {
            n_clusters,
            linkage,
            labels: None,
        })
    }

    pub fn labels(&self) -> Option<&Array1<usize>> {
        self.labels.as_ref()
    }

    fn cluster_distance(&self, a: &[usize], b: &[usize], pdist: &[Vec<f64>]) -> f64 {
        let mut acc = match self.linkage {
            Linkage::Single => f64::INFINITY,
            Linkage::Complete => f64::NEG_INFINITY,
            Linkage::Average => 0.0,
        };
        for &i in a {
            for &j in b {
                let d = pdist[i][j];
                match self.linkage {
                    Linkage::Single => acc = acc.min(d),
                    Linkage::Complete => acc = acc.max(d),
                    Linkage::Average => acc += d,
                }
            }
        }
        if self.linkage == Linkage::Average {
            acc / (a.len() * b.len()) as f64
        } else {
            acc
        }
    }

    pub fn fit(&mut self, x: ArrayView2<f64>) -> Result<()> {
        let n = x.nrows();
        if n < self.n_clusters {
            return Err(MlError::InvalidParameter(
                "n_samples must be >= n_clusters".into(),
            ));
        }

        // Precompute the pairwise distance matrix.
        let mut pdist = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = euclidean(x.row(i), x.row(j));
                pdist[i][j] = d;
                pdist[j][i] = d;
            }
        }

        // Active clusters, each a list of point indices.
        let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

        while clusters.len() > self.n_clusters {
            // Find the closest pair of clusters.
            let mut best = (0usize, 1usize);
            let mut best_d = f64::INFINITY;
            for a in 0..clusters.len() {
                for b in (a + 1)..clusters.len() {
                    let d = self.cluster_distance(&clusters[a], &clusters[b], &pdist);
                    if d < best_d {
                        best_d = d;
                        best = (a, b);
                    }
                }
            }
            // Merge b into a, remove b.
            let (a, b) = best;
            let moved = clusters.remove(b);
            clusters[a].extend(moved);
        }

        // Assign labels by final cluster membership.
        let mut labels = Array1::<usize>::zeros(n);
        for (c, members) in clusters.iter().enumerate() {
            for &i in members {
                labels[i] = c;
            }
        }
        self.labels = Some(labels);
        Ok(())
    }

    pub fn fit_predict(&mut self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        self.fit(x)?;
        Ok(self.labels.as_ref().unwrap().mapv(|v| v as f64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn agglomerative_merges_nearby_points() {
        let x = array![
            [0.0, 0.0],
            [0.2, 0.1],
            [0.1, 0.2],
            [9.0, 9.0],
            [9.2, 8.9],
            [8.9, 9.1]
        ];
        let mut ac = AgglomerativeClustering::new(2, Linkage::Average).unwrap();
        let labels = ac.fit_predict(x.view()).unwrap();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_ne!(labels[0], labels[3]);
    }
}
