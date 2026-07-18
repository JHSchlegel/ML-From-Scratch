//! DBSCAN density-based clustering.

use crate::common::distance::euclidean;
use crate::error::{MlError, Result};
use ndarray::{Array1, ArrayView2};

const NOISE: i64 = -1;
const UNVISITED: i64 = -2;

/// DBSCAN: groups together points in dense regions and marks low-density points
/// as noise (label `-1`). No need to specify the number of clusters.
#[derive(Debug, Clone)]
pub struct DBSCAN {
    eps: f64,
    min_samples: usize,
    labels: Option<Array1<i64>>,
}

impl DBSCAN {
    pub fn new(eps: f64, min_samples: usize) -> Result<Self> {
        if eps <= 0.0 {
            return Err(MlError::InvalidParameter("eps must be > 0".into()));
        }
        Ok(Self {
            eps,
            min_samples,
            labels: None,
        })
    }

    pub fn labels(&self) -> Option<&Array1<i64>> {
        self.labels.as_ref()
    }

    fn region_query(&self, x: ArrayView2<f64>, i: usize) -> Vec<usize> {
        (0..x.nrows())
            .filter(|&j| euclidean(x.row(i), x.row(j)) <= self.eps)
            .collect()
    }

    pub fn fit(&mut self, x: ArrayView2<f64>) -> Result<()> {
        let n = x.nrows();
        let mut labels = Array1::<i64>::from_elem(n, UNVISITED);
        let mut cluster = 0i64;

        for i in 0..n {
            if labels[i] != UNVISITED {
                continue;
            }
            let neighbors = self.region_query(x, i);
            if neighbors.len() < self.min_samples {
                labels[i] = NOISE;
                continue;
            }
            // Start a new cluster and expand it.
            labels[i] = cluster;
            let mut queue = neighbors;
            let mut qi = 0;
            while qi < queue.len() {
                let j = queue[qi];
                qi += 1;
                if labels[j] == NOISE {
                    labels[j] = cluster; // border point
                }
                if labels[j] != UNVISITED {
                    continue;
                }
                labels[j] = cluster;
                let j_neighbors = self.region_query(x, j);
                if j_neighbors.len() >= self.min_samples {
                    queue.extend(j_neighbors);
                }
            }
            cluster += 1;
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
    fn dbscan_finds_dense_regions_and_noise() {
        let x = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [5.0, 5.0],
            [5.1, 5.0],
            [5.0, 5.1],
            [5.1, 5.1],
            [100.0, 100.0] // outlier
        ];
        let mut db = DBSCAN::new(0.5, 3).unwrap();
        let labels = db.fit_predict(x.view()).unwrap();
        // Two dense clusters plus one noise point.
        assert_eq!(labels[0], labels[3]);
        assert_eq!(labels[4], labels[7]);
        assert_ne!(labels[0], labels[4]);
        assert_eq!(labels[8], -1.0);
    }
}
