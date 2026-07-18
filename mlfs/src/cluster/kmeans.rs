//! K-Means clustering with k-means++ initialisation (Lloyd's algorithm).

use crate::common::distance::squared_euclidean;
use crate::error::{MlError, Result};
use ndarray::{Array1, Array2, ArrayView2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// K-Means clustering: partitions data into `k` clusters by minimising the
/// within-cluster sum of squares.
#[derive(Debug, Clone)]
pub struct KMeans {
    k: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
    centroids: Option<Array2<f64>>,
    labels: Option<Array1<usize>>,
    inertia: f64,
}

impl KMeans {
    pub fn new(k: usize, max_iter: usize, tol: f64, seed: u64) -> Result<Self> {
        if k == 0 {
            return Err(MlError::InvalidParameter("k must be >= 1".into()));
        }
        Ok(Self {
            k,
            max_iter,
            tol,
            seed,
            centroids: None,
            labels: None,
            inertia: 0.0,
        })
    }

    pub fn centroids(&self) -> Option<&Array2<f64>> {
        self.centroids.as_ref()
    }
    pub fn labels(&self) -> Option<&Array1<usize>> {
        self.labels.as_ref()
    }
    /// Final within-cluster sum of squared distances.
    pub fn inertia(&self) -> f64 {
        self.inertia
    }

    /// k-means++ seeding: spread initial centers out probabilistically.
    fn kmeans_plus_plus(&self, x: ArrayView2<f64>, rng: &mut StdRng) -> Array2<f64> {
        let n = x.nrows();
        let p = x.ncols();
        let mut centers = Array2::<f64>::zeros((self.k, p));
        // First center: uniformly at random.
        let first = rng.gen_range(0..n);
        centers.row_mut(0).assign(&x.row(first));

        let mut closest: Vec<f64> = (0..n)
            .map(|i| squared_euclidean(x.row(i), x.row(first)))
            .collect();

        for c in 1..self.k {
            let total: f64 = closest.iter().sum();
            let mut target = rng.gen::<f64>() * total;
            let mut chosen = n - 1;
            for (i, &d) in closest.iter().enumerate() {
                target -= d;
                if target <= 0.0 {
                    chosen = i;
                    break;
                }
            }
            centers.row_mut(c).assign(&x.row(chosen));
            // Update nearest-center distances.
            for i in 0..n {
                let d = squared_euclidean(x.row(i), x.row(chosen));
                if d < closest[i] {
                    closest[i] = d;
                }
            }
        }
        centers
    }

    pub fn fit(&mut self, x: ArrayView2<f64>) -> Result<()> {
        let n = x.nrows();
        if n < self.k {
            return Err(MlError::InvalidParameter("n_samples must be >= k".into()));
        }
        let p = x.ncols();
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut centroids = self.kmeans_plus_plus(x, &mut rng);
        let mut labels = Array1::<usize>::zeros(n);

        for _ in 0..self.max_iter {
            // Assignment step.
            for i in 0..n {
                let mut best = 0usize;
                let mut best_d = f64::INFINITY;
                for c in 0..self.k {
                    let d = squared_euclidean(x.row(i), centroids.row(c));
                    if d < best_d {
                        best_d = d;
                        best = c;
                    }
                }
                labels[i] = best;
            }

            // Update step.
            let mut new_centroids = Array2::<f64>::zeros((self.k, p));
            let mut counts = vec![0usize; self.k];
            for i in 0..n {
                let c = labels[i];
                counts[c] += 1;
                let mut row = new_centroids.row_mut(c);
                row += &x.row(i);
            }
            for c in 0..self.k {
                if counts[c] > 0 {
                    new_centroids
                        .row_mut(c)
                        .mapv_inplace(|v| v / counts[c] as f64);
                } else {
                    // Empty cluster: reseed to a random point.
                    let r = rng.gen_range(0..n);
                    new_centroids.row_mut(c).assign(&x.row(r));
                }
            }

            let shift: f64 = (&new_centroids - &centroids).mapv(|v| v * v).sum().sqrt();
            centroids = new_centroids;
            if shift < self.tol {
                break;
            }
        }

        // Final assignment + inertia.
        let mut inertia = 0.0;
        for i in 0..n {
            let mut best = 0usize;
            let mut best_d = f64::INFINITY;
            for c in 0..self.k {
                let d = squared_euclidean(x.row(i), centroids.row(c));
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            labels[i] = best;
            inertia += best_d;
        }

        self.centroids = Some(centroids);
        self.labels = Some(labels);
        self.inertia = inertia;
        Ok(())
    }

    /// Assign new points to the nearest fitted centroid.
    pub fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        let centroids = self
            .centroids
            .as_ref()
            .ok_or_else(|| MlError::NotFitted("KMeans".into()))?;
        let mut out = Array1::<f64>::zeros(x.nrows());
        for (i, row) in x.rows().into_iter().enumerate() {
            let mut best = 0usize;
            let mut best_d = f64::INFINITY;
            for c in 0..centroids.nrows() {
                let d = squared_euclidean(row, centroids.row(c));
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            out[i] = best as f64;
        }
        Ok(out)
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
    fn kmeans_finds_two_blobs() {
        let x = array![
            [0.0, 0.0],
            [0.2, 0.1],
            [0.1, 0.2],
            [5.0, 5.0],
            [5.2, 4.9],
            [4.9, 5.1]
        ];
        let mut km = KMeans::new(2, 100, 1e-6, 0).unwrap();
        let labels = km.fit_predict(x.view()).unwrap();
        // The first three and last three points should share a cluster each.
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_ne!(labels[0], labels[3]);
    }
}
