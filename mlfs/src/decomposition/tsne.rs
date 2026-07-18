//! t-distributed Stochastic Neighbor Embedding (t-SNE).
//!
//! A direct O(n^2) implementation of van der Maaten & Hinton (2008): calibrate
//! per-point Gaussian bandwidths to a target perplexity, symmetrise the
//! affinities, then minimise the KL divergence to a Student-t embedding by
//! gradient descent with momentum and early exaggeration.

use crate::error::{MlError, Result};
use ndarray::{Array1, Array2, ArrayView2, Axis};
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// t-SNE embedding into `n_components` (usually 2) dimensions.
#[derive(Debug, Clone)]
pub struct TSNE {
    n_components: usize,
    perplexity: f64,
    learning_rate: f64,
    n_iter: usize,
    seed: u64,
}

impl TSNE {
    pub fn new(
        n_components: usize,
        perplexity: f64,
        learning_rate: f64,
        n_iter: usize,
        seed: u64,
    ) -> Self {
        Self {
            n_components,
            perplexity,
            learning_rate,
            n_iter,
            seed,
        }
    }

    /// Fit and return the low-dimensional embedding, shape (n_samples, n_components).
    pub fn fit_transform(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        let n = x.nrows();
        if n < 3 {
            return Err(MlError::InvalidParameter("t-SNE needs >= 3 samples".into()));
        }

        let distances = pairwise_sq_dists(x);
        let mut p = self.high_dim_affinities(&distances)?;
        // Early exaggeration encourages tight, well-separated clusters early on.
        p.mapv_inplace(|v| v * 12.0);

        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut y = Array2::random_using(
            (n, self.n_components),
            Normal::new(0.0, 1e-4).unwrap(),
            &mut rng,
        );
        let mut y_inc = Array2::<f64>::zeros((n, self.n_components));

        for iter in 0..self.n_iter {
            // Student-t affinities in the embedding.
            let (q, num) = self.low_dim_affinities(&y);

            // Gradient: 4 * sum_j (P_ij - Q_ij) * num_ij * (y_i - y_j).
            let pq = &p - &q;
            let mut grad = Array2::<f64>::zeros((n, self.n_components));
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let mult = pq[[i, j]] * num[[i, j]];
                    for d in 0..self.n_components {
                        grad[[i, d]] += 4.0 * mult * (y[[i, d]] - y[[j, d]]);
                    }
                }
            }

            let momentum = if iter < 250 { 0.5 } else { 0.8 };
            y_inc = &y_inc * momentum - &(self.learning_rate * &grad);
            y = &y + &y_inc;
            // Re-center.
            let mean = y.mean_axis(Axis(0)).unwrap();
            y = &y - &mean;

            // End early exaggeration.
            if iter == 100 {
                p.mapv_inplace(|v| v / 12.0);
            }
        }
        Ok(y)
    }

    /// Symmetrised high-dimensional affinities `P` from squared distances.
    fn high_dim_affinities(&self, distances: &Array2<f64>) -> Result<Array2<f64>> {
        let n = distances.nrows();
        let target = self.perplexity.log2();
        let mut p = Array2::<f64>::zeros((n, n));

        for i in 0..n {
            // Binary-search beta = 1 / (2 sigma_i^2) to hit the target perplexity.
            let mut beta = 1.0_f64;
            let mut beta_min = f64::NEG_INFINITY;
            let mut beta_max = f64::INFINITY;
            let mut row = Array1::<f64>::zeros(n);

            for _ in 0..50 {
                let mut sum = 0.0;
                for j in 0..n {
                    if i == j {
                        row[j] = 0.0;
                    } else {
                        row[j] = (-beta * distances[[i, j]]).exp();
                        sum += row[j];
                    }
                }
                if sum < 1e-12 {
                    sum = 1e-12;
                }
                // Shannon entropy H = log(sum) + beta * <d>.
                let mut dp = 0.0;
                for j in 0..n {
                    dp += beta * distances[[i, j]] * row[j];
                }
                let h = sum.ln() / std::f64::consts::LN_2 + (dp / sum) / std::f64::consts::LN_2;

                let diff = h - target;
                if diff.abs() < 1e-5 {
                    break;
                }
                if diff > 0.0 {
                    beta_min = beta;
                    beta = if beta_max.is_infinite() {
                        beta * 2.0
                    } else {
                        (beta + beta_max) / 2.0
                    };
                } else {
                    beta_max = beta;
                    beta = if beta_min.is_infinite() {
                        beta / 2.0
                    } else {
                        (beta + beta_min) / 2.0
                    };
                }
            }
            let sum: f64 = row.sum().max(1e-12);
            for j in 0..n {
                p[[i, j]] = row[j] / sum;
            }
        }

        // Symmetrise and normalise: P_ij = (P_j|i + P_i|j) / (2n).
        let mut sym = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                sym[[i, j]] = (p[[i, j]] + p[[j, i]]) / (2.0 * n as f64);
            }
        }
        sym.mapv_inplace(|v| v.max(1e-12));
        Ok(sym)
    }

    /// Student-t affinities `Q` and the unnormalised numerators.
    fn low_dim_affinities(&self, y: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        let n = y.nrows();
        let mut num = Array2::<f64>::zeros((n, n));
        let mut sum = 0.0;
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let mut d2 = 0.0;
                for k in 0..y.ncols() {
                    d2 += (y[[i, k]] - y[[j, k]]).powi(2);
                }
                let v = 1.0 / (1.0 + d2);
                num[[i, j]] = v;
                sum += v;
            }
        }
        sum = sum.max(1e-12);
        let q = num.mapv(|v| (v / sum).max(1e-12));
        (q, num)
    }
}

/// Pairwise squared Euclidean distances.
fn pairwise_sq_dists(x: ArrayView2<f64>) -> Array2<f64> {
    let n = x.nrows();
    let mut d = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            let v: f64 = x
                .row(i)
                .iter()
                .zip(x.row(j).iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            d[[i, j]] = v;
            d[[j, i]] = v;
        }
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn tsne_separates_two_clusters() {
        // Two tight clusters far apart in 3-D.
        let x = array![
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.1],
            [0.0, 0.1, 0.0],
            [0.1, 0.1, 0.0],
            [10.0, 10.0, 10.0],
            [10.1, 10.0, 10.1],
            [10.0, 10.1, 10.0],
            [10.1, 10.1, 10.0]
        ];
        let tsne = TSNE::new(2, 3.0, 200.0, 300, 0);
        let y = tsne.fit_transform(x.view()).unwrap();
        assert_eq!(y.dim(), (8, 2));

        // Within-cluster distances should be smaller than across-cluster ones.
        let d = |a: usize, b: usize| -> f64 {
            ((y[[a, 0]] - y[[b, 0]]).powi(2) + (y[[a, 1]] - y[[b, 1]]).powi(2)).sqrt()
        };
        let within = d(0, 1);
        let across = d(0, 4);
        assert!(
            across > within,
            "clusters should be separated: {across} vs {within}"
        );
    }
}
