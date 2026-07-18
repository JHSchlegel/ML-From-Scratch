//! Gaussian Mixture Model fit with the Expectation-Maximization algorithm
//! (full covariance matrices).

use crate::common::linalg::sym_inv_logdet;
use crate::error::{MlError, Result};
use ndarray::{Array1, Array2, Array3, ArrayView2, Axis};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

/// Gaussian Mixture Model: density estimation / soft clustering with `k`
/// full-covariance Gaussian components, trained by EM.
#[derive(Debug, Clone)]
pub struct GaussianMixture {
    k: usize,
    max_iter: usize,
    tol: f64,
    reg: f64,
    seed: u64,
    weights: Option<Array1<f64>>,
    means: Option<Array2<f64>>,
    covs: Option<Array3<f64>>,
    labels: Option<Array1<usize>>,
    log_likelihood: f64,
}

impl GaussianMixture {
    pub fn new(k: usize, max_iter: usize, tol: f64, reg: f64, seed: u64) -> Result<Self> {
        if k == 0 {
            return Err(MlError::InvalidParameter("k must be >= 1".into()));
        }
        Ok(Self {
            k,
            max_iter,
            tol,
            reg,
            seed,
            weights: None,
            means: None,
            covs: None,
            labels: None,
            log_likelihood: 0.0,
        })
    }

    pub fn weights(&self) -> Option<&Array1<f64>> {
        self.weights.as_ref()
    }
    pub fn means(&self) -> Option<&Array2<f64>> {
        self.means.as_ref()
    }
    pub fn labels(&self) -> Option<&Array1<usize>> {
        self.labels.as_ref()
    }

    /// Per-sample, per-component log density: log( w_k * N(x_i | mu_k, Sigma_k) ).
    fn log_prob(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        let means = self.means.as_ref().unwrap();
        let covs = self.covs.as_ref().unwrap();
        let weights = self.weights.as_ref().unwrap();
        let (n, p) = (x.nrows(), x.ncols());
        let const_term = (p as f64) * (2.0 * std::f64::consts::PI).ln();

        let mut lp = Array2::<f64>::zeros((n, self.k));
        for c in 0..self.k {
            let cov = covs.index_axis(Axis(0), c);
            let (inv, logdet) = sym_inv_logdet(cov)?;
            let mean = means.row(c);
            let log_w = weights[c].max(1e-12).ln();
            for i in 0..n {
                let diff = &x.row(i) - &mean;
                // Mahalanobis distance: diff^T inv diff.
                let maha = diff.dot(&inv.dot(&diff));
                lp[[i, c]] = log_w - 0.5 * (const_term + logdet + maha);
            }
        }
        Ok(lp)
    }

    pub fn fit(&mut self, x: ArrayView2<f64>) -> Result<()> {
        let (n, p) = (x.nrows(), x.ncols());
        if n < self.k {
            return Err(MlError::InvalidParameter("n_samples must be >= k".into()));
        }

        // Initialise means from random samples, covs as identity, weights uniform.
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut idx: Vec<usize> = (0..n).collect();
        idx.shuffle(&mut rng);
        let mut means = Array2::<f64>::zeros((self.k, p));
        for c in 0..self.k {
            means.row_mut(c).assign(&x.row(idx[c]));
        }
        let mut covs = Array3::<f64>::zeros((self.k, p, p));
        for c in 0..self.k {
            for d in 0..p {
                covs[[c, d, d]] = 1.0;
            }
        }
        let mut weights = Array1::<f64>::from_elem(self.k, 1.0 / self.k as f64);

        self.means = Some(means);
        self.covs = Some(covs);
        self.weights = Some(weights.clone());

        let mut prev_ll = f64::NEG_INFINITY;
        for _ in 0..self.max_iter {
            // E-step: responsibilities via log-sum-exp.
            let lp = self.log_prob(x)?;
            let mut resp = Array2::<f64>::zeros((n, self.k));
            let mut ll = 0.0;
            for i in 0..n {
                let row = lp.row(i);
                let max = row.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let mut sum = 0.0;
                for c in 0..self.k {
                    let e = (row[c] - max).exp();
                    resp[[i, c]] = e;
                    sum += e;
                }
                for c in 0..self.k {
                    resp[[i, c]] /= sum;
                }
                ll += max + sum.ln();
            }

            // M-step.
            let nk = resp.sum_axis(Axis(0)); // length k
            let mut new_means = Array2::<f64>::zeros((self.k, p));
            for c in 0..self.k {
                let denom = nk[c].max(1e-12);
                for i in 0..n {
                    let r = resp[[i, c]];
                    let mut mrow = new_means.row_mut(c);
                    mrow.scaled_add(r, &x.row(i));
                }
                new_means.row_mut(c).mapv_inplace(|v| v / denom);
            }
            let mut new_covs = Array3::<f64>::zeros((self.k, p, p));
            for c in 0..self.k {
                let denom = nk[c].max(1e-12);
                let mean = new_means.row(c);
                for i in 0..n {
                    let r = resp[[i, c]];
                    let diff = &x.row(i) - &mean;
                    for a in 0..p {
                        for b in 0..p {
                            new_covs[[c, a, b]] += r * diff[a] * diff[b];
                        }
                    }
                }
                for a in 0..p {
                    for b in 0..p {
                        new_covs[[c, a, b]] /= denom;
                    }
                    // Regularise the diagonal for numerical stability.
                    new_covs[[c, a, a]] += self.reg;
                }
            }
            weights = &nk / n as f64;

            self.means = Some(new_means);
            self.covs = Some(new_covs);
            self.weights = Some(weights.clone());

            if (ll - prev_ll).abs() < self.tol {
                prev_ll = ll;
                break;
            }
            prev_ll = ll;
        }
        self.log_likelihood = prev_ll;

        // Hard labels from final responsibilities.
        let lp = self.log_prob(x)?;
        let labels = Array1::from_iter((0..n).map(|i| {
            lp.row(i)
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(c, _)| c)
                .unwrap()
        }));
        self.labels = Some(labels);
        Ok(())
    }

    /// Hard cluster assignment for new points.
    pub fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        if self.means.is_none() {
            return Err(MlError::NotFitted("GaussianMixture".into()));
        }
        let lp = self.log_prob(x)?;
        Ok(Array1::from_iter((0..x.nrows()).map(|i| {
            lp.row(i)
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(c, _)| c as f64)
                .unwrap()
        })))
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
    fn gmm_recovers_two_clusters() {
        let x = array![
            [0.0, 0.0],
            [0.3, -0.2],
            [-0.2, 0.1],
            [0.1, 0.2],
            [8.0, 8.0],
            [8.3, 7.8],
            [7.8, 8.1],
            [8.1, 8.2]
        ];
        let mut gmm = GaussianMixture::new(2, 100, 1e-4, 1e-6, 0).unwrap();
        let labels = gmm.fit_predict(x.view()).unwrap();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[4]);
    }
}
