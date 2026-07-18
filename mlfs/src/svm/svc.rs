//! Support Vector Classifier trained with a simplified SMO solver.
//!
//! A binary solver ([`BinarySvc`]) handles the two-class dual problem with
//! linear or RBF kernels; the public [`SVC`] wraps it in a one-vs-rest scheme
//! for multi-class problems.

use crate::common::labels::LabelEncoder;
use crate::common::traits::{Estimator, Predictor};
use crate::common::validation::{check_n_features, check_xy};
use crate::error::{MlError, Result};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Kernel choice for the SVC.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Kernel {
    Linear,
    /// Gaussian RBF with bandwidth `gamma`.
    Rbf {
        gamma: f64,
    },
}

impl Kernel {
    fn eval(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
        match self {
            Kernel::Linear => a.dot(&b),
            Kernel::Rbf { gamma } => {
                let d2: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
                (-gamma * d2).exp()
            }
        }
    }
}

/// Binary SVC over labels in `{-1, +1}`.
#[derive(Debug, Clone)]
struct BinarySvc {
    kernel: Kernel,
    sv_x: Array2<f64>,
    sv_y: Array1<f64>,
    sv_alpha: Array1<f64>,
    b: f64,
}

impl BinarySvc {
    fn train(
        x: ArrayView2<f64>,
        y: &Array1<f64>, // in {-1, +1}
        c: f64,
        kernel: Kernel,
        tol: f64,
        max_passes: usize,
        rng: &mut StdRng,
    ) -> Self {
        let n = x.nrows();
        // Precompute the kernel (Gram) matrix.
        let mut k = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in i..n {
                let v = kernel.eval(x.row(i), x.row(j));
                k[[i, j]] = v;
                k[[j, i]] = v;
            }
        }

        let mut alpha = Array1::<f64>::zeros(n);
        let mut b = 0.0;

        // f(x_i) = sum_j alpha_j y_j K[j,i] + b
        let decision = |alpha: &Array1<f64>, b: f64, i: usize| -> f64 {
            let mut s = b;
            for j in 0..n {
                if alpha[j] != 0.0 {
                    s += alpha[j] * y[j] * k[[j, i]];
                }
            }
            s
        };

        let mut passes = 0;
        while passes < max_passes {
            let mut num_changed = 0;
            for i in 0..n {
                let e_i = decision(&alpha, b, i) - y[i];
                if (y[i] * e_i < -tol && alpha[i] < c) || (y[i] * e_i > tol && alpha[i] > 0.0) {
                    // Pick j != i at random.
                    let mut j = rng.gen_range(0..n);
                    if j == i {
                        j = (j + 1) % n;
                    }
                    let e_j = decision(&alpha, b, j) - y[j];
                    let (ai_old, aj_old) = (alpha[i], alpha[j]);

                    let (l, h) = if (y[i] - y[j]).abs() > 1e-12 {
                        ((aj_old - ai_old).max(0.0), c + (aj_old - ai_old).min(0.0))
                    } else {
                        ((ai_old + aj_old - c).max(0.0), (ai_old + aj_old).min(c))
                    };
                    if (h - l).abs() < 1e-12 {
                        continue;
                    }
                    let eta = 2.0 * k[[i, j]] - k[[i, i]] - k[[j, j]];
                    if eta >= 0.0 {
                        continue;
                    }
                    let mut aj_new = aj_old - y[j] * (e_i - e_j) / eta;
                    aj_new = aj_new.clamp(l, h);
                    if (aj_new - aj_old).abs() < 1e-5 {
                        continue;
                    }
                    let ai_new = ai_old + y[i] * y[j] * (aj_old - aj_new);
                    alpha[i] = ai_new;
                    alpha[j] = aj_new;

                    // Update the threshold b.
                    let b1 = b
                        - e_i
                        - y[i] * (ai_new - ai_old) * k[[i, i]]
                        - y[j] * (aj_new - aj_old) * k[[i, j]];
                    let b2 = b
                        - e_j
                        - y[i] * (ai_new - ai_old) * k[[i, j]]
                        - y[j] * (aj_new - aj_old) * k[[j, j]];
                    b = if ai_new > 0.0 && ai_new < c {
                        b1
                    } else if aj_new > 0.0 && aj_new < c {
                        b2
                    } else {
                        0.5 * (b1 + b2)
                    };
                    num_changed += 1;
                }
            }
            if num_changed == 0 {
                passes += 1;
            } else {
                passes = 0;
            }
        }

        // Keep only support vectors (alpha > 0).
        let sv: Vec<usize> = (0..n).filter(|&i| alpha[i] > 1e-8).collect();
        let sv_x = x.select(Axis(0), &sv);
        let sv_y = Array1::from_iter(sv.iter().map(|&i| y[i]));
        let sv_alpha = Array1::from_iter(sv.iter().map(|&i| alpha[i]));

        Self {
            kernel,
            sv_x,
            sv_y,
            sv_alpha,
            b,
        }
    }

    /// Signed decision value for a single sample.
    fn decision_value(&self, x: ArrayView1<f64>) -> f64 {
        let mut s = self.b;
        for i in 0..self.sv_alpha.len() {
            s += self.sv_alpha[i] * self.sv_y[i] * self.kernel.eval(self.sv_x.row(i), x);
        }
        s
    }
}

/// Multi-class Support Vector Classifier (one-vs-rest).
#[derive(Debug, Clone)]
pub struct SVC {
    c: f64,
    kernel: Kernel,
    tol: f64,
    max_passes: usize,
    seed: u64,
    encoder: Option<LabelEncoder>,
    models: Vec<BinarySvc>, // one per class (OvR); for 2 classes, a single model
    n_features: usize,
}

impl SVC {
    pub fn new(c: f64, kernel: Kernel, tol: f64, max_passes: usize, seed: u64) -> Result<Self> {
        if c <= 0.0 {
            return Err(MlError::InvalidParameter("C must be > 0".into()));
        }
        Ok(Self {
            c,
            kernel,
            tol,
            max_passes,
            seed,
            encoder: None,
            models: Vec::new(),
            n_features: 0,
        })
    }

    pub fn classes(&self) -> Option<&[f64]> {
        self.encoder.as_ref().map(|e| e.classes())
    }
}

impl Estimator for SVC {
    fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<()> {
        check_xy(x, y)?;
        let encoder = LabelEncoder::fit(y)?;
        let k = encoder.n_classes();
        if k < 2 {
            return Err(MlError::InvalidParameter(
                "SVC needs at least 2 classes".into(),
            ));
        }
        self.n_features = x.ncols();
        let y_idx = encoder.encode(y)?;
        let mut rng = StdRng::seed_from_u64(self.seed);

        self.models.clear();
        if k == 2 {
            // Single binary problem: class 1 = +1, class 0 = -1.
            let yb = Array1::from_iter(y_idx.iter().map(|&c| if c == 1 { 1.0 } else { -1.0 }));
            self.models.push(BinarySvc::train(
                x,
                &yb,
                self.c,
                self.kernel,
                self.tol,
                self.max_passes,
                &mut rng,
            ));
        } else {
            // One-vs-rest: class c = +1, all others = -1.
            for c in 0..k {
                let yb =
                    Array1::from_iter(y_idx.iter().map(|&ci| if ci == c { 1.0 } else { -1.0 }));
                self.models.push(BinarySvc::train(
                    x,
                    &yb,
                    self.c,
                    self.kernel,
                    self.tol,
                    self.max_passes,
                    &mut rng,
                ));
            }
        }
        self.encoder = Some(encoder);
        Ok(())
    }
}

impl Predictor for SVC {
    fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        if self.models.is_empty() {
            return Err(MlError::NotFitted("SVC".into()));
        }
        check_n_features(x, self.n_features)?;
        let encoder = self.encoder.as_ref().unwrap();

        let idx: Vec<usize> = x
            .rows()
            .into_iter()
            .map(|row| {
                if self.models.len() == 1 {
                    // Binary: positive decision -> class index 1.
                    if self.models[0].decision_value(row) >= 0.0 {
                        1
                    } else {
                        0
                    }
                } else {
                    // OvR: pick the class with the largest decision value.
                    self.models
                        .iter()
                        .map(|m| m.decision_value(row))
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                        .map(|(c, _)| c)
                        .unwrap()
                }
            })
            .collect();
        Ok(encoder.decode(&idx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::metrics::accuracy_score;
    use ndarray::array;

    #[test]
    fn linear_svc_separates() {
        let x = array![
            [-2.0, -2.0],
            [-2.1, -1.9],
            [-1.8, -2.2],
            [2.0, 2.0],
            [2.2, 1.8],
            [1.9, 2.1]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut m = SVC::new(1.0, Kernel::Linear, 1e-3, 5, 0).unwrap();
        m.fit(x.view(), y.view()).unwrap();
        let pred = m.predict(x.view()).unwrap();
        assert_eq!(accuracy_score(y.view(), pred.view()), 1.0);
    }

    #[test]
    fn rbf_svc_handles_nonlinear() {
        // Concentric-ish: inner points class 0, outer class 1.
        let x = array![
            [0.0, 0.0],
            [0.2, -0.1],
            [-0.1, 0.15],
            [3.0, 0.0],
            [-3.0, 0.0],
            [0.0, 3.0],
            [0.0, -3.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let mut m = SVC::new(1.0, Kernel::Rbf { gamma: 0.5 }, 1e-3, 10, 0).unwrap();
        m.fit(x.view(), y.view()).unwrap();
        let pred = m.predict(x.view()).unwrap();
        assert!(accuracy_score(y.view(), pred.view()) >= 0.85);
    }
}
