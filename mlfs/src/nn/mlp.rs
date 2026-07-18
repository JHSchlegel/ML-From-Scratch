//! Multi-layer perceptron (feed-forward neural network) with backpropagation
//! and the Adam optimizer. A single core network powers both the classifier
//! (softmax + cross-entropy) and the regressor (linear + squared error).

use crate::common::labels::LabelEncoder;
use crate::common::traits::{Estimator, Predictor};
use crate::common::validation::{check_n_features, check_xy};
use crate::error::{MlError, Result};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

/// Hidden-layer activation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    Relu,
    Tanh,
}

impl Activation {
    fn apply(&self, z: &Array2<f64>) -> Array2<f64> {
        match self {
            Activation::Relu => z.mapv(|v| v.max(0.0)),
            Activation::Tanh => z.mapv(|v| v.tanh()),
        }
    }
    /// Derivative given the pre-activation `z`.
    fn deriv(&self, z: &Array2<f64>) -> Array2<f64> {
        match self {
            Activation::Relu => z.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 }),
            Activation::Tanh => z.mapv(|v| 1.0 - v.tanh().powi(2)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Objective {
    /// Softmax output + cross-entropy loss.
    Classification,
    /// Linear output + squared-error loss.
    Regression,
}

fn softmax_rows(z: &Array2<f64>) -> Array2<f64> {
    let mut out = z.clone();
    for mut row in out.rows_mut() {
        let max = row.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        row.mapv_inplace(|v| (v - max).exp());
        let s: f64 = row.sum();
        if s > 0.0 {
            row /= s;
        }
    }
    out
}

/// Adam moment buffers for a single parameter tensor.
struct Adam {
    m: Array2<f64>,
    v: Array2<f64>,
}
struct AdamVec {
    m: Array1<f64>,
    v: Array1<f64>,
}

/// Core network: fully-connected layers with one shared hidden activation.
struct Net {
    weights: Vec<Array2<f64>>,
    biases: Vec<Array1<f64>>,
    activation: Activation,
    objective: Objective,
}

impl Net {
    fn new(
        sizes: &[usize],
        activation: Activation,
        objective: Objective,
        rng: &mut StdRng,
    ) -> Self {
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        for l in 0..sizes.len() - 1 {
            let (fan_in, fan_out) = (sizes[l], sizes[l + 1]);
            // He initialisation for ReLU, Xavier-ish otherwise.
            let std = match activation {
                Activation::Relu => (2.0 / fan_in as f64).sqrt(),
                Activation::Tanh => (1.0 / fan_in as f64).sqrt(),
            };
            let w = Array2::random_using((fan_in, fan_out), Normal::new(0.0, std).unwrap(), rng);
            weights.push(w);
            biases.push(Array1::zeros(fan_out));
        }
        Self {
            weights,
            biases,
            activation,
            objective,
        }
    }

    /// Forward pass returning per-layer pre-activations `z` and activations `a`
    /// (a[0] is the input, a.last() is the network output).
    fn forward(&self, x: ArrayView2<f64>) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
        let mut a = vec![x.to_owned()];
        let mut zs = Vec::new();
        let n_layers = self.weights.len();
        for l in 0..n_layers {
            let mut z = a[l].dot(&self.weights[l]);
            z += &self.biases[l].view().insert_axis(Axis(0));
            let out = if l == n_layers - 1 {
                match self.objective {
                    Objective::Classification => softmax_rows(&z),
                    Objective::Regression => z.clone(),
                }
            } else {
                self.activation.apply(&z)
            };
            zs.push(z);
            a.push(out);
        }
        (zs, a)
    }

    fn output(&self, x: ArrayView2<f64>) -> Array2<f64> {
        self.forward(x).1.pop().unwrap()
    }
}

/// Feed-forward neural network estimator (shared by classifier and regressor).
struct Mlp {
    hidden: Vec<usize>,
    activation: Activation,
    objective: Objective,
    lr: f64,
    max_iter: usize,
    batch_size: usize,
    l2: f64,
    seed: u64,
    net: Option<Net>,
    n_features: usize,
    n_outputs: usize,
}

impl Mlp {
    fn fit_core(&mut self, x: ArrayView2<f64>, y_target: &Array2<f64>) -> Result<()> {
        let n = x.nrows();
        self.n_features = x.ncols();
        self.n_outputs = y_target.ncols();

        let mut sizes = vec![self.n_features];
        sizes.extend_from_slice(&self.hidden);
        sizes.push(self.n_outputs);

        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut net = Net::new(&sizes, self.activation, self.objective, &mut rng);

        // Adam state.
        let (beta1, beta2, eps): (f64, f64, f64) = (0.9, 0.999, 1e-8);
        let mut mw: Vec<Adam> = net
            .weights
            .iter()
            .map(|w| Adam {
                m: Array2::zeros(w.raw_dim()),
                v: Array2::zeros(w.raw_dim()),
            })
            .collect();
        let mut mb: Vec<AdamVec> = net
            .biases
            .iter()
            .map(|b| AdamVec {
                m: Array1::zeros(b.raw_dim()),
                v: Array1::zeros(b.raw_dim()),
            })
            .collect();

        let batch = self.batch_size.min(n).max(1);
        let mut t = 0i32;

        for _ in 0..self.max_iter {
            let mut order: Vec<usize> = (0..n).collect();
            order.shuffle(&mut rng);
            for chunk in order.chunks(batch) {
                let xb = x.select(Axis(0), chunk);
                let yb = y_target.select(Axis(0), chunk);
                let bn = chunk.len() as f64;

                let (zs, a) = net.forward(xb.view());
                let n_layers = net.weights.len();

                // Output-layer error (softmax-CE and linear-MSE share this form).
                let mut delta = (a[n_layers].clone() - &yb) / bn;

                // Backpropagate.
                let mut grad_w: Vec<Array2<f64>> = vec![Array2::zeros((0, 0)); n_layers];
                let mut grad_b: Vec<Array1<f64>> = vec![Array1::zeros(0); n_layers];
                for l in (0..n_layers).rev() {
                    let gw = a[l].t().dot(&delta) + &(&net.weights[l] * self.l2);
                    let gb = delta.sum_axis(Axis(0));
                    grad_w[l] = gw;
                    grad_b[l] = gb;
                    if l > 0 {
                        let back = delta.dot(&net.weights[l].t());
                        delta = back * self.activation.deriv(&zs[l - 1]);
                    }
                }

                // Adam update.
                t += 1;
                let bc1 = 1.0 - beta1.powi(t);
                let bc2 = 1.0 - beta2.powi(t);
                for l in 0..n_layers {
                    mw[l].m = &mw[l].m * beta1 + &(&grad_w[l] * (1.0 - beta1));
                    mw[l].v = &mw[l].v * beta2 + &(grad_w[l].mapv(|g| g * g) * (1.0 - beta2));
                    let mhat = &mw[l].m / bc1;
                    let vhat = &mw[l].v / bc2;
                    net.weights[l] =
                        &net.weights[l] - &(self.lr * &mhat / (vhat.mapv(f64::sqrt) + eps));

                    mb[l].m = &mb[l].m * beta1 + &(&grad_b[l] * (1.0 - beta1));
                    mb[l].v = &mb[l].v * beta2 + &(grad_b[l].mapv(|g| g * g) * (1.0 - beta2));
                    let mhb = &mb[l].m / bc1;
                    let vhb = &mb[l].v / bc2;
                    net.biases[l] =
                        &net.biases[l] - &(self.lr * &mhb / (vhb.mapv(f64::sqrt) + eps));
                }
            }
        }
        self.net = Some(net);
        Ok(())
    }
}

/// MLP classifier (softmax output, cross-entropy loss).
pub struct MLPClassifier {
    mlp: Mlp,
    encoder: Option<LabelEncoder>,
}

impl MLPClassifier {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden: Vec<usize>,
        activation: Activation,
        lr: f64,
        max_iter: usize,
        batch_size: usize,
        l2: f64,
        seed: u64,
    ) -> Self {
        Self {
            mlp: Mlp {
                hidden,
                activation,
                objective: Objective::Classification,
                lr,
                max_iter,
                batch_size,
                l2,
                seed,
                net: None,
                n_features: 0,
                n_outputs: 0,
            },
            encoder: None,
        }
    }

    pub fn classes(&self) -> Option<&[f64]> {
        self.encoder.as_ref().map(|e| e.classes())
    }
}

impl Estimator for MLPClassifier {
    fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<()> {
        check_xy(x, y)?;
        let encoder = LabelEncoder::fit(y)?;
        let k = encoder.n_classes();
        let y_idx = encoder.encode(y)?;
        let mut y_oh = Array2::<f64>::zeros((x.nrows(), k));
        for (i, &c) in y_idx.iter().enumerate() {
            y_oh[[i, c]] = 1.0;
        }
        self.mlp.fit_core(x, &y_oh)?;
        self.encoder = Some(encoder);
        Ok(())
    }
}

impl Predictor for MLPClassifier {
    fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        let net = self
            .mlp
            .net
            .as_ref()
            .ok_or_else(|| MlError::NotFitted("MLPClassifier".into()))?;
        check_n_features(x, self.mlp.n_features)?;
        let out = net.output(x);
        let idx: Vec<usize> = out
            .rows()
            .into_iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap()
            })
            .collect();
        Ok(self.encoder.as_ref().unwrap().decode(&idx))
    }
}

/// MLP regressor (linear output, squared-error loss). Single-target.
pub struct MLPRegressor {
    mlp: Mlp,
}

impl MLPRegressor {
    pub fn new(
        hidden: Vec<usize>,
        activation: Activation,
        lr: f64,
        max_iter: usize,
        batch_size: usize,
        l2: f64,
        seed: u64,
    ) -> Self {
        Self {
            mlp: Mlp {
                hidden,
                activation,
                objective: Objective::Regression,
                lr,
                max_iter,
                batch_size,
                l2,
                seed,
                net: None,
                n_features: 0,
                n_outputs: 0,
            },
        }
    }
}

impl Estimator for MLPRegressor {
    fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<()> {
        check_xy(x, y)?;
        let y_target = y.to_owned().insert_axis(Axis(1));
        self.mlp.fit_core(x, &y_target)
    }
}

impl Predictor for MLPRegressor {
    fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        let net = self
            .mlp
            .net
            .as_ref()
            .ok_or_else(|| MlError::NotFitted("MLPRegressor".into()))?;
        check_n_features(x, self.mlp.n_features)?;
        Ok(net.output(x).column(0).to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::metrics::{accuracy_score, r2_score};
    use ndarray::{array, Array, Array2};
    use ndarray_rand::rand_distr::Uniform;

    #[test]
    fn mlp_classifier_learns_xor() {
        // XOR is the classic non-linear test a hidden layer can solve.
        let x = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let y = array![0.0, 1.0, 1.0, 0.0];
        let mut m = MLPClassifier::new(vec![8], Activation::Tanh, 0.1, 2000, 4, 0.0, 0);
        m.fit(x.view(), y.view()).unwrap();
        let pred = m.predict(x.view()).unwrap();
        assert_eq!(accuracy_score(y.view(), pred.view()), 1.0);
    }

    #[test]
    fn mlp_regressor_fits_nonlinear() {
        let x: Array2<f64> = Array::random((200, 1), Uniform::new(-2.0, 2.0));
        let y = x.column(0).mapv(|v| (2.0 * v).sin());
        let mut m = MLPRegressor::new(vec![32, 32], Activation::Tanh, 0.01, 1500, 32, 0.0, 0);
        m.fit(x.view(), y.view()).unwrap();
        let pred = m.predict(x.view()).unwrap();
        assert!(r2_score(y.view(), pred.view()) > 0.9);
    }
}
