//! Thin wrappers over `linfa-linalg` decompositions.
//!
//! These are the only "borrowed" linear-algebra primitives in the crate.
//! Every ML algorithm is built on top of these + plain `ndarray` arithmetic.

use crate::error::{MlError, Result};
use linfa_linalg::svd::SVD;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Solve the least-squares problem `min_beta ||X beta - y||_2` via the SVD.
///
/// Using the SVD makes this robust to rank-deficient / collinear `X`
/// (small singular values below tolerance are dropped).
pub fn lstsq(x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<Array1<f64>> {
    let (n, p) = (x.nrows(), x.ncols());
    if y.len() != n {
        return Err(MlError::ShapeMismatch(format!(
            "X has {n} rows but y has length {}",
            y.len()
        )));
    }
    // X = U * diag(s) * V^T
    let (u, s, vt) = x
        .to_owned()
        .svd(true, true)
        .map_err(|e| MlError::Numerical(format!("SVD failed: {e:?}")))?;
    let u = u.ok_or_else(|| MlError::Numerical("SVD returned no U".into()))?;
    let vt = vt.ok_or_else(|| MlError::Numerical("SVD returned no V^T".into()))?;

    let k = s.len();
    let mut beta = Array1::<f64>::zeros(p);
    if k == 0 {
        return Ok(beta);
    }
    // Relative tolerance for treating a singular value as zero.
    let tol = s[0] * (n.max(p) as f64) * f64::EPSILON;
    for i in 0..k {
        if s[i] > tol {
            // beta += (u_i . y / s_i) * v_i
            let coef = u.column(i).dot(&y) / s[i];
            beta.scaled_add(coef, &vt.row(i));
        }
    }
    Ok(beta)
}

/// Inverse and log-determinant of a symmetric positive-definite matrix,
/// computed from its SVD (eigenvalues of an SPD matrix are its singular values).
/// Used by the Gaussian mixture model for the multivariate normal density.
pub fn sym_inv_logdet(a: ArrayView2<f64>) -> Result<(Array2<f64>, f64)> {
    let p = a.nrows();
    let (u, s, vt) = a
        .to_owned()
        .svd(true, true)
        .map_err(|e| MlError::Numerical(format!("SVD failed: {e:?}")))?;
    let u = u.ok_or_else(|| MlError::Numerical("SVD returned no U".into()))?;
    let vt = vt.ok_or_else(|| MlError::Numerical("SVD returned no V^T".into()))?;

    let mut inv = Array2::<f64>::zeros((p, p));
    let mut logdet = 0.0;
    let floor = 1e-12;
    for i in 0..s.len() {
        let si = s[i].max(floor);
        logdet += si.ln();
        let inv_s = 1.0 / si;
        // inv += (1/s_i) v_i u_i^T  (for symmetric PSD, U and V align up to sign)
        let v_i = vt.row(i);
        let u_i = u.column(i);
        for r in 0..p {
            for c in 0..p {
                inv[[r, c]] += inv_s * v_i[r] * u_i[c];
            }
        }
    }
    Ok((inv, logdet))
}

/// Moore-Penrose pseudo-inverse of `a` via the SVD.
#[allow(dead_code)]
pub fn pinv(a: ArrayView2<f64>) -> Result<Array2<f64>> {
    let (n, p) = (a.nrows(), a.ncols());
    let (u, s, vt) = a
        .to_owned()
        .svd(true, true)
        .map_err(|e| MlError::Numerical(format!("SVD failed: {e:?}")))?;
    let u = u.ok_or_else(|| MlError::Numerical("SVD returned no U".into()))?;
    let vt = vt.ok_or_else(|| MlError::Numerical("SVD returned no V^T".into()))?;

    let k = s.len();
    let mut out = Array2::<f64>::zeros((p, n));
    if k == 0 {
        return Ok(out);
    }
    let tol = s[0] * (n.max(p) as f64) * f64::EPSILON;
    // pinv(A) = V * diag(1/s) * U^T = sum_i (1/s_i) v_i u_i^T
    for i in 0..k {
        if s[i] > tol {
            let inv = 1.0 / s[i];
            let v_i = vt.row(i); // length p
            let u_i = u.column(i); // length n
            for r in 0..p {
                for c in 0..n {
                    out[[r, c]] += inv * v_i[r] * u_i[c];
                }
            }
        }
    }
    Ok(out)
}
