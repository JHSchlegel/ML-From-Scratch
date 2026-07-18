//! Scoring metrics for regression and classification.

use ndarray::ArrayView1;

/// Mean squared error.
pub fn mean_squared_error(y_true: ArrayView1<f64>, y_pred: ArrayView1<f64>) -> f64 {
    let n = y_true.len().max(1) as f64;
    y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| (t - p).powi(2))
        .sum::<f64>()
        / n
}

/// Coefficient of determination R^2.
pub fn r2_score(y_true: ArrayView1<f64>, y_pred: ArrayView1<f64>) -> f64 {
    let mean = y_true.mean().unwrap_or(0.0);
    let ss_res: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| (t - p).powi(2))
        .sum();
    let ss_tot: f64 = y_true.iter().map(|t| (t - mean).powi(2)).sum();
    if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    }
}

/// Fraction of correctly classified samples (labels compared exactly).
pub fn accuracy_score(y_true: ArrayView1<f64>, y_pred: ArrayView1<f64>) -> f64 {
    if y_true.is_empty() {
        return 0.0;
    }
    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(t, p)| (*t - *p).abs() < 1e-9)
        .count();
    correct as f64 / y_true.len() as f64
}
