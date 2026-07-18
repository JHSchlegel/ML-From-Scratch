//! Error type shared across the crate.

use std::fmt;

/// Errors returned by estimators when inputs are invalid or fitting fails.
#[derive(Debug, Clone, PartialEq)]
pub enum MlError {
    /// A dimension mismatch, e.g. `X` and `y` disagree on the sample count.
    ShapeMismatch(String),
    /// A hyper-parameter was set to an invalid value.
    InvalidParameter(String),
    /// `predict`/`transform` was called before `fit`.
    NotFitted(String),
    /// A numerical routine (decomposition, inversion, ...) failed.
    Numerical(String),
}

impl fmt::Display for MlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MlError::ShapeMismatch(m) => write!(f, "shape mismatch: {m}"),
            MlError::InvalidParameter(m) => write!(f, "invalid parameter: {m}"),
            MlError::NotFitted(m) => write!(f, "not fitted: {m}"),
            MlError::Numerical(m) => write!(f, "numerical error: {m}"),
        }
    }
}

impl std::error::Error for MlError {}

/// Convenient result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, MlError>;
