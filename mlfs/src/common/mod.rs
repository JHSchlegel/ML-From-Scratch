//! Cross-cutting building blocks: estimator traits, linear-algebra helpers,
//! metrics, preprocessing, label encoding and input validation.

pub mod distance;
pub mod labels;
pub mod linalg;
pub mod metrics;
pub mod preprocessing;
pub mod traits;
pub mod validation;

pub use traits::{Estimator, Predictor, Transformer};
