//! Ensemble methods built on decision trees.

mod gradient_boosting;
mod random_forest;

pub use gradient_boosting::{GradientBoostingClassifier, GradientBoostingRegressor};
pub use random_forest::{RandomForestClassifier, RandomForestRegressor};
