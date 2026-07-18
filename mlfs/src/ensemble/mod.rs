//! Ensemble methods built on decision trees.

mod adaboost;
mod gradient_boosting;
mod random_forest;

pub use adaboost::AdaBoostClassifier;
pub use gradient_boosting::{GradientBoostingClassifier, GradientBoostingRegressor};
pub use random_forest::{RandomForestClassifier, RandomForestRegressor};
