//! Linear models: OLS, ridge, lasso and (multinomial) logistic regression.

mod lasso;
mod linear_regression;
mod logistic;
mod ridge;

pub use lasso::Lasso;
pub use linear_regression::LinearRegression;
pub use logistic::LogisticRegression;
pub use ridge::Ridge;
