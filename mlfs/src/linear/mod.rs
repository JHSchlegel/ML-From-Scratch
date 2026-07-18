//! Linear models.

mod lasso;
mod linear_regression;
mod ridge;

pub use lasso::Lasso;
pub use linear_regression::LinearRegression;
pub use ridge::Ridge;
