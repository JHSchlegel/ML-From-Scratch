#include "lin_reg.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <cmath>

LinReg::LinReg() {}


/**
 * @brief Fit the linear regression model to the data
 * 
 * @param X Feature matrix
 * @param y Target vector
 */
void LinReg::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    /*QR-decomposition of X (use ColPivHouseholderQR for numerical stability
    in case X is rank-deficient) and solve the normal equations to get the
    coefficients of the linear regression model (y = X * coefficients) 
    */
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(X);
    coefficients = qr.solve(y);
}

/**
 * @brief Predict the target values for new data points
 * 
 * @param X Feature matrix
 * @return Eigen::VectorXd Predicted target values
 */
Eigen::VectorXd LinReg::predict(const Eigen::MatrixXd& X) {
    return X * coefficients;
}


/**
 * @brief Get the coefficients of the linear regression model
 * 
 * @return Eigen::VectorXd Coefficients of the linear regression model
 */
Eigen::VectorXd LinReg::get_coefficients() const {
    return coefficients;
}