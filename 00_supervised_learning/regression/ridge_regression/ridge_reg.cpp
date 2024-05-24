#include "ridge_reg.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <cmath>


/**
 * @brief Fit the linear regression model to the data
 * 
 * @param X Feature matrix
 * @param y Target vector
 */
void RidgeReg::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    /*QR-decomposition of X (use ColPivHouseholderQR for numerical stability
    in case X is rank-deficient) and solve the normal equations to get the
    coefficients of the linear regression model (y = X * coefficients) 
    */

    //beta = (X^T * X + lambda * I)^-1 * X^T * y
    //define X^T * X and X^T * y:
    Eigen::MatrixXd XTX = X.transpose() * X;
    Eigen::MatrixXd XTy = X.transpose() * y;

    // Add the regularization term to the normal equations
    int n = X.cols();
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);

    //numerically stable QR decomposition:
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(XTX + lambda * I);
    
    //finally, solve the normal equations to get the coefficients
    coefficients = qr.solve(XTy);
}

/**
 * @brief Predict the target values for new data points
 * 
 * @param X Feature matrix
 * @return Eigen::VectorXd Predicted target values
 */
Eigen::VectorXd RidgeReg::predict(const Eigen::MatrixXd& X) {
    return X * coefficients;
}


/**
 * @brief Get the coefficients of the linear regression model
 * 
 * @return Eigen::VectorXd Coefficients of the linear regression model
 */
Eigen::VectorXd RidgeReg::get_coefficients() const {
    return coefficients;
}