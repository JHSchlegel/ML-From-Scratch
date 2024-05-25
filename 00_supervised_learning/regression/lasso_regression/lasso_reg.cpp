#include "lasso_reg.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <cmath>


/**
 * @brief Fit the linear regression model to the data
 * 
 * @param X Feature matrix
 * @param y Target vector
 */
void LassoReg::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    std::cout << "Fitting Lasso regression model..." << std::endl;
    int n = X.cols();
    int m = X.rows();
    // Initialize coefficients to zero
    coefficients = Eigen::VectorXd::Zero(n);
    // Copy of the coefficients to check for convergence:
    Eigen::VectorXd old_coefficients = coefficients;


    // Standardize X:
    Eigen::VectorXd X_mean = X.colwise().mean();
    Eigen::VectorXd std = ((X.rowwise() - X_mean.transpose()).array().square().colwise().sum() / (m - 1)).sqrt();
    // Add 1e-12 to avoid division by zero
    Eigen::MatrixXd X_standardized = (X.rowwise() - X_mean.transpose()).array().rowwise() / (std.transpose().array() + 1e-12);

    for (int iter = 0; iter < max_iter; ++iter) {
        // Copy of the old coefficients to check for convergence:
        old_coefficients = coefficients;
        std::cout << "Iteration: " << iter << "/" << max_iter << std::endl;
        // Coordinate descent: iterate over coefficients
        for (int j = 0; j < n; ++j) {
            coefficients[j] = 0.0;
            // Calculate partial residuals
            Eigen::VectorXd residuals = y - X_standardized * coefficients;
            double rho = X_standardized.col(j).dot(residuals);
            // Update the intercept of the regression model (not regularized)
            if (j == 0){
                coefficients[j] = rho / (X_standardized.col(j).squaredNorm() + 1e-12);
            }
            else{
                // Update j'th coefficient with soft-thresholding
                coefficients[j] = soft_thresholding(rho, lambda) / (X_standardized.col(j).squaredNorm() + 1e-12);
            }
        }

        // Check for convergence
        if ((coefficients - old_coefficients).norm() < tol) {
            std::cout << "Converged after " << iter << " iterations" << std::endl;
            break;
        }
    }
}


/**
 * @brief Predict the target values for new data points
 * 
 * @param X Feature matrix
 * @return Eigen::VectorXd Predicted target values
 */
Eigen::VectorXd LassoReg::predict(const Eigen::MatrixXd& X) {
    int m = X.rows();
    //standardize X:
    Eigen::VectorXd mean = X.colwise().mean();
    Eigen::VectorXd std = ((X.rowwise() - mean.transpose()).array().square().colwise().sum() / (m - 1)).sqrt();
    //add 1e-12 to avoid division by zero
    Eigen::MatrixXd X_standardized = (X.rowwise() - mean.transpose()).array().rowwise() / (std.transpose().array() + 1e-12);

    return X_standardized * coefficients;
}


/**
 * @brief Get the coefficients of the linear regression model
 * 
 * @return Eigen::VectorXd Coefficients of the linear regression model
 */
Eigen::VectorXd LassoReg::get_coefficients() const {
    return coefficients;
}


/**
 * @brief Soft thresholding operator
 * 
 * @param rho Input value
 * @param lambda Regularization parameter
 * @return double Soft thresholded value
 */
double LassoReg::soft_thresholding(double rho, double lambda) {
    if (rho < -lambda) {
        return rho + lambda;
    } else if (rho > lambda) {
        return rho - lambda;
    } else {
        return 0;
    }
}