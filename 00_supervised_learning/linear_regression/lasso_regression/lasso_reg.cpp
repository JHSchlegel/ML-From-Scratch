#include "lasso_reg.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <iostream>

/**
 * @brief Fit the linear regression model to the data
 *
 * @param X Feature matrix
 * @param y Target vector
 */
void LassoReg::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y)
{
    std::cout << "Fitting Lasso regression model..." << std::endl;
    int n = X.cols();
    int m = X.rows();
    // Initialize coefficients to zero
    coefficients = Eigen::VectorXd::Zero(n);
    // Copy of the coefficients to check for convergence:
    Eigen::VectorXd old_coefficients = coefficients;

    // Standardize X:
    X_mean = X.colwise().mean();
    X_std =
        ((X.rowwise() - X_mean.transpose()).array().square().colwise().sum() /
         (m - 1))
            .sqrt();
    y_mean = y.mean();
    y_std = sqrt((y.array() - y_mean).square().sum() / (m - 1));
    // Add 1e-12 to avoid division by zero
    Eigen::MatrixXd X_standardized =
        (X.rowwise() - X_mean.transpose()).array().rowwise() /
        (X_std.transpose().array() + 1e-12);
    Eigen::VectorXd y_standardized =
        (y.array() - y_mean) / (y_std + 1e-12);

    for (int iter = 0; iter < max_iter; ++iter)
    {
        // Copy of the old coefficients to check for convergence:
        old_coefficients = coefficients;
        std::cout << "Iteration: " << iter << "/" << max_iter << std::endl;
        // Coordinate descent: iterate over coefficients
        for (int j = 0; j < n; ++j)
        {
            coefficients[j] = 0.0;
            
            //Calculate partial residuals
            Eigen::VectorXd residuals = y_standardized -
                                        X_standardized * coefficients;
            //Calculate correlation between j-th feature and residuals
            double rho = X_standardized.col(j).dot(residuals);

            //Apply soft thresholding
            coefficients[j] = soft_thresholding(rho, lambda) /
                              (X_standardized.col(j).squaredNorm() + 1e-12);
        }

        // Check for convergence
        if ((coefficients - old_coefficients).norm() < tol)
        {
            std::cout << "Converged after " << iter << " iterations"
                      << std::endl;
            break;
        }
    }
    //Scale coefficients back to original scale:
    // beta_original[j] = beta_standardized[j] * (y_std / X_std[j]);
    for (int j = 0; j < n; ++j)
    {
        coefficients[j] *= (y_std / (X_std[j] + 1e-12));
    }
    // Calculate intercept on original scale:
    intercept = y_mean;
    for (int j = 0; j < n; ++j) {
        intercept -= X_mean[j] * coefficients[j];
    }
}

/**
 * @brief Predict the target values for new data points
 *
 * @param X Feature matrix
 * @return Eigen::VectorXd Predicted target values
 */
Eigen::VectorXd LassoReg::predict(const Eigen::MatrixXd& X)
{
    return X * coefficients + Eigen::VectorXd::Constant(X.rows(), intercept);
}

/**
 * @brief Get the coefficients of the linear regression model
 *
 * @return Eigen::VectorXd Coefficients of the linear regression model
 */
Eigen::VectorXd LassoReg::get_coefficients() const
{
    return coefficients;
}

/**
 * @brief Soft thresholding operator
 *
 * @param rho Input value
 * @param lambda Regularization parameter
 * @return double Soft thresholded value
 */
double LassoReg::soft_thresholding(double rho, double lambda)
{
    if (rho < -lambda)
    {
        return rho + lambda;
    }
    else if (rho > lambda)
    {
        return rho - lambda;
    }
    else
    {
        return 0;
    }
}