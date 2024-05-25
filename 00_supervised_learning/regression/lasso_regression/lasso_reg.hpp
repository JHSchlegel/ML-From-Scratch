#ifndef LassoReg_H
#define LassoReg_H

#include <iostream>
#include <Eigen/Dense>


/**
 * @brief LassoReg class for lasso regression
 * 
 */
class LassoReg {
public:
    LassoReg(
        double lambda = 1.0, 
        int max_iter = 1000,
        double tol = 1e-6
    ): lambda(lambda), max_iter(max_iter), tol(tol){};
    /**
     * @brief Fit the linear regression model to the data
     * 
     * @param X Feature matrix
     * @param y Target vector
     */
    void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
    /**
     * @brief Predict the target values for new data points
     * 
     * @param X Feature matrix
     * @return Eigen::VectorXd  Predicted target values
     */
    Eigen::VectorXd predict(const Eigen::MatrixXd& X);

    /**
     * @brief Get the coefficients of the linear regression model
     * 
     * @return Eigen::VectorXd Coefficients of the linear regression model
     */
    Eigen::VectorXd get_coefficients() const;

private:
    // Coefficients of the linear regression model
    Eigen::VectorXd coefficients;
    // Regularization parameter
    double lambda;
    // Maximum number of iterations for coordinate descent
    int max_iter;
    // Tolerance for stopping criterion
    double tol;

    /**
     * @brief Soft thresholding operator
     * 
     * @param rho Input value
     * @param lambda Regularization parameter
     * @return double Soft thresholded value
     */
    double soft_thresholding(double rho, double lambda);

};

#endif //LassoReg_H
