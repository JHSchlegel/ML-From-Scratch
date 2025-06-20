#ifndef RidgeReg_H
#define RidgeReg_H

#include <iostream>
#include <Eigen/Dense>


/**
 * @brief RidgeReg class for ridge regression
 * 
 */
class RidgeReg {
public:
    RidgeReg(double lambda = 1.0): lambda(lambda) {};
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
};

#endif //RidgeReg_H
