#ifndef LinReg_H
#define LinReg_H

#include <iostream>
#include <Eigen/Dense>


/**
 * @brief LinReg class for linear regression
 * 
 */
class LinReg {
public:
    LinReg();
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
};

#endif //LinReg_H
